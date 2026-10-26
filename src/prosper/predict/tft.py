"""Temporal Fusion Transformer predictions using pytorch-forecasting."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DIRECTION_CLASSES,
    IDX_TO_DIR,
    parse_depth_bins,
)
from prosper.labels.forward import forward_labels
from prosper.predict.calibration import (
    TemperatureCalibrator,
    calibrated_nll,
    calibration_split,
    fit_temperature,
    usable_split,
)
from prosper.predict.defaults import (
    DEFAULT_EPOCH_BUDGET,
    DEFAULT_TRAIN_WINDOW_DAYS,
    EARLY_STOPPING_PATIENCE,
    MIN_EARLY_STOPPING_SAMPLES,
    parse_epoch_overrides,
    resolve_epochs,
)
from prosper.predict.sequence import SequenceForecaster, load_sequence
from prosper.predict.walk_forward import (
    Month,
    RunPlan,
    empirical_depth,
    plan_run,
    run,
    seed_run,
)


def _robust_normalize(X: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    X_safe = np.where(np.isfinite(X), X, 0.0)
    X_norm = (X_safe - med) / iqr
    return np.clip(np.nan_to_num(X_norm, nan=0.0, posinf=3.0, neginf=-3.0), -5.0, 5.0)


def _class_columns(dataset_ref: Any) -> list[int]:
    """Output-column index of each direction class, in `DIRECTION_CLASSES` order.

    This must be read from the encoder and never assumed. `NaNLabelEncoder`
    numbers its classes by sorting the labels, so with `add_nan=True` it yields
    ``{'nan': 0, 'long': 1, 'short': 2}`` - alphabetical - while
    `DIRECTION_CLASSES` is ordered ``['short', 'long']`` because the sign of the
    return defines it. Dropping the leading column and reading the rest
    positionally therefore lined `short` up with the model's `long` logit and
    vice versa: every TFT run written before this was fixed carries the two
    probabilities exchanged, so its edge was the negation of what the model had
    learned. A synthetic series whose direction is a deterministic function of
    the previous bar scored 0.013 agreement where it should have scored 0.987.
    """
    normalizer = getattr(dataset_ref, "target_normalizer", None)
    mapping = getattr(normalizer, "classes_", None)
    if not mapping:
        raise ValueError(
            "The TFT dataset exposes no target_normalizer.classes_, so the "
            "direction columns cannot be identified."
        )
    missing = [name for name in DIRECTION_CLASSES if name not in mapping]
    if missing:
        raise ValueError(f"Direction class(es) {missing} absent from the encoder: {dict(mapping)}")
    return [int(mapping[name]) for name in DIRECTION_CLASSES]


def _logits_to_probs(logits: np.ndarray, columns: list[int]) -> tuple[float, ...]:
    """Softmax a raw output row into one probability per direction class.

    *columns* comes from `_class_columns`. Selecting after the softmax and
    renormalising is equivalent to softmaxing the selected logits alone, so the
    `nan` column simply drops out.
    """
    row = np.asarray(logits).reshape(-1)
    shifted = row - np.max(row)
    exp = np.exp(shifted)
    picked = np.array([exp[c] for c in columns], dtype=float)
    total = picked.sum()
    if not np.isfinite(total) or total <= 0:
        return tuple([1.0 / len(columns)] * len(columns))
    return tuple(float(v) for v in picked / total)


def _unwrap_prediction(raw: Any) -> Any:
    """Pull the prediction tensor out of whatever pytorch-forecasting returned.

    The shape of `predict(mode="raw")` has changed across versions: a dict, a
    tuple whose first element is a dict or a tensor, or an object exposing
    `.output.prediction` / `.prediction`.
    """
    if isinstance(raw, dict):
        return raw["prediction"]
    if isinstance(raw, tuple):
        head = raw[0]
        if isinstance(head, dict):
            return head["prediction"]
        return getattr(head, "prediction", head)
    if hasattr(raw, "output"):
        return raw.output.prediction
    if hasattr(raw, "prediction"):
        return raw.prediction
    return raw


CALIBRATED_VAL_METRIC = "val_calibrated_nll"


def _calibrated_val_loss_callback(val_dataset: Any) -> Any:
    """Callback publishing the validation loss that survives calibration.

    Lightning's own `val_loss` is raw cross-entropy, and stopping on it is
    wrong for the same reason it is wrong for the GRU: it charges for
    overconfidence, the network becomes overconfident well before it stops
    discriminating, and `_fit_tft_calibrator` removes that overconfidence
    immediately afterwards. See `calibrated_nll`.

    The value is written straight into `trainer.callback_metrics`, which is
    where `EarlyStopping` reads from, so no logger has to be enabled to carry
    it. Failing to compute it is not fatal: the metric is simply absent and the
    paired `EarlyStopping(strict=False)` degrades to running the full budget.
    """
    import torch
    from lightning.pytorch.callbacks import Callback

    columns = _class_columns(val_dataset)
    # The dataloader hands back the encoder's own class index; invert the
    # column map to get the position in `DIRECTION_CLASSES` that indexes the
    # probability vector `_logits_to_probs` returns.
    to_direction_idx = {column: position for position, column in enumerate(columns)}

    class _CalibratedValLoss(Callback):
        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            loaders = trainer.val_dataloaders
            if loaders is None:
                return
            if not isinstance(loaders, list | tuple):
                loaders = [loaders]

            probs: list[tuple[float, ...]] = []
            labels: list[int] = []
            was_training = pl_module.training
            pl_module.eval()
            try:
                with torch.no_grad():
                    for loader in loaders:
                        for batch in loader:
                            x, y = batch[0], batch[1]
                            target = y[0] if isinstance(y, list | tuple) else y
                            target = np.asarray(target.detach().cpu()).reshape(-1)
                            logits = np.asarray(
                                _unwrap_prediction(pl_module(x)).detach().cpu()
                            )
                            for position, encoded in enumerate(target):
                                direction = to_direction_idx.get(int(encoded))
                                if direction is None:
                                    continue  # the encoder's `nan` class
                                probs.append(_logits_to_probs(logits[position, 0], columns))
                                labels.append(direction)
            except Exception:
                return
            finally:
                if was_training:
                    pl_module.train()

            if len(labels) < MIN_EARLY_STOPPING_SAMPLES:
                return
            value = calibrated_nll(np.array(probs, dtype=float), labels)
            if not np.isfinite(value):
                return
            trainer.callback_metrics[CALIBRATED_VAL_METRIC] = torch.tensor(float(value))

    return _CalibratedValLoss()


def _predict_month(
    model: Any,
    dataset_ref: Any,
    dataset_cls: Any,
    *,
    X_norm: np.ndarray,
    y_dir: np.ndarray,
    feature_cols: list[str],
    symbol: str,
    bar_indices: list[int],
    seq_len: int,
    forward_steps: int,
    cutoff: int,
    trainer_kwargs: dict[str, Any] | None = None,
) -> dict[int, tuple[float, ...]]:
    """Predict every bar of a retraining month in one batched pass.

    Building a TimeSeriesDataSet and a DataLoader per bar would dominate the
    runtime: a full run would take hours, almost none of it spent in the model.

    The target history is masked at *cutoff*, the first bar of the month. The
    model was fitted on information available at that point, so using the same
    horizon for its inference keeps the two consistent and cannot leak: labels
    needing a close at or after `cutoff` are hidden for every bar in the batch.
    """
    if not bar_indices:
        return {}

    first, last = min(bar_indices), max(bar_indices)
    lo = max(0, first - seq_len)

    rows = []
    for k in range(lo, last + 1):
        known = k + forward_steps < cutoff and y_dir[k] >= 0
        row = {
            "time_idx": k,
            "group": symbol,
            "target": IDX_TO_DIR[int(y_dir[k])] if known else np.nan,
        }
        for fi, fc in enumerate(feature_cols):
            row[fc] = float(X_norm[k, fi])
        rows.append(row)

    frame = pd.DataFrame(rows)
    inference_set = dataset_cls.from_dataset(
        dataset_ref, frame, predict=False, stop_randomization=True
    )
    loader = inference_set.to_dataloader(train=False, batch_size=64, num_workers=0)

    # `predict` builds its own Trainer when none is configured, and that one
    # defaults to a TensorBoard logger writing `lightning_logs/version_N` in
    # the CWD - the flags on the *training* Trainer do not reach it. Batching
    # cuts the call count from one per bar to one per month, which would hide
    # such a leak rather than close it.
    #
    # The copy is not optional. pytorch-forecasting does
    #   trainer_kwargs.setdefault("callbacks", ... + [predict_callback])
    # on the dict it is handed, so a shared dict keeps the callback from the
    # first call: every later call builds a Trainer whose collector is the
    # previous one, `predict_callback.result` comes back empty, and the horizon
    # silently degrades to untrained.
    result = model.predict(
        loader,
        mode="raw",
        return_index=True,
        trainer_kwargs=dict(trainer_kwargs or {}),
    )
    # pytorch-forecasting >=1.x returns a Prediction namedtuple
    # (output, x, index, decoder_lengths, y); older versions returned a plain
    # (output, index) tuple - which cannot be unpacked into two names. Test for
    # `output`, not `index`: every tuple has an `.index` *method*, so that check
    # is true for both shapes and would send the old one down the wrong branch.
    if hasattr(result, "output"):
        raw, index = result.output, result.index
    else:
        raw, index = result[0], result[1]
    predictions = _unwrap_prediction(raw)

    # Ask the fitted encoder which column belongs to which direction rather
    # than inferring it from position; see `_class_columns`.
    columns = _class_columns(inference_set)

    wanted = set(bar_indices)
    out: dict[int, tuple[float, ...]] = {}
    for position, decoder_start in enumerate(index["time_idx"].tolist()):
        bar = int(decoder_start)
        if bar in wanted:
            out[bar] = _logits_to_probs(predictions[position, 0].cpu().numpy(), columns)
    return out


def _fit_tft_calibrator(
    model: Any,
    dataset_ref: Any,
    dataset_cls: Any,
    *,
    X_norm: np.ndarray,
    y_dir: np.ndarray,
    feature_cols: list[str],
    symbol: str,
    bar_indices: list[int],
    seq_len: int,
    forward_steps: int,
    trainer_kwargs: dict[str, Any] | None = None,
) -> TemperatureCalibrator:
    """Score the held-out trainable bars and fit a temperature on the result.

    The cutoff is the first held-out bar, so the encoder cannot read a label
    from the very region being scored - the same masking rule inference uses.
    """
    if not bar_indices:
        return TemperatureCalibrator(1.0)

    try:
        scored = _predict_month(
            model,
            dataset_ref,
            dataset_cls,
            X_norm=X_norm,
            y_dir=y_dir,
            feature_cols=feature_cols,
            symbol=symbol,
            bar_indices=bar_indices,
            seq_len=seq_len,
            forward_steps=forward_steps,
            cutoff=min(bar_indices),
            trainer_kwargs=trainer_kwargs,
        )
    except Exception:
        # Calibration is an improvement, never a prerequisite for a forecast.
        return TemperatureCalibrator(1.0)

    bars = [bar for bar in bar_indices if bar in scored and y_dir[bar] >= 0]
    if not bars:
        return TemperatureCalibrator(1.0)

    probs = np.array([scored[bar] for bar in bars], dtype=float)
    labels = np.array([int(y_dir[bar]) for bar in bars], dtype=int)
    return fit_temperature(probs, labels)


class TFTForecaster(SequenceForecaster):
    """A Temporal Fusion Transformer per horizon, predicting a month in one pass."""

    name = "tft"

    def __init__(
        self,
        *,
        feature_cols: list[str],
        forecasting: dict[str, Any],
        hidden_size: int,
        attention_head_size: int,
        learning_rate: float,
        inference_kwargs: dict[str, Any],
        **common: Any,
    ) -> None:
        super().__init__(**common)
        self.feature_cols = feature_cols
        # Lightning and pytorch-forecasting, imported by the caller so a
        # missing install is an error message rather than an import failure.
        self.lib = forecasting
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.learning_rate = learning_rate
        self.inference_kwargs = inference_kwargs
        self.datasets: dict[str, Any] = {}
        # bar index -> one probability per direction class, filled once per month
        self.month_predictions: dict[str, dict[int, tuple[float, ...]]] = {}

    def fit_month(self, plan: RunPlan, month: Month, bars: list[int]) -> None:
        self.datasets, self.month_predictions = {}, {}
        super().fit_month(plan, month, bars)

    def normalise(self, window: np.ndarray) -> np.ndarray | None:
        # Missing values become a raw 0 *before* scaling here, unlike the GRU;
        # see the module docstring of `prosper.predict.sequence`.
        if window.shape[0] == 0:
            return None
        safe = np.where(np.isfinite(window), window, 0.0)
        median = np.median(safe, axis=0)
        iqr = np.percentile(safe, 75, axis=0) - np.percentile(safe, 25, axis=0)
        iqr[iqr == 0] = 1.0
        return _robust_normalize(self.features, median, iqr)

    def fit_horizon(self, plan: RunPlan, horizon: str, first: int, month: Month) -> Any:
        assert self.normalised is not None
        normalised = self.normalised
        TimeSeriesDataSet = self.lib["TimeSeriesDataSet"]
        forward_steps = plan.steps[horizon]
        y_dir = self.labels.direction[horizon]
        train_start, train_end = plan.windows.bounds(first, horizon)
        valid_idx = [k for k in range(train_start, train_end) if y_dir[k] >= 0]
        if len(valid_idx) < self.seq_len + 10 or len(set(y_dir[valid_idx])) < 2:
            return None

        # Hold out the newest trainable bars to calibrate on. They sit inside
        # `training_bounds`, so their labels are realised and invariant 6 is
        # untouched; the model simply never fits them.
        split, _ = calibration_split(0, len(valid_idx))
        window_labels = [int(y_dir[k]) for k in valid_idx]
        if len(set(window_labels[:split])) < 2:
            # Same rescue as the GRU: a single-class fit half is unlearnable,
            # but a nearby split usually is not.
            alternative = usable_split(window_labels, split)
            split = len(valid_idx) if alternative is None else alternative
        fit_idx = valid_idx[:split]
        cal_idx = valid_idx[split:]
        if len(fit_idx) < self.seq_len + 10 or len(set(y_dir[fit_idx])) < 2:
            fit_idx, cal_idx = valid_idx, []

        def frame(bars: list[int]) -> pd.DataFrame:
            # Every bar from the first to the last of `bars`: TimeSeriesDataSet
            # needs consecutive time steps, and a bar whose forward return is
            # exactly zero has no label. It keeps its place and its features,
            # with an unknown target, as in `_predict_month`; `unlabelled`
            # then drops it as a training target.
            built = []
            for k in range(bars[0], bars[-1] + 1):
                target = IDX_TO_DIR[int(y_dir[k])] if y_dir[k] >= 0 else np.nan
                row = {"time_idx": k, "group": self.symbol, "target": target}
                for fi, fc in enumerate(self.feature_cols):
                    row[fc] = float(normalised[k, fi])
                built.append(row)
            return pd.DataFrame(built)

        def labelled_only(dataset: Any, df: pd.DataFrame) -> Any:
            unlabelled = df.loc[df["target"].isna(), "time_idx"]
            if unlabelled.empty:
                return dataset
            return dataset.filter(lambda index: ~index.time_idx_first_prediction.isin(unlabelled))

        df_pd = frame(fit_idx)
        max_enc = min(self.seq_len, len(df_pd) - self.seq_len)
        if max_enc < 5:
            return None

        # The encoder cannot be fitted on a column holding NaN, so where the
        # frame has an unlabelled bar it is fitted on the labels alone. Same
        # classes either way: `add_nan` reserves index 0 for the unknown one.
        target_encoder = self.lib["NaNLabelEncoder"](add_nan=True)
        if df_pd["target"].isna().any():
            target_encoder.fit(df_pd["target"].dropna())

        try:
            ds = TimeSeriesDataSet(
                df_pd,
                time_idx="time_idx",
                target="target",
                group_ids=["group"],
                min_encoder_length=max_enc // 2,
                max_encoder_length=max_enc,
                min_prediction_length=1,
                max_prediction_length=1,
                time_varying_unknown_reals=self.feature_cols,
                target_normalizer=target_encoder,
                add_relative_time_idx=True,
                add_target_scales=False,
                add_encoder_length=True,
            )
            ds = labelled_only(ds, df_pd)
            loader = ds.to_dataloader(train=True, batch_size=32, num_workers=0)

            # A validation set built from the held-out slice. Without one,
            # `reduce_on_plateau_patience` could never fire - Lightning warns
            # that val_loss is unavailable and skips the schedule - and there
            # was nothing to stop training on. The encoder needs history before
            # the first scored bar, so the validation frame reaches back seq_len
            # bars, all of them still inside the training window.
            val_loader = None
            val_ds = None
            if len(cal_idx) >= MIN_EARLY_STOPPING_SAMPLES:
                lo = max(0, min(cal_idx) - self.seq_len)
                val_bars = [k for k in range(lo, max(cal_idx) + 1) if y_dir[k] >= 0]
                if len(val_bars) >= MIN_EARLY_STOPPING_SAMPLES + self.seq_len:
                    val_frame = frame(val_bars)
                    val_ds = labelled_only(
                        TimeSeriesDataSet.from_dataset(
                            ds, val_frame, predict=False, stop_randomization=True
                        ),
                        val_frame,
                    )
                    val_loader = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

            self.reseed(horizon, month)
            tft = self.lib["TemporalFusionTransformer"].from_dataset(
                ds,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                attention_head_size=self.attention_head_size,
                dropout=0.1,
                hidden_continuous_size=16,
                loss=self.lib["CrossEntropy"](),
                log_interval=-1,
                reduce_on_plateau_patience=2,
            )

            # The model is used immediately and discarded, so neither a logger
            # nor a checkpoint is wanted. Left on, Lightning writes one
            # lightning_logs/version_N/ and one .ckpt per Trainer - a full run
            # would leave ~18k directories and 260 MB of dead artifacts.
            trainer_kwargs: dict[str, Any] = dict(
                max_epochs=self.epoch_budget[horizon],
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
                accelerator="cpu",
                default_root_dir=str(self.settings.meta_dir / "lightning"),
            )
            if self.settings.deterministic:
                trainer_kwargs["deterministic"] = True
            if val_loader is not None:
                trainer_kwargs["callbacks"] = [
                    _calibrated_val_loss_callback(val_ds),
                    self.lib["EarlyStopping"](
                        monitor=CALIBRATED_VAL_METRIC,
                        patience=EARLY_STOPPING_PATIENCE,
                        mode="min",
                        # The metric is absent when the callback could not score
                        # the slice; run the full budget then rather than abort.
                        strict=False,
                    ),
                ]
            trainer = self.lib["Trainer"](**trainer_kwargs)
            trainer.fit(tft, train_dataloaders=loader, val_dataloaders=val_loader)
            # After `fit`, current_epoch is the count that ran: Lightning
            # increments it past the last completed epoch, and an early stop
            # leaves it at the epoch the patience expired on.
            self.epochs_used[horizon].append(int(trainer.current_epoch))
            self.datasets[horizon] = ds
            self.calibrators[horizon] = _fit_tft_calibrator(
                tft,
                ds,
                TimeSeriesDataSet,
                X_norm=normalised,
                y_dir=y_dir,
                feature_cols=self.feature_cols,
                symbol=self.symbol,
                bar_indices=cal_idx,
                seq_len=self.seq_len,
                forward_steps=forward_steps,
                trainer_kwargs=self.inference_kwargs,
            )
            self.depths[horizon] = empirical_depth(
                y_dir[train_start:train_end],
                self.labels.depth[horizon][train_start:train_end],
                self.depth_labels,
            )
            return tft
        except Exception as e:
            self.failures.record("training", horizon, str(month), e)
            return None

    def after_fit_month(self, plan: RunPlan, month: Month, bars: list[int]) -> None:
        """One batched pass per horizon for the whole month.

        Rebuilding a dataset and dataloader for every single bar would cost more
        than the model itself - about 1.5 h per run.
        """
        month_bars = [b for b in bars if b >= self.seq_len]
        for horizon in plan.horizons:
            model = self.models.get(horizon.name)
            dataset = self.datasets.get(horizon.name)
            if model is None or dataset is None or self.normalised is None or not month_bars:
                continue
            try:
                self.month_predictions[horizon.name] = _predict_month(
                    model,
                    dataset,
                    self.lib["TimeSeriesDataSet"],
                    X_norm=self.normalised,
                    y_dir=self.labels.direction[horizon.name],
                    feature_cols=self.feature_cols,
                    symbol=self.symbol,
                    bar_indices=month_bars,
                    seq_len=self.seq_len,
                    forward_steps=plan.steps[horizon.name],
                    cutoff=bars[0],
                    trainer_kwargs=self.inference_kwargs,
                )
            except Exception as e:
                # This turns the whole month into untrained rows, so a bare
                # message is not enough to tell a data problem from a library one.
                self.failures.record(
                    "inference", horizon.name, str(month), e, with_traceback=True
                )
                self.month_predictions[horizon.name] = {}

    def probabilities(self, bar: int, horizon: str) -> tuple[float, ...] | None:
        return self.month_predictions.get(horizon, {}).get(bar)


def predict_tft(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    seq_len: int = 60,
    train_window_days: int | None = DEFAULT_TRAIN_WINDOW_DAYS,
    train_window_by_horizon: str | None = None,
    horizons_selected: str | None = None,
    max_epochs: int = DEFAULT_EPOCH_BUDGET["tft"],
    epochs_by_horizon: str | None = None,
    hidden_size: int = 32,
    attention_head_size: int = 2,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    """Train a Temporal Fusion Transformer every calendar month (rolling window)
    and produce per-bar JSONL predictions identical in format to ML/GRU models.
    """
    if settings is None:
        settings = get_settings()

    epoch_budget = resolve_epochs(max_epochs, parse_epoch_overrides(epochs_by_horizon))
    seed_run(settings, torch_seed=True)

    # Suppress noisy upstream warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Applied to the Trainer that `predict` creates for itself; without it that
    # one logs to `lightning_logs/` in whatever directory the CLI was run from.
    inference_kwargs: dict[str, Any] = {
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "accelerator": "cpu",
        "default_root_dir": str(settings.meta_dir / "lightning"),
    }

    loaded = load_sequence(symbol, interval, settings)
    if isinstance(loaded, dict):
        return loaded
    frame, feature_cols = loaded

    depth_bins, depth_labels = parse_depth_bins(DEFAULT_DEPTH_BINS_STR)
    plan = plan_run(
        symbol,
        interval,
        start,
        end,
        open_times=frame["open_time"].to_list(),
        dates=frame["_date"].to_list(),
        depth_labels=depth_labels,
        train_window_days=train_window_days,
        train_window_by_horizon=train_window_by_horizon,
        horizons_selected=horizons_selected,
    )

    try:
        import lightning as L
        from lightning.pytorch.callbacks import EarlyStopping
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        from pytorch_forecasting.metrics import CrossEntropy
    except ImportError:
        return {
            "error": "pytorch-forecasting not installed. Run: poetry add pytorch-forecasting lightning",
            "symbol": symbol,
        }

    model = TFTForecaster(
        feature_cols=feature_cols,
        forecasting={
            "Trainer": L.Trainer,
            "EarlyStopping": EarlyStopping,
            "TemporalFusionTransformer": TemporalFusionTransformer,
            "TimeSeriesDataSet": TimeSeriesDataSet,
            "NaNLabelEncoder": NaNLabelEncoder,
            "CrossEntropy": CrossEntropy,
        },
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        learning_rate=learning_rate,
        inference_kwargs=inference_kwargs,
        settings=settings,
        symbol=symbol,
        features=frame.select(feature_cols).to_numpy().astype(np.float32),
        labels=forward_labels(frame["close"].to_list(), plan.steps, depth_bins),
        depth_labels=depth_labels,
        seq_len=seq_len,
        epoch_budget=epoch_budget,
    )
    return run(model, plan, settings)
