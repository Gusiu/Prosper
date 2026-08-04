"""Temporal Fusion Transformer predictions using pytorch-forecasting."""

from __future__ import annotations

import traceback
import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    DIR_TO_IDX,
    DIRECTION_CLASSES,
    IDX_TO_DIR,
    assign_depth_bin,
    direction_from_return,
    parse_depth_bins,
)
from prosper.predict.calibration import (
    TemperatureCalibrator,
    calibration_split,
    fit_temperature,
)
from prosper.predict.window import (
    days_to_steps,
    training_bounds,
    untrained_horizon_payload,
    validate_train_window,
)
from prosper.storage.layout import get_features_parquet_path
from prosper.storage.predictions import write_versioned_predictions
from prosper.utils.time import parse_date


def _robust_normalize(X: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    X_safe = np.where(np.isfinite(X), X, 0.0)
    X_norm = (X_safe - med) / iqr
    return np.clip(np.nan_to_num(X_norm, nan=0.0, posinf=3.0, neginf=-3.0), -5.0, 5.0)


def _logits_to_probs(logits: np.ndarray) -> tuple[float, ...]:
    """Softmax a raw output row into one probability per direction class.

    Returned in `DIRECTION_CLASSES` order, so the caller never has to know how
    many classes there are.
    """
    row = np.asarray(logits).reshape(-1)
    # NaNLabelEncoder(add_nan=True) prepends an "unknown" class; drop it so the
    # direction classes line up.
    if row.shape[0] == len(DIRECTION_CLASSES) + 1:
        row = row[1:]
    shifted = row - np.max(row)
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    return tuple(float(v) for v in probs)


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

    Building a TimeSeriesDataSet and a DataLoader per bar — which is what this
    replaces — dominated the runtime: a full run took hours, almost none of it
    spent in the model.

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
    # the CWD — the flags on the *training* Trainer do not reach it. Batching
    # cut the call count from one per bar to one per month, which hid the leak
    # rather than closing it.
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
    # (output, index) tuple — which cannot be unpacked into two names. Test for
    # `output`, not `index`: every tuple has an `.index` *method*, so that check
    # is true for both shapes and would send the old one down the wrong branch.
    if hasattr(result, "output"):
        raw, index = result.output, result.index
    else:
        raw, index = result[0], result[1]
    predictions = _unwrap_prediction(raw)

    wanted = set(bar_indices)
    out: dict[int, tuple[float, ...]] = {}
    for position, decoder_start in enumerate(index["time_idx"].tolist()):
        bar = int(decoder_start)
        if bar in wanted:
            out[bar] = _logits_to_probs(predictions[position, 0].cpu().numpy())
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
    from the very region being scored — the same masking rule inference uses.
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


def predict_tft(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    seq_len: int = 60,
    train_window_days: int = 730,
    max_epochs: int = 10,
    hidden_size: int = 32,
    attention_head_size: int = 2,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    """
    Train a Temporal Fusion Transformer every calendar month (rolling window)
    and produce per-bar JSONL predictions identical in format to ML/GRU models.
    """
    if settings is None:
        settings = get_settings()

    horizon_specs = list(DEFAULT_HORIZONS)
    steps_by_horizon = validate_train_window(train_window_days, interval, horizon_specs)
    train_window_steps = days_to_steps(train_window_days, interval)

    # ── Seed & deterministic mode (research) ─────────────────────────────────
    if settings.deterministic:
        seed = settings.seed if settings.seed is not None else 42
        np.random.seed(seed)
        import random

        random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except ImportError:
            pass
        print(f"[research] Deterministic mode ON (seed={seed})")
    elif settings.seed is not None:
        np.random.seed(settings.seed)
        import random

        random.seed(settings.seed)
        print(f"[research] Seed set to {settings.seed} (non-deterministic CUDA)")

    # Suppress noisy upstream warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Applied to the Trainer that `predict` creates for itself; without it that
    # one logs to `lightning_logs/` in whatever directory the CLI was run from.
    inference_trainer_kwargs: dict[str, Any] = {
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "accelerator": "cpu",
        "default_root_dir": str(settings.meta_dir / "lightning"),
    }

    # ── 1. Load features ──────────────────────────────────────────────────────
    feat_path = get_features_parquet_path(symbol, interval, settings=settings)
    if not feat_path.exists():
        return {"error": f"No features parquet for {symbol}. Run: features build", "symbol": symbol}

    df_pl = pl.read_parquet(feat_path, hive_partitioning=False).sort("open_time")
    if df_pl.is_empty():
        return {"error": "Empty features DataFrame", "symbol": symbol}

    df_pl = df_pl.with_columns(pl.col("open_time").dt.date().alias("_date"))
    exclude_cols = {"open_time", "close_time", "_date", "symbol"}
    feature_cols = [c for c in df_pl.columns if c not in exclude_cols and c != "close"]

    open_times = df_pl["open_time"].to_list()
    dates = df_pl["_date"].to_list()
    closes = df_pl["close"].to_list()
    X_raw = df_pl.select(feature_cols).to_numpy().astype(np.float32)
    n = len(df_pl)

    start_dt = parse_date(start).date()
    end_dt = parse_date(end).date()

    # ── 2. Pre-compute forward labels ─────────────────────────────────────────
    dir_map = DIR_TO_IDX
    depth_bins, depth_labels = parse_depth_bins(DEFAULT_DEPTH_BINS_STR)
    y_dir_all: dict[str, np.ndarray] = {}
    y_depth_all: dict[str, np.ndarray] = {}
    for h in horizon_specs:
        forward_steps = steps_by_horizon[h.name]
        dir_arr = np.full(n, -1, dtype=np.int64)
        depth_arr = np.full(n, -1, dtype=np.int64)
        for i in range(n - forward_steps):
            j = i + forward_steps
            if closes[i] and closes[j]:
                r = closes[j] / closes[i] - 1.0
                direction = direction_from_return(r)
                if direction is not None:
                    dir_arr[i] = dir_map[direction]
                    bin_idx = assign_depth_bin(abs(r) * 100.0, depth_bins)
                    if bin_idx is not None:
                        depth_arr[i] = bin_idx
        y_dir_all[h.name] = dir_arr
        y_depth_all[h.name] = depth_arr

    # ── 3. Rolling-month train + inference ───────────────────────────────────
    try:
        import lightning as L
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        from pytorch_forecasting.metrics import CrossEntropy
    except ImportError:
        return {
            "error": "pytorch-forecasting not installed. Run: poetry add pytorch-forecasting lightning",
            "symbol": symbol,
        }

    predictions: list[dict[str, Any]] = []
    untrained_counts: dict[str, int] = {h.name: 0 for h in horizon_specs}
    current_train_month = None
    tft_model: dict[str, Any] = {}
    dataset_cache: dict[str, Any] = {}
    calibrator_cache: dict[str, TemperatureCalibrator] = {}
    depth_cache: dict[str, dict[str, dict[str, float]]] = {}
    # bar index -> one probability per direction class, filled once per month
    month_predictions: dict[str, dict[int, tuple[float, float, float]]] = {}
    X_norm_all: np.ndarray | None = None

    # Bars to predict, grouped by the month whose model will serve them.
    bars_by_month: dict[tuple[int, int], list[int]] = {}
    for idx in range(n):
        if start_dt <= dates[idx] <= end_dt:
            bars_by_month.setdefault((dates[idx].year, dates[idx].month), []).append(idx)

    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt or row_date > end_dt:
            continue

        month_key = (row_date.year, row_date.month)

        # ── Train once per calendar month ─────────────────────────────────
        if month_key != current_train_month:
            current_train_month = month_key
            tft_model = {}
            dataset_cache = {}
            depth_cache = {}

            # Normalise on past rows only; stats never see future bars.
            X_tr = X_raw[max(0, i - train_window_steps) : i]
            X_safe = np.where(np.isfinite(X_tr), X_tr, 0.0)
            med = np.median(X_safe, axis=0)
            iqr_arr = np.percentile(X_safe, 75, axis=0) - np.percentile(X_safe, 25, axis=0)
            iqr_arr[iqr_arr == 0] = 1.0
            X_norm_all = _robust_normalize(X_raw, med, iqr_arr)

            for h in horizon_specs:
                h_name = h.name
                forward_steps = steps_by_horizon[h_name]
                y_dir = y_dir_all[h_name]
                train_start, train_end = training_bounds(i, train_window_steps, forward_steps)
                # Build pandas dataframe for TimeSeriesDataSet
                valid_idx = [k for k in range(train_start, train_end) if y_dir[k] >= 0]
                if len(valid_idx) < seq_len + 10 or len(set(y_dir[valid_idx])) < 2:
                    tft_model[h_name] = None
                    continue

                # Hold out the newest trainable bars to calibrate on. They sit
                # inside `training_bounds`, so their labels are realised and
                # invariant 6 is untouched; the model simply never fits them.
                split, _ = calibration_split(0, len(valid_idx))
                fit_idx = valid_idx[:split]
                cal_idx = valid_idx[split:]
                if len(fit_idx) < seq_len + 10 or len(set(y_dir[fit_idx])) < 2:
                    fit_idx, cal_idx = valid_idx, []

                rows = []
                for k in fit_idx:
                    row = {"time_idx": k, "group": symbol, "target": IDX_TO_DIR[int(y_dir[k])]}
                    for fi, fc in enumerate(feature_cols):
                        row[fc] = float(X_norm_all[k, fi])
                    rows.append(row)

                df_pd = pd.DataFrame(rows)

                max_enc = min(seq_len, len(df_pd) - seq_len)
                if max_enc < 5:
                    tft_model[h_name] = None
                    continue

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
                        time_varying_unknown_reals=feature_cols,
                        target_normalizer=NaNLabelEncoder(add_nan=True),
                        add_relative_time_idx=True,
                        add_target_scales=False,
                        add_encoder_length=True,
                    )

                    loader = ds.to_dataloader(train=True, batch_size=32, num_workers=0)

                    tft = TemporalFusionTransformer.from_dataset(
                        ds,
                        learning_rate=learning_rate,
                        hidden_size=hidden_size,
                        attention_head_size=attention_head_size,
                        dropout=0.1,
                        hidden_continuous_size=16,
                        loss=CrossEntropy(),
                        log_interval=-1,
                        reduce_on_plateau_patience=2,
                    )

                    # The model is used immediately and discarded, so neither a
                    # logger nor a checkpoint is wanted. Left on, Lightning
                    # writes one lightning_logs/version_N/ and one .ckpt per
                    # Trainer — a full run once left ~18k directories and
                    # 260 MB of dead artifacts in the repo root.
                    trainer_kwargs: dict[str, Any] = dict(
                        max_epochs=max_epochs,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                        enable_checkpointing=False,
                        logger=False,
                        accelerator="cpu",
                        default_root_dir=str(settings.meta_dir / "lightning"),
                    )
                    if settings.deterministic:
                        trainer_kwargs["deterministic"] = True
                    trainer = L.Trainer(**trainer_kwargs)
                    trainer.fit(tft, train_dataloaders=loader)
                    tft_model[h_name] = tft
                    dataset_cache[h_name] = ds
                    calibrator_cache[h_name] = _fit_tft_calibrator(
                        tft,
                        ds,
                        TimeSeriesDataSet,
                        X_norm=X_norm_all,
                        y_dir=y_dir,
                        feature_cols=feature_cols,
                        symbol=symbol,
                        bar_indices=cal_idx,
                        seq_len=seq_len,
                        forward_steps=forward_steps,
                        trainer_kwargs=inference_trainer_kwargs,
                    )
                    depth_cache[h_name] = _empirical_depth(
                        y_dir_all[h_name][train_start:train_end],
                        y_depth_all[h_name][train_start:train_end],
                        depth_labels,
                    )
                except Exception as e:
                    print(f"Training exception for month {current_train_month}: {e}")
                    tft_model[h_name] = None

            # One batched pass per horizon for the whole month, instead of
            # rebuilding a dataset and dataloader for every single bar.
            month_predictions = {}
            month_bars = [b for b in bars_by_month.get(month_key, []) if b >= seq_len]
            for h in horizon_specs:
                model = tft_model.get(h.name)
                ds_ref = dataset_cache.get(h.name)
                if model is None or ds_ref is None or X_norm_all is None or not month_bars:
                    continue
                try:
                    month_predictions[h.name] = _predict_month(
                        model,
                        ds_ref,
                        TimeSeriesDataSet,
                        X_norm=X_norm_all,
                        y_dir=y_dir_all[h.name],
                        feature_cols=feature_cols,
                        symbol=symbol,
                        bar_indices=month_bars,
                        seq_len=seq_len,
                        forward_steps=steps_by_horizon[h.name],
                        cutoff=i,
                        trainer_kwargs=inference_trainer_kwargs,
                    )
                except Exception as e:
                    # This path turns the whole month into untrained rows, so a
                    # bare message is not enough to tell a data problem from a
                    # library one.
                    print(
                        f"Batched inference failed for {h.name} {month_key}: "
                        f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )
                    month_predictions[h.name] = {}

        # ── Inference: read the month's batched result ────────────────────
        out: dict[str, Any] = {
            "open_time": open_times[i].isoformat(),
            "date": row_date.isoformat(),
            "symbol": symbol,
        }

        for h in horizon_specs:
            probs = month_predictions.get(h.name, {}).get(i)
            if probs is None:
                out[h.name] = untrained_horizon_payload(depth_labels)
                untrained_counts[h.name] += 1
                continue

            probs = calibrator_cache.get(h.name, TemperatureCalibrator(1.0)).apply(probs)
            depths = depth_cache.get(h.name, {})
            out[h.name] = {
                **{f"P_{IDX_TO_DIR[k]}": p for k, p in enumerate(probs)},
                "depth_long_bins": depths.get("long", _uniform_depth(depth_labels)),
                "depth_short_bins": depths.get("short", _uniform_depth(depth_labels)),
                "trained": True,
            }

        predictions.append(out)

    if not predictions:
        return {"error": "No predictions generated for the requested date range", "symbol": symbol}


    run_dir = write_versioned_predictions(
        predictions, symbol, "tft", settings, interval=interval
    )

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "predictions": len(predictions),
        "untrained_horizons": untrained_counts,
        "run_dir": str(run_dir),
    }


def _uniform_depth(labels: list[str]) -> dict[str, float]:
    return {label: 1.0 / len(labels) for label in labels}


def _empirical_depth(
    directions: np.ndarray,
    depths: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Realised depth-bin distribution per direction over a training window."""
    counts = {"long": [0] * len(labels), "short": [0] * len(labels)}
    for direction_idx, depth_idx in zip(directions, depths, strict=True):
        if depth_idx < 0 or direction_idx < 0:
            continue
        direction = IDX_TO_DIR.get(int(direction_idx))
        if direction in counts:
            counts[direction][int(depth_idx)] += 1

    out: dict[str, dict[str, float]] = {}
    for direction, bucket in counts.items():
        total = sum(bucket)
        out[direction] = (
            _uniform_depth(labels)
            if total == 0
            else {label: c / total for label, c in zip(labels, bucket, strict=True)}
        )
    return out
