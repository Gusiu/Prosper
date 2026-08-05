"""GRU-based Deep Learning model for probabilistic directional predictions."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

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
    usable_split,
)
from prosper.predict.defaults import (
    DEFAULT_EPOCH_BUDGET,
    DEFAULT_TRAIN_WINDOW_DAYS,
    EARLY_STOPPING_PATIENCE,
    MIN_EARLY_STOPPING_SAMPLES,
    parse_epoch_overrides,
    parse_horizons,
    resolve_epochs,
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


# ── dataset ──────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """Sliding-window sequences of feature rows with direction target."""

    def __init__(
        self,
        X: np.ndarray,  # (T, F)  – feature matrix, already normalised
        y_dir: np.ndarray,  # (T,)    – int labels 0/1/2
        seq_len: int,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_dir, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]  # (seq_len, F)
        y_lbl = self.y[idx + self.seq_len]  # scalar
        return x_seq, y_lbl


# ── model ────────────────────────────────────────────────────────────────────
class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = len(DIRECTION_CLASSES),
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # (B, T, H)
        last = out[:, -1, :]  # (B, H) – last timestep
        last = self.norm(last)
        return self.head(last)  # (B, C)


def _holdout_loss(
    model: nn.Module,
    dataset: Dataset,
    indices: list[int],
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Cross-entropy on the held-out sequences, without touching the gradients."""
    model.eval()
    with torch.no_grad():
        xs = torch.stack([dataset[k][0] for k in indices]).to(device)
        ys = torch.stack([dataset[k][1] for k in indices]).to(device)
        loss = float(loss_fn(model(xs), ys).item())
    model.train()
    return loss


def _train_gru(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    dataset: Dataset,
    holdout: list[int],
    max_epochs: int,
    device: torch.device,
) -> int:
    """Train until the held-out loss stops improving, or the budget runs out.

    `max_epochs` is a ceiling, not a target. A retraining window holds only a
    few hundred sequences against ~45k parameters, and how long that takes to
    saturate differs by horizon: measured on BTCUSDT, the short horizon still
    gained accuracy at 20 epochs while medium and long stopped gaining after
    about five and their NLL degraded monotonically past it — they were not
    getting more often right, only more confidently wrong.

    The best weights are restored on the way out. Stopping without that would
    hand back the *worst* model of the patience window, which is the opposite
    of the intent.

    The held-out slice is the same one the calibrator uses. Splitting it again
    would drop each half under the sample floor at the long horizon, so the
    stopping decision and the temperature fit share it; the contamination is
    one scalar's worth and is preferable to calibrating on 40 rows.
    """
    if len(holdout) < MIN_EARLY_STOPPING_SAMPLES:
        model.train()
        for _ in range(max_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                loss_fn(model(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
        return max_epochs

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    since_improved = 0
    epochs_run = 0

    model.train()
    for epoch in range(max_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss_fn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        epochs_run = epoch + 1

        loss = _holdout_loss(model, dataset, holdout, loss_fn, device)
        if loss < best_loss - 1e-6:
            best_loss = loss
            best_state = copy.deepcopy(model.state_dict())
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return epochs_run


def _fit_gru_calibrator(
    model: nn.Module,
    dataset: Dataset,
    indices: list[int],
    device: torch.device,
) -> TemperatureCalibrator:
    """Score the held-out sequences and fit a temperature on the result."""
    if not indices:
        return TemperatureCalibrator(1.0)

    model.eval()
    xs = torch.stack([dataset[k][0] for k in indices]).to(device)
    ys = np.array([int(dataset[k][1]) for k in indices])
    with torch.no_grad():
        probs = torch.softmax(model(xs), dim=-1).cpu().numpy()
    model.train()
    return fit_temperature(probs, ys)


# ── normalisation ─────────────────────────────────────────────────────────────
def _robust_normalise(X_train: np.ndarray, X_all: np.ndarray):
    """Median/IQR normalisation fitted on train, applied to all."""
    # Replace NaN before statistics so numpy doesn't warn on all-NaN columns
    X_safe = np.where(np.isfinite(X_train), X_train, 0.0)
    med = np.median(X_safe, axis=0)
    q75 = np.percentile(X_safe, 75, axis=0)
    q25 = np.percentile(X_safe, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    X_norm = (X_all - med) / iqr
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=3.0, neginf=-3.0)
    X_norm = np.clip(X_norm, -5.0, 5.0)
    return X_norm


# ── main entry-point ──────────────────────────────────────────────────────────
def predict_gru(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    seq_len: int = 30,
    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
    horizons_selected: str | None = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = DEFAULT_EPOCH_BUDGET["gru"],
    epochs_by_horizon: str | None = None,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """
    Train a GRU classifier on rolling windows and produce per-bar JSONL predictions.
    Output format is identical to predict_baseline / predict_ml.
    """
    if settings is None:
        settings = get_settings()

    horizon_specs = list(DEFAULT_HORIZONS)
    selected = set(parse_horizons(horizons_selected))
    epoch_budget = resolve_epochs(epochs, parse_epoch_overrides(epochs_by_horizon))
    steps_by_horizon = validate_train_window(train_window_days, interval, horizon_specs)
    train_window_steps = days_to_steps(train_window_days, interval)

    # ── Seed & deterministic mode (research) ─────────────────────────────────
    if settings.deterministic:
        seed = settings.seed if settings.seed is not None else 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        import random

        random.seed(seed)
        torch.use_deterministic_algorithms(True)
        print(f"[research] Deterministic mode ON (seed={seed})")
    elif settings.seed is not None:
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        import random

        random.seed(settings.seed)
        print(f"[research] Seed set to {settings.seed} (non-deterministic CUDA)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load features ──────────────────────────────────────────────────────
    feat_path = get_features_parquet_path(symbol, interval, settings=settings)
    if not feat_path.exists():
        return {"error": f"No features parquet for {symbol}. Run: features build", "symbol": symbol}

    df = pl.read_parquet(feat_path, hive_partitioning=False).sort("open_time")
    if df.is_empty():
        return {"error": "Empty features DataFrame", "symbol": symbol}

    df = df.with_columns(pl.col("open_time").dt.date().alias("_date"))

    exclude_cols = {"open_time", "close_time", "_date", "symbol"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and c != "close"]
    if not feature_cols:
        return {"error": "No feature columns found", "symbol": symbol}

    open_times = df["open_time"].to_list()
    dates = df["_date"].to_list()
    closes = df["close"].to_list()
    X_all_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    n = len(df)

    start_dt = parse_date(start).date()
    end_dt = parse_date(end).date()

    # ── 2. Pre-compute forward labels ─────────────────────────────────────────
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
                    dir_arr[i] = DIR_TO_IDX[direction]
                    bin_idx = assign_depth_bin(abs(r) * 100.0, depth_bins)
                    if bin_idx is not None:
                        depth_arr[i] = bin_idx
        y_dir_all[h.name] = dir_arr
        y_depth_all[h.name] = depth_arr

    # ── 3. Rolling-month training + inference ─────────────────────────────────
    predictions: list[dict[str, Any]] = []
    untrained_counts: dict[str, int] = {h.name: 0 for h in horizon_specs}
    # We retrain once per calendar month to balance speed vs. freshness
    current_train_month = None
    models_cache: dict[str, Any] = {}
    calibrator_cache: dict[str, TemperatureCalibrator] = {}
    depth_cache: dict[str, dict[str, dict[str, float]]] = {}
    X_norm_current: np.ndarray | None = None

    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt or row_date > end_dt:
            continue

        # Retrain once per month
        month_key = (row_date.year, row_date.month)
        if month_key != current_train_month:
            current_train_month = month_key
            models_cache = {}
            depth_cache = {}

            # Normalise using only past rows, then apply to the full array so
            # inference indices stay valid. Stats never see future bars.
            X_norm_current = _robust_normalise(X_all_raw[max(0, i - train_window_steps) : i], X_all_raw)

            for h in horizon_specs:
                if h.name not in selected:
                    models_cache[h.name] = None
                    continue
                forward_steps = steps_by_horizon[h.name]
                train_start, train_end = training_bounds(i, train_window_steps, forward_steps)
                if train_end - train_start < seq_len + 10:
                    models_cache[h.name] = None
                    continue

                y_slice = y_dir_all[h.name][train_start:train_end]
                X_slice = X_norm_current[train_start:train_end]
                valid = [k for k in range(len(y_slice)) if y_slice[k] >= 0]
                if len(valid) < seq_len + 5 or len(set(y_slice[valid])) <= 1:
                    models_cache[h.name] = None
                    continue

                ds = SequenceDataset(X_slice, y_slice, seq_len)
                if len(ds) < 8:
                    models_cache[h.name] = None
                    continue

                # Hold out the newest sequences of the training region for the
                # calibrator and the stopping rule. They still end before
                # `train_end`, so their labels are realised — invariant 6 is
                # untouched.
                split, _ = calibration_split(0, len(ds))
                fit_ds: Any = ds
                cal_range: list[int] = []
                if split < len(ds):
                    seq_labels = [int(ds[k][1]) for k in range(len(ds))]
                    if len(set(seq_labels[:split])) <= 1:
                        # Nothing to learn from a single-class fit half. Look
                        # for a nearby split that works before giving up: doing
                        # neither cost 5 of the 13 affected windows their
                        # holdout, and with it early stopping and calibration.
                        alternative = usable_split(seq_labels, split)
                        split = len(ds) if alternative is None else alternative
                    if split < len(ds):
                        fit_ds = Subset(ds, list(range(split)))
                        cal_range = list(range(split, len(ds)))

                loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, drop_last=False)
                m = GRUClassifier(len(feature_cols), hidden_size, num_layers, dropout).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                loss_fn = nn.CrossEntropyLoss()

                _train_gru(
                    m,
                    loader,
                    opt,
                    loss_fn,
                    dataset=ds,
                    holdout=cal_range,
                    max_epochs=epoch_budget[h.name],
                    device=device,
                )

                models_cache[h.name] = m
                calibrator_cache[h.name] = _fit_gru_calibrator(m, ds, cal_range, device)
                # The GRU head only models direction; depth comes from the
                # realised conditional distribution over the same window.
                depth_cache[h.name] = _empirical_depth(
                    y_dir_all[h.name][train_start:train_end],
                    y_depth_all[h.name][train_start:train_end],
                    depth_labels,
                )

        # ── Inference ─────────────────────────────────────────────────────────
        out: dict[str, Any] = {
            "open_time": open_times[i].isoformat(),
            "date": row_date.isoformat(),
            "symbol": symbol,
        }

        for h in horizon_specs:
            m = models_cache.get(h.name)
            if m is None or X_norm_current is None or i < seq_len:
                out[h.name] = untrained_horizon_payload(depth_labels)
                untrained_counts[h.name] += 1
                continue

            x_seq = (
                torch.tensor(X_norm_current[i - seq_len : i], dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            m.eval()
            with torch.no_grad():
                logits = m(x_seq)  # (1, len(DIRECTION_CLASSES))
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            probs = calibrator_cache.get(h.name, TemperatureCalibrator(1.0)).apply(probs)

            depths = depth_cache.get(h.name, {})
            out[h.name] = {
                **{
                    f"P_{IDX_TO_DIR[k]}": float(probs[k])
                    for k in range(len(DIRECTION_CLASSES))
                },
                "depth_long_bins": depths.get("long", _uniform_depth(depth_labels)),
                "depth_short_bins": depths.get("short", _uniform_depth(depth_labels)),
                "trained": True,
            }

        predictions.append(out)

    if not predictions:
        return {"error": "No predictions generated for the requested date range", "symbol": symbol}


    run_dir = write_versioned_predictions(
        predictions, symbol, "gru", settings, interval=interval
    )

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "predictions": len(predictions),
        "untrained_horizons": untrained_counts,
        "run_dir": str(run_dir),
        "device": str(device),
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
        if depth_idx < 0:
            continue
        direction = IDX_TO_DIR.get(int(direction_idx))
        if direction in counts:
            counts[direction][int(depth_idx)] += 1

    out: dict[str, dict[str, float]] = {}
    for direction, bucket in counts.items():
        total = sum(bucket)
        if total == 0:
            out[direction] = _uniform_depth(labels)
        else:
            out[direction] = {
                label: count / total for label, count in zip(labels, bucket, strict=True)
            }
    return out
