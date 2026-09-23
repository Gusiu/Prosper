"""GRU: a recurrent network over the last *seq_len* bars, one per horizon."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from prosper.config import Settings, get_settings
from prosper.domain import DEFAULT_DEPTH_BINS_STR, DIRECTION_CLASSES, parse_depth_bins
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
    device: torch.device,
) -> float:
    """Held-out loss after temperature scaling — see `calibrated_nll`.

    Raw cross-entropy is not usable as a stopping signal here: it charges for
    overconfidence, the network becomes overconfident long before it stops
    learning to discriminate, and the calibrator undoes exactly that
    overconfidence one step later. Stopping on the raw value gave the epoch
    budget back for nothing and cost GRU 1.7 points of composite score.
    """
    model.eval()
    with torch.no_grad():
        xs = torch.stack([dataset[k][0] for k in indices]).to(device)
        ys = torch.stack([dataset[k][1] for k in indices]).to(device)
        probs = torch.softmax(model(xs), dim=1).cpu().numpy()
    model.train()
    return calibrated_nll(probs, ys.cpu().numpy())


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
    about five.

    What is measured is the loss *after* temperature scaling. The raw
    cross-entropy on this slice rises from the first epoch — the network
    becomes overconfident faster than it stops discriminating — so stopping on
    it returned a one-epoch model while accuracy was still climbing.
    `_holdout_loss` therefore reports the calibrated value, which is also what
    the stored artifact will carry.

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

        loss = _holdout_loss(model, dataset, holdout, device)
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
    """Median/IQR normalisation fitted on train, applied to all.

    Returns ``None`` when there is nothing to fit on. That happens at the very
    first bar of a symbol whose history begins inside the requested range — the
    causal window reaches back from bar 0 and finds nothing — and it used to
    crash inside `np.percentile`, which indexes `arr[-1]` and cannot do so on an
    empty array. A bar with no history behind it has no training rows either, so
    the honest outcome is an untrained bar (invariant 3), not an exception that
    loses the whole run.
    """
    if X_train.size == 0 or X_train.shape[0] == 0:
        return None
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


# ── the model in the walk-forward loop ────────────────────────────────────────
class GRUForecaster(SequenceForecaster):
    """A GRU classifier per horizon, scored bar by bar on the last *seq_len* bars."""

    name = "gru"

    def __init__(
        self,
        *,
        device: torch.device,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        batch_size: int,
        lr: float,
        **common: Any,
    ) -> None:
        super().__init__(**common)
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr

    def normalise(self, window: np.ndarray) -> np.ndarray | None:
        return _robust_normalise(window, self.features)

    def fit_horizon(self, plan: RunPlan, horizon: str, first: int, month: Month) -> Any:
        assert self.normalised is not None
        train_start, train_end = plan.windows.bounds(first, horizon)
        if train_end - train_start < self.seq_len + 10:
            return None

        directions = self.labels.direction[horizon]
        y_slice = directions[train_start:train_end]
        X_slice = self.normalised[train_start:train_end]
        valid = [k for k in range(len(y_slice)) if y_slice[k] >= 0]
        if len(valid) < self.seq_len + 5 or len(set(y_slice[valid])) <= 1:
            return None

        ds = SequenceDataset(X_slice, y_slice, self.seq_len)
        if len(ds) < 8:
            return None

        # Hold out the newest sequences of the training region for the
        # calibrator and the stopping rule. They still end before `train_end`,
        # so their labels are realised - invariant 6 is untouched.
        split, _ = calibration_split(0, len(ds))
        fit_ds: Any = ds
        cal_range: list[int] = []
        if split < len(ds):
            seq_labels = [int(ds[k][1]) for k in range(len(ds))]
            if len(set(seq_labels[:split])) <= 1:
                # Nothing to learn from a single-class fit half. Look for a
                # nearby split that works before giving up: doing neither cost 5
                # of the 13 affected windows their holdout, and with it early
                # stopping and calibration.
                alternative = usable_split(seq_labels, split)
                split = len(ds) if alternative is None else alternative
            if split < len(ds):
                fit_ds = Subset(ds, list(range(split)))
                cal_range = list(range(split, len(ds)))

        self.reseed(horizon, month)
        loader = DataLoader(fit_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        model = GRUClassifier(
            self.features.shape[1], self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.epochs_used[horizon].append(
            _train_gru(
                model,
                loader,
                optimiser,
                nn.CrossEntropyLoss(),
                dataset=ds,
                holdout=cal_range,
                max_epochs=self.epoch_budget[horizon],
                device=self.device,
            )
        )
        self.calibrators[horizon] = _fit_gru_calibrator(model, ds, cal_range, self.device)
        # The GRU head only models direction; depth comes from the realised
        # conditional distribution over the same window.
        self.depths[horizon] = empirical_depth(
            directions[train_start:train_end],
            self.labels.depth[horizon][train_start:train_end],
            self.depth_labels,
        )
        return model

    def probabilities(self, bar: int, horizon: str) -> np.ndarray | None:
        if bar < self.seq_len or self.normalised is None:
            return None
        model = self.models[horizon]
        sequence = (
            torch.tensor(self.normalised[bar - self.seq_len : bar], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        model.eval()
        with torch.no_grad():
            logits = model(sequence)  # (1, len(DIRECTION_CLASSES))
            return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    def summary(self) -> dict[str, Any]:
        return {**super().summary(), "device": str(self.device)}


# ── main entry-point ──────────────────────────────────────────────────────────
def predict_gru(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    seq_len: int = 30,
    train_window_days: int | None = DEFAULT_TRAIN_WINDOW_DAYS,
    train_window_by_horizon: str | None = None,
    horizons_selected: str | None = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = DEFAULT_EPOCH_BUDGET["gru"],
    epochs_by_horizon: str | None = None,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Train a GRU classifier on rolling windows and produce per-bar JSONL predictions.

    Output format is identical to predict_baseline / predict_ml.
    """
    if settings is None:
        settings = get_settings()

    epoch_budget = resolve_epochs(epochs, parse_epoch_overrides(epochs_by_horizon))
    seed_run(settings, torch_seed=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = GRUForecaster(
        device=device,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        settings=settings,
        symbol=symbol,
        features=frame.select(feature_cols).to_numpy().astype(np.float32),
        labels=forward_labels(frame["close"].to_list(), plan.steps, depth_bins),
        depth_labels=depth_labels,
        seq_len=seq_len,
        epoch_budget=epoch_budget,
    )
    return run(model, plan, settings)
