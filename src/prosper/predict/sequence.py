"""What the two sequence models share: a causal normalisation and a monthly refit.

GRU and TFT read windows of consecutive bars from one symbol. At the first bar
of each month both normalise the features on the bars before it, fit one model
per horizon on the labelled rows of that horizon's window, and keep a
calibrator and a realised depth distribution beside it. They differ in the
network and in how a prediction is produced - GRU scores bar by bar, TFT
predicts a whole month in one batched pass.

They normalise the same way, with `robust_scale`. The first 89 bars of every
symbol have missing features (indicator warm-up) and early training windows
reach back that far, so how a missing value is scaled is part of what both
networks are trained on, and comparing them is fair only under one rule.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prosper.config import Settings
from prosper.domain import DIRECTION_CLASSES, IDX_TO_DIR, training_features
from prosper.labels.forward import ForwardLabels
from prosper.predict.calibration import TemperatureCalibrator
from prosper.predict.defaults import summarise_epochs, window_seed
from prosper.predict.failures import WindowFailures
from prosper.predict.walk_forward import Month, RunPlan, uniform_depth
from prosper.storage.layout import get_features_parquet_path


def robust_scale(window: np.ndarray, features: np.ndarray) -> np.ndarray | None:
    """*features* scaled by the median and interquartile range of *window*.

    The statistics are taken over the values present: a missing value counted
    as a zero would drag the median and widen the range of every column still
    in its warm-up. A missing value itself becomes 0 after scaling, the window's
    median - the value that tells the network nothing about the bar. Infinities
    go to +-3 and everything is clipped to +-5, so one extreme bar cannot
    dominate a sequence. A column with no value in the window, or no spread,
    is scaled by 1 around 0.

    Returns ``None`` when *window* is empty. That happens at the very first bar
    of a symbol whose history begins inside the requested range - the causal
    window reaches back from bar 0 and finds nothing - and a bar with no
    history behind it has no training rows either, so the honest outcome is an
    untrained bar (invariant 3), not an exception that loses the whole run.
    """
    if window.size == 0:
        return None
    present = np.where(np.isfinite(window), window, np.nan)
    with warnings.catch_warnings():
        # A column with no value present yields NaN statistics, handled below.
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(present, axis=0)
        q75, q25 = np.nanpercentile(present, [75, 25], axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    iqr = q75 - q25
    iqr = np.where(np.isfinite(iqr) & (iqr != 0), iqr, 1.0)
    scaled = np.nan_to_num((features - median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0)
    return np.clip(scaled, -5.0, 5.0)


class SequenceForecaster:
    """Monthly refit of one sequence model per horizon on normalised features."""

    name = "sequence"

    def __init__(
        self,
        *,
        settings: Settings,
        symbol: str,
        features: np.ndarray,
        labels: ForwardLabels,
        depth_labels: list[str],
        seq_len: int,
        epoch_budget: dict[str, int],
    ) -> None:
        self.settings = settings
        self.symbol = symbol
        self.features = features
        self.labels = labels
        self.depth_labels = depth_labels
        self.seq_len = seq_len
        self.epoch_budget = epoch_budget
        self.normalised: np.ndarray | None = None
        self.models: dict[str, Any] = {}
        # Kept across months: a horizon that fails to fit has no model and so
        # never reaches its calibrator, whatever the cache holds.
        self.calibrators: dict[str, TemperatureCalibrator] = {}
        self.depths: dict[str, dict[str, dict[str, float]]] = {}
        # How many epochs each horizon actually trained for, per window. The
        # budget is a ceiling, so this is a result, and without it whether
        # early stopping fired at all was invisible.
        self.epochs_used: dict[str, list[int]] = {name: [] for name in epoch_budget}
        self.failures = WindowFailures()

    # ── the walk-forward hooks ──────────────────────────────────────────────

    def fit_month(self, plan: RunPlan, month: Month, bars: list[int]) -> None:
        first = bars[0]
        self.models, self.depths = {}, {}
        # The widest window in use, because normalisation is fitted once per
        # month and shared by every horizon. Still causal: it reads only bars
        # strictly before the first one being predicted.
        span = plan.windows.widest_window_steps
        self.normalised = self.normalise(self.features[max(0, first - span) : first])
        if self.normalised is None:
            # No history behind this bar: nothing to normalise on and nothing to
            # train on, so every horizon of the month is untrained. Skipping the
            # bar instead would leave it missing from the output.
            return
        for horizon in plan.horizons:
            if horizon.name in plan.selected:
                self.models[horizon.name] = self.fit_horizon(plan, horizon.name, first, month)
        self.after_fit_month(plan, month, bars)

    def predict(self, bar: int, horizon: str) -> dict[str, Any] | None:
        if self.models.get(horizon) is None or self.normalised is None:
            return None
        probabilities = self.probabilities(bar, horizon)
        if probabilities is None:
            return None
        calibrated = self.calibrators.get(horizon, TemperatureCalibrator(1.0)).apply(probabilities)
        depths = self.depths.get(horizon, {})
        return {
            **{f"P_{IDX_TO_DIR[k]}": float(calibrated[k]) for k in range(len(DIRECTION_CLASSES))},
            "depth_long_bins": depths.get("long", uniform_depth(self.depth_labels)),
            "depth_short_bins": depths.get("short", uniform_depth(self.depth_labels)),
            "trained": True,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "epochs_used": summarise_epochs(self.epochs_used, self.epoch_budget),
            "failed_windows": self.failures.summary(),
        }

    def after_write(self, run_dir: Path) -> None:
        self.failures.warn_if_any()

    # ── what a subclass supplies ────────────────────────────────────────────

    def normalise(self, window: np.ndarray) -> np.ndarray | None:
        """All bars scaled by statistics of *window*, or None if it is empty."""
        return robust_scale(window, self.features)

    @abstractmethod
    def fit_horizon(self, plan: RunPlan, horizon: str, first: int, month: Month) -> Any:
        """Fit one horizon for the month starting at bar *first*; the model or None."""

    @abstractmethod
    def probabilities(self, bar: int, horizon: str) -> Sequence[float] | np.ndarray | None:
        """Uncalibrated direction probabilities for *bar*, in `DIRECTION_CLASSES` order."""

    def after_fit_month(self, plan: RunPlan, month: Month, bars: list[int]) -> None:
        """Anything done once per month after every horizon is fitted."""

    # ── shared by both subclasses ───────────────────────────────────────────

    def reseed(self, horizon: str, month: Month) -> None:
        """Seed torch from this window's own identity.

        Seeding once before the walk-forward loop left every model downstream of
        every other: raising one horizon's epoch ceiling retrained the rest. See
        `window_seed`.
        """
        if self.settings.deterministic or self.settings.seed is not None:
            import torch

            base = self.settings.seed if self.settings.seed is not None else 42
            torch.manual_seed(window_seed(base, self.name, horizon, *month, symbol=self.symbol))


def load_sequence(
    symbol: str, interval: str, settings: Settings
) -> tuple[pl.DataFrame, list[str]] | dict[str, Any]:
    """One symbol's features in time order, and the columns a model trains on.

    An include list, not an exclude list: taking every column in the parquet
    bar a handful of names would feed the model raw price levels - and adopt
    each new relative column alongside its absolute twin.
    """
    path = get_features_parquet_path(symbol, interval, settings=settings)
    if not path.exists():
        return {"error": f"No features parquet for {symbol}. Run: features build", "symbol": symbol}
    frame = pl.read_parquet(path, hive_partitioning=False).sort("open_time")
    if frame.is_empty():
        return {"error": "Empty features DataFrame", "symbol": symbol}
    feature_cols = training_features(frame.columns)
    if not feature_cols:
        return {"error": "No feature columns found", "symbol": symbol}
    return frame.with_columns(pl.col("open_time").dt.date().alias("_date")), feature_cols
