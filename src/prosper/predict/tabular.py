"""What the two tabular models share: pooled rows in, one classifier per horizon.

The ML and XGBoost predictors were 59% the same lines. Both draw each month's
training rows from a pool of symbols by instant (`pool.training_slice`), drop
rows whose label is not yet known, hold out the newest slice for calibration,
fit a direction classifier, and read depth from the realised distribution.
They differ in the classifier and in how they handle a window where one class
is missing - which is exactly what a subclass supplies here.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from prosper.config import Settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    DIRECTION_CLASSES,
    IDX_TO_DIR,
    effective_symbols,
    parse_depth_bins,
)
from prosper.predict.calibration import TemperatureCalibrator
from prosper.predict.pool import (
    PooledSlice,
    TrainingPool,
    describe_pool,
    load_pool,
    parse_symbols,
)
from prosper.predict.walk_forward import Month, RunPlan, plan_run

# Fewer labelled rows than this and a window is not fitted at all.
MIN_LABELLED_ROWS = 10


class TabularForecaster:
    """Monthly refit of one classifier per horizon on pooled, labelled rows."""

    name = "tabular"

    def __init__(self, pool: TrainingPool, depth_labels: list[str]) -> None:
        self.pool = pool
        self.depth_labels = depth_labels
        self.features = pool.frames[0].features
        self.fitted: dict[str, dict[str, Any] | None] = {}
        self.pool_contributions: dict[str, Any] = {}

    # ── the walk-forward hooks ──────────────────────────────────────────────

    def fit_month(self, plan: RunPlan, month: Month, bars: list[int]) -> None:
        first = bars[0]
        self.fitted = {}
        slices: dict[str, PooledSlice] = {}
        for horizon in plan.horizons:
            if horizon.name not in plan.selected:
                continue
            # Bounded by instants, not row indices: row 400 of BTCUSDT is
            # September 2018 and row 400 of SOLUSDT September 2021, so an
            # index-based window would train a 2021 forecast on bars that had
            # not happened yet.
            pooled = self.pool.training_slice(plan.open_times[first], horizon.name, plan.windows)
            slices[horizon.name] = pooled
            valid = [k for k, side in enumerate(pooled.direction) if side >= 0]
            if len(valid) < MIN_LABELLED_ROWS:
                self.fitted[horizon.name] = None
                continue
            self.fitted[horizon.name] = self.fit_horizon(
                horizon.name, str(plan.dates[first]), pooled, valid
            )
        if slices:
            self.pool_contributions = describe_pool(slices, self.pool, plan.steps)

    def predict(self, bar: int, horizon: str) -> dict[str, Any] | None:
        fitted = self.fitted.get(horizon)
        if not fitted or fitted["clf"] is None:
            return None
        return self.predict_horizon(fitted, self.features[bar : bar + 1])

    def summary(self) -> dict[str, Any]:
        return {
            "trained_on": list(self.pool.symbols),
            "pool_contributions": self.pool_contributions,
        }

    def after_write(self, run_dir: Path) -> None:
        return None

    # ── what a subclass supplies ────────────────────────────────────────────

    @abstractmethod
    def fit_horizon(
        self, horizon: str, window: str, pooled: PooledSlice, valid: list[int]
    ) -> dict[str, Any]:
        """Fit one horizon's models on the labelled rows *valid* of *pooled*."""

    @abstractmethod
    def predict_horizon(self, fitted: dict[str, Any], row: np.ndarray) -> dict[str, Any]:
        """One horizon's payload for a single feature row."""

    # ── shared by both subclasses ───────────────────────────────────────────

    @staticmethod
    def direction_probabilities(
        clf: Any, classes: Any, calibrator: TemperatureCalibrator, row: np.ndarray
    ) -> dict[str, float]:
        """Classifier output mapped back to direction names, then calibrated.

        Column j of `predict_proba` belongs to ``classes[j]``, which is not
        always both directions: a window can lack one side entirely.
        """
        probabilities = dict.fromkeys(DIRECTION_CLASSES, 0.0)
        for class_index, probability in zip(classes, clf.predict_proba(row)[0], strict=True):
            probabilities[IDX_TO_DIR[class_index]] = probability
        return calibrator.apply_to_mapping(probabilities)


def load_tabular(
    symbol: str,
    start: str,
    end: str,
    settings: Settings,
    *,
    interval: str,
    symbols: str | None,
    train_window_days: int | None,
    train_window_by_horizon: str | None,
    horizons_selected: str | None,
    depth_bins_str: str = DEFAULT_DEPTH_BINS_STR,
) -> tuple[TrainingPool, list[str], RunPlan] | dict[str, Any]:
    """Load the pool and plan the run, or say why not. Also returns the feature names.

    One path whether pooling or not, so "which rows may this model see" has a
    single implementation.
    """
    depth_bins, depth_labels = parse_depth_bins(depth_bins_str)
    steps = {h.name: h.steps(interval) for h in DEFAULT_HORIZONS}
    try:
        pool, feature_cols = load_pool(
            parse_symbols(symbols, symbol), interval, steps, depth_bins, settings
        )
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc), "symbol": symbol}

    target = pool.frames[0]
    # Pooling scales the training side, but by the effective number of symbols
    # rather than the count: correlated symbols do not each bring an observation.
    power = {
        name: effective_symbols(len(pool.frames), pool.return_correlation(forward))
        for name, forward in steps.items()
    }
    plan = plan_run(
        symbol,
        interval,
        start,
        end,
        open_times=target.timestamps,
        dates=[value.date() for value in target.timestamps],
        depth_labels=depth_labels,
        train_window_days=train_window_days,
        train_window_by_horizon=train_window_by_horizon,
        horizons_selected=horizons_selected,
        effective_symbols=power,
    )
    return pool, feature_cols, plan
