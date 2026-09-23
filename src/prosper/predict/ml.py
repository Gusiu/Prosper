"""Gradient-boosted trees from scikit-learn: one direction and two depth models per horizon."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from prosper.config import Settings, get_settings
from prosper.domain import DEFAULT_DEPTH_BINS_STR, DIRECTION_CLASSES, IDX_TO_DIR
from prosper.predict.calibration import calibration_split, fit_calibrator_from_model
from prosper.predict.defaults import DEFAULT_TRAIN_WINDOW_DAYS
from prosper.predict.pool import PooledSlice, TrainingPool
from prosper.predict.tabular import TabularForecaster, load_tabular
from prosper.predict.walk_forward import empirical_depth, run, seed_run


class MLForecaster(TabularForecaster):
    """HistGradientBoosting for direction, and for depth where a side has enough rows.

    Depth is modelled separately for long and short moves. A side with five
    rows or fewer, or a single depth bin, falls back to its realised
    distribution over the window.
    """

    name = "ml"

    def __init__(self, pool: TrainingPool, depth_labels: list[str], seed: int) -> None:
        super().__init__(pool, depth_labels)
        self.seed = seed

    def fit_horizon(
        self, horizon: str, window: str, pooled: PooledSlice, valid: list[int]
    ) -> dict[str, Any]:
        y_dirs, y_depths, X_train = pooled.direction, pooled.depth, pooled.features
        X_valid = X_train[valid]
        y_valid = [int(y_dirs[k]) for k in valid]

        # Reserve the most recent slice of the *training* region to calibrate
        # on. Holding out later bars instead would break invariant 6: their
        # labels need closes the forecast cannot see.
        split, _ = calibration_split(0, len(valid))
        fit_idx = list(range(split))
        cal_idx = list(range(split, len(valid)))

        model = HistGradientBoostingClassifier(
            random_state=self.seed, max_iter=50, max_leaf_nodes=15
        )
        clf: HistGradientBoostingClassifier | None = model
        y_fit = [y_valid[k] for k in fit_idx] if fit_idx else y_valid
        X_fit = X_valid[fit_idx] if fit_idx else X_valid
        if len(set(y_fit)) > 1:
            model.fit(X_fit, y_fit)
        elif len(set(y_valid)) > 1:
            # The held-out split left a single-class fit set; train on
            # everything and skip calibration rather than lose the bar.
            model.fit(X_valid, y_valid)
            cal_idx = []
        else:
            clf = None

        calibrator = fit_calibrator_from_model(
            clf,
            X_valid[cal_idx] if cal_idx else np.empty((0, X_valid.shape[1])),
            [y_valid[k] for k in cal_idx],
            present_classes=list(clf.classes_) if clf is not None else None,
        )

        idx_long = [k for k in valid if IDX_TO_DIR[int(y_dirs[k])] == "long"]
        idx_short = [k for k in valid if IDX_TO_DIR[int(y_dirs[k])] == "short"]
        return {
            "clf": clf,
            "classes": clf.classes_ if clf is not None else [],
            "calibrator": calibrator,
            "depth_long": self._depth_model(X_train, y_depths, idx_long),
            "depth_short": self._depth_model(X_train, y_depths, idx_short),
            "fallback_depth": empirical_depth(y_dirs, y_depths, self.depth_labels),
        }

    def _depth_model(
        self, X_train: np.ndarray, y_depths: np.ndarray, rows: list[int]
    ) -> HistGradientBoostingClassifier | None:
        if len(rows) <= 5 or len({y_depths[k] for k in rows}) <= 1:
            return None
        model = HistGradientBoostingClassifier(
            random_state=self.seed, max_iter=30, max_leaf_nodes=10
        )
        model.fit(X_train[rows], [int(y_depths[k]) for k in rows])
        return model

    def predict_horizon(self, fitted: dict[str, Any], row: np.ndarray) -> dict[str, Any]:
        probabilities = self.direction_probabilities(
            fitted["clf"], fitted["classes"], fitted["calibrator"], row
        )
        return {
            **{f"P_{d}": probabilities[d] for d in DIRECTION_CLASSES},
            "depth_long_bins": self._depth_bins(fitted, "long", row),
            "depth_short_bins": self._depth_bins(fitted, "short", row),
            "trained": True,
        }

    def _depth_bins(self, fitted: dict[str, Any], side: str, row: np.ndarray) -> dict[str, float]:
        model = fitted[f"depth_{side}"]
        if model is None:
            return fitted["fallback_depth"][side]
        bins = {label: 0.0 for label in self.depth_labels}
        for class_index, probability in zip(model.classes_, model.predict_proba(row)[0], strict=True):
            bins[self.depth_labels[class_index]] = probability
        return bins


def predict_ml(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    train_window_days: int | None = DEFAULT_TRAIN_WINDOW_DAYS,
    train_window_by_horizon: str | None = None,
    horizons_selected: str | None = None,
    symbols: str | None = None,
    depth_bins_str: str = DEFAULT_DEPTH_BINS_STR,
) -> dict[str, Any]:
    """Walk-forward ML predictions for every horizon.

    ``train_window_days`` and the horizons are calendar spans; both are
    converted to bar counts for *interval*, so a 1h run forecasts the same
    real-world durations as a 1d run and the two stay comparable. *symbols*
    pools extra symbols into the training set; predictions are for *symbol*.
    """
    if settings is None:
        settings = get_settings()
    seed = seed_run(settings)

    loaded = load_tabular(
        symbol, start, end, settings,
        interval=interval,
        symbols=symbols,
        train_window_days=train_window_days,
        train_window_by_horizon=train_window_by_horizon,
        horizons_selected=horizons_selected,
        depth_bins_str=depth_bins_str,
    )
    if isinstance(loaded, dict):
        return loaded
    pool, _feature_cols, plan = loaded
    return run(MLForecaster(pool, plan.depth_labels, seed), plan, settings)
