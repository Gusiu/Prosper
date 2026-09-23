"""XGBoost: one gradient-boosted direction classifier per horizon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prosper.config import Settings, get_settings
from prosper.domain import DIRECTION_CLASSES
from prosper.predict.calibration import (
    TemperatureCalibrator,
    calibration_split,
    fit_calibrator_from_model,
)
from prosper.predict.defaults import DEFAULT_TRAIN_WINDOW_DAYS
from prosper.predict.failures import WindowFailures
from prosper.predict.pool import PooledSlice, TrainingPool
from prosper.predict.tabular import TabularForecaster, load_tabular
from prosper.predict.walk_forward import empirical_depth, run, seed_run


class XGBoostForecaster(TabularForecaster):
    """A direction classifier per horizon; depth is the realised distribution."""

    name = "xgboost"

    def __init__(
        self,
        pool: TrainingPool,
        depth_labels: list[str],
        feature_cols: list[str],
        *,
        xgb: Any,
        seed: int,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
    ) -> None:
        super().__init__(pool, depth_labels)
        self.feature_cols = feature_cols
        self.xgb = xgb
        self.seed = seed
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }
        self.failures = WindowFailures()

    def fit_horizon(
        self, horizon: str, window: str, pooled: PooledSlice, valid: list[int]
    ) -> dict[str, Any]:
        y_dirs, X_train = pooled.direction, pooled.features
        X_valid = X_train[valid]
        y_valid = [int(y_dirs[k]) for k in valid]

        # The newest slice of the training region is held out to fit the
        # calibrator; both halves end before `train_end`, so no unrealised
        # label reaches either (invariant 6).
        split, _ = calibration_split(0, len(valid))
        fit_rows = list(range(split))
        cal_rows = list(range(split, len(valid)))
        y_fit = [y_valid[k] for k in fit_rows] if fit_rows else y_valid
        if len(set(y_fit)) <= 1:
            y_fit, fit_rows, cal_rows = y_valid, list(range(len(valid))), []

        # XGBClassifier requires labels 0..k-1. A window where only one side
        # occurs, or where the classes are not contiguous, is remapped to a
        # dense range; `present_classes` maps the probabilities back. Without
        # it, such a window used to fail and disable the horizon entirely.
        present_classes = sorted(set(y_fit))
        clf = None
        calibrator = TemperatureCalibrator(1.0)
        if len(present_classes) > 1:
            dense = {label: idx for idx, label in enumerate(present_classes)}
            try:
                clf = self.xgb.XGBClassifier(
                    **self.params, random_state=self.seed, eval_metric="mlogloss"
                )
                clf.fit(X_valid[fit_rows], [dense[y] for y in y_fit])
            except Exception as e:
                self.failures.record("training", horizon, window, e)
                clf = None
            if clf is not None and cal_rows:
                calibrator = fit_calibrator_from_model(
                    clf,
                    X_valid[cal_rows],
                    [y_valid[k] for k in cal_rows],
                    present_classes=present_classes,
                )

        return {
            "clf": clf,
            "classes": present_classes if clf is not None else [],
            "calibrator": calibrator,
            "depth": empirical_depth(y_dirs, pooled.depth, self.depth_labels),
        }

    def predict_horizon(self, fitted: dict[str, Any], row: np.ndarray) -> dict[str, Any]:
        probabilities = self.direction_probabilities(
            fitted["clf"], fitted["classes"], fitted["calibrator"], row
        )
        return {
            **{f"P_{d}": probabilities[d] for d in DIRECTION_CLASSES},
            "depth_long_bins": fitted["depth"]["long"],
            "depth_short_bins": fitted["depth"]["short"],
            "trained": True,
        }

    def summary(self) -> dict[str, Any]:
        return {**super().summary(), "failed_windows": self.failures.summary()}

    def after_write(self, run_dir: Path) -> None:
        """The last month's models' feature importances, per horizon."""
        importances: dict[str, dict[str, float]] = {}
        for horizon, fitted in self.fitted.items():
            if fitted and fitted["clf"] is not None:
                values = getattr(fitted["clf"], "feature_importances_", None)
                if values is not None:
                    importances[horizon] = {
                        col: float(value)
                        for col, value in zip(self.feature_cols, values, strict=True)
                    }
        if importances:
            (run_dir / "feature_importances.json").write_text(
                json.dumps(importances, indent=2), encoding="utf-8"
            )
        self.failures.warn_if_any()


def predict_xgboost(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    train_window_days: int | None = DEFAULT_TRAIN_WINDOW_DAYS,
    train_window_by_horizon: str | None = None,
    horizons_selected: str | None = None,
    symbols: str | None = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    """XGBoost probabilistic direction predictions, one model per horizon.

    *symbols* pools extra symbols into the training set; predictions are always
    for *symbol* alone. The pooled path is taken even for one symbol, so there
    is a single implementation of "which rows may this model see" rather than
    two that agree until they quietly stop - the mistake this project already
    made once with three copies of the grading rule.
    """
    try:
        import xgboost as xgb
    except Exception as e:
        return {
            "error": "xgboost not installed: install xgboost to use this command",
            "detail": str(e),
        }

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
    )
    if isinstance(loaded, dict):
        return loaded
    pool, feature_cols, plan = loaded
    model = XGBoostForecaster(
        pool, plan.depth_labels, feature_cols,
        xgb=xgb, seed=seed,
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
    )
    return run(model, plan, settings)
