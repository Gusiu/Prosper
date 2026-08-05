from __future__ import annotations

import json
from typing import Any

import numpy as np
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
    fit_calibrator_from_model,
)
from prosper.predict.defaults import DEFAULT_TRAIN_WINDOW_DAYS, parse_horizons
from prosper.predict.window import (
    days_to_steps,
    training_bounds,
    untrained_horizon_payload,
    validate_train_window,
)
from prosper.storage.layout import get_features_parquet_path
from prosper.storage.predictions import write_versioned_predictions
from prosper.utils.time import parse_date


def predict_xgboost(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
    horizons_selected: str | None = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    """
    XGBoost probabilistic direction predictions (short/medium/long).

    Trains an `XGBClassifier` per horizon on a rolling window and writes a
    versioned run folder containing `predictions.jsonl` and
    `feature_importances.json`.
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

    horizons = list(DEFAULT_HORIZONS)
    selected = set(parse_horizons(horizons_selected))
    steps_by_horizon = validate_train_window(train_window_days, interval, horizons)
    train_window_steps = days_to_steps(train_window_days, interval)

    effective_seed = settings.seed if settings.seed is not None else 42
    if settings.deterministic or settings.seed is not None:
        import random

        random.seed(effective_seed)
        np.random.seed(effective_seed)

    depth_bins, depth_labels = parse_depth_bins(DEFAULT_DEPTH_BINS_STR)

    start_dt = parse_date(start)
    end_dt = parse_date(end)

    feat_path = get_features_parquet_path(symbol, interval, settings=settings)
    if not feat_path.exists():
        return {
            "error": f"No features parquet found for {symbol}. Run features build.",
            "symbol": symbol,
        }

    df_feat = pl.read_parquet(feat_path, hive_partitioning=False).sort("open_time")
    if df_feat.is_empty():
        return {"error": "Empty features DataFrame", "symbol": symbol}

    df_feat = df_feat.with_columns(pl.col("open_time").dt.date().alias("_date"))

    exclude_cols = ["open_time", "close_time", "_date", "close", "symbol"]
    feature_cols = [c for c in df_feat.columns if c not in exclude_cols]

    open_times = df_feat["open_time"].to_list()
    dates = df_feat["_date"].to_list()
    closes = df_feat["close"].to_list()
    n = len(df_feat)

    X_all = df_feat.select(feature_cols).to_numpy()

    # Pre-calc direction and depth targets, indexed in bars.
    horizon_targets = {}
    for h in horizons:
        forward_steps = steps_by_horizon[h.name]
        directions: list[str | None] = [None] * n
        depths: list[int | None] = [None] * n
        for i in range(n):
            j = i + forward_steps
            if j < n:
                r = closes[j] / closes[i] - 1.0
                d = direction_from_return(r)
                directions[i] = d
                if d is not None:
                    depths[i] = assign_depth_bin(abs(r) * 100.0, depth_bins)
        horizon_targets[h.name] = {"direction": directions, "depth": depths}

    predictions: list[dict[str, Any]] = []
    untrained_counts: dict[str, int] = {h.name: 0 for h in horizons}

    dir_map = DIR_TO_IDX
    inv_dir_map = IDX_TO_DIR

    current_train_month = None
    models_cache: dict[str, Any] = {}

    # Walk-forward: train and infer within the same loop
    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt.date() or row_date > end_dt.date():
            continue

        train_month = row_date.replace(day=1)

        # Retrain models once per calendar month
        if current_train_month != train_month:
            current_train_month = train_month
            models_cache = {}

            for h in horizons:
                if h.name not in selected:
                    models_cache[h.name] = None
                    continue
                forward_steps = steps_by_horizon[h.name]
                train_start, train_end = training_bounds(i, train_window_steps, forward_steps)
                y_dirs = horizon_targets[h.name]["direction"][train_start:train_end]
                y_depths = horizon_targets[h.name]["depth"][train_start:train_end]
                X_train = X_all[train_start:train_end]

                valid_idx = [k for k, d in enumerate(y_dirs) if d is not None]

                if len(valid_idx) < 10:
                    models_cache[h.name] = None
                    continue

                X_train_valid = X_train[valid_idx]
                y_train_valid = [dir_map[y_dirs[k]] for k in valid_idx]

                # XGBClassifier requires labels 0..k-1. A window where only
                # short and long occur yields {0, 2}, which it rejects — and
                # the resulting failure used to disable the horizon entirely.
                # Remap to a dense range and keep the original class order to
                # map the probabilities back.
                # The newest slice of the training region is held out to fit
                # the calibrator; both halves end before `train_end`, so no
                # unrealised label reaches either (invariant 6).
                split, _ = calibration_split(0, len(valid_idx))
                fit_rows = list(range(split))
                cal_rows = list(range(split, len(valid_idx)))
                y_fit = [y_train_valid[k] for k in fit_rows] if fit_rows else y_train_valid
                if len(set(y_fit)) <= 1:
                    y_fit, fit_rows, cal_rows = y_train_valid, list(range(len(valid_idx))), []

                present_classes = sorted(set(y_fit))
                clf = None
                calibrator = TemperatureCalibrator(1.0)
                if len(present_classes) > 1:
                    dense = {label: idx for idx, label in enumerate(present_classes)}
                    try:
                        clf = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=effective_seed,
                            eval_metric="mlogloss",
                        )
                        clf.fit(X_train_valid[fit_rows], [dense[y] for y in y_fit])
                    except Exception as e:
                        print(f"[xgboost] {h.name} training failed at {row_date}: {e}")
                        clf = None
                    if clf is not None and cal_rows:
                        calibrator = fit_calibrator_from_model(
                            clf,
                            X_train_valid[cal_rows],
                            [y_train_valid[k] for k in cal_rows],
                            present_classes=present_classes,
                        )

                # Empirical depth distributions conditioned on realised direction.
                emp_long = [0] * len(depth_labels)
                emp_short = [0] * len(depth_labels)
                for k in valid_idx:
                    if y_depths[k] is None:
                        continue
                    if y_dirs[k] == "long":
                        emp_long[y_depths[k]] += 1
                    elif y_dirs[k] == "short":
                        emp_short[y_depths[k]] += 1

                models_cache[h.name] = {
                    "clf": clf,
                    # Column j of predict_proba corresponds to present_classes[j].
                    "classes": present_classes if clf is not None else [],
                    "calibrator": calibrator,
                    "depth_long": _to_prob_dict(emp_long, depth_labels),
                    "depth_short": _to_prob_dict(emp_short, depth_labels),
                }

        # ── Inference for current bar ─────────────────────────────────────
        out: dict[str, Any] = {
            "open_time": open_times[i].isoformat(),
            "date": row_date.isoformat(),
            "symbol": symbol,
        }

        X_test = X_all[i : i + 1]
        for h in horizons:
            cache = models_cache.get(h.name)
            if not cache or cache.get("clf") is None:
                out[h.name] = untrained_horizon_payload(depth_labels)
                untrained_counts[h.name] += 1
                continue

            probs = cache["clf"].predict_proba(X_test)[0]
            p_dict = dict.fromkeys(DIRECTION_CLASSES, 0.0)
            for cls_idx, prob in zip(cache["classes"], probs, strict=True):
                p_dict[inv_dir_map[cls_idx]] = prob
            p_dict = cache["calibrator"].apply_to_mapping(p_dict)

            out[h.name] = {
                **{f"P_{d}": p_dict[d] for d in DIRECTION_CLASSES},
                "depth_long_bins": cache["depth_long"],
                "depth_short_bins": cache["depth_short"],
                "trained": True,
            }

        predictions.append(out)

    if not predictions:
        return {"error": "No predictions generated for the requested date range", "symbol": symbol}

    # Persist predictions grouped by month

    out_dir = write_versioned_predictions(
        predictions, symbol, "xgboost", settings, interval=interval
    )
    fi: dict[str, dict[str, float]] = {}
    for h in horizons:
        cache = models_cache.get(h.name)
        if cache and cache.get("clf") is not None:
            importances = getattr(cache["clf"], "feature_importances_", None)
            if importances is not None:
                fi[h.name] = {
                    col: float(imp)
                    for col, imp in zip(feature_cols, importances, strict=True)
                }
    if fi:
        (out_dir / "feature_importances.json").write_text(
            json.dumps(fi, indent=2), encoding="utf-8"
        )

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "predictions": len(predictions),
        "untrained_horizons": untrained_counts,
        "run_dir": str(out_dir),
    }


def _to_prob_dict(counts: list[int], labels: list[str]) -> dict[str, float]:
    total = sum(counts)
    if total == 0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: count / total for label, count in zip(labels, counts, strict=True)}
