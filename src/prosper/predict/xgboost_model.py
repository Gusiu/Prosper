from __future__ import annotations

import json
from typing import Any

import numpy as np

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    DIRECTION_CLASSES,
    IDX_TO_DIR,
    effective_symbols,
    parse_depth_bins,
)
from prosper.predict.calibration import (
    TemperatureCalibrator,
    calibration_split,
    fit_calibrator_from_model,
)
from prosper.predict.defaults import (
    DEFAULT_TRAIN_WINDOW_DAYS,
    parse_horizons,
    parse_window_overrides,
)
from prosper.predict.pool import describe_pool, load_pool, parse_symbols
from prosper.predict.window import (
    count_scored_bars,
    resolve_train_windows,
    untrained_horizon_payload,
    warn_low_power,
)
from prosper.storage.predictions import write_run_summary, write_versioned_predictions
from prosper.utils.time import parse_date


def predict_xgboost(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
    train_window_by_horizon: str | None = None,
    horizons_selected: str | None = None,
    symbols: str | None = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    """
    XGBoost probabilistic direction predictions, one model per horizon.

    Trains an `XGBClassifier` per horizon on a rolling window and writes a
    versioned run folder containing `predictions.jsonl` and
    `feature_importances.json`.

    *symbols* pools extra symbols into the training set; predictions are always
    for *symbol* alone. The pooled path is taken even for one symbol, so there
    is a single implementation of "which rows may this model see" rather than
    two that agree until they quietly stop — the mistake this project already
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

    horizons = list(DEFAULT_HORIZONS)
    selected = set(parse_horizons(horizons_selected))
    window_overrides = parse_window_overrides(train_window_by_horizon)

    effective_seed = settings.seed if settings.seed is not None else 42
    if settings.deterministic or settings.seed is not None:
        import random

        random.seed(effective_seed)
        np.random.seed(effective_seed)

    depth_bins, depth_labels = parse_depth_bins(DEFAULT_DEPTH_BINS_STR)

    start_dt = parse_date(start)
    end_dt = parse_date(end)

    pooled_symbols = parse_symbols(symbols, symbol)
    forward_steps_by_horizon = {h.name: h.steps(interval) for h in horizons}
    try:
        pool, feature_cols = load_pool(
            pooled_symbols, interval, forward_steps_by_horizon, depth_bins, settings
        )
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc), "symbol": symbol}

    target = pool.frames[0]
    open_times = target.timestamps
    dates = [t.date() for t in open_times]
    n = len(target)

    # After the load, not before: the window a horizon should get depends on how
    # much history there is to divide between training it and scoring it.
    windows = resolve_train_windows(
        train_window_days, interval, horizons, window_overrides, history_bars=n
    )
    scored_bars = count_scored_bars(dates, start_dt.date(), end_dt.date(), windows)
    # Pooling scales the training side, but by  rather than by
    # the symbol count: correlated symbols do not each bring an observation.
    pool_power = {
        name: effective_symbols(len(pool.frames), pool.return_correlation(forward))
        for name, forward in forward_steps_by_horizon.items()
    }
    warn_low_power(windows, scored_bars, pool_power)

    X_all = target.features
    pool_contributions: dict[str, Any] = {}

    predictions: list[dict[str, Any]] = []
    untrained_counts: dict[str, int] = {h.name: 0 for h in horizons}

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

            month_slices: dict[str, Any] = {}
            for h in horizons:
                if h.name not in selected:
                    models_cache[h.name] = None
                    continue
                # Bounded by instants, not row indices: row 400 of BTCUSDT is
                # September 2018 and row 400 of SOLUSDT September 2021, so an
                # index-based window would train a 2021 forecast on bars that
                # had not happened yet.
                pooled = pool.training_slice(open_times[i], h.name, windows)
                month_slices[h.name] = pooled
                y_dirs = pooled.direction
                y_depths = pooled.depth
                X_train = pooled.features

                valid_idx = [k for k, d in enumerate(y_dirs) if d >= 0]

                if len(valid_idx) < 10:
                    models_cache[h.name] = None
                    continue

                X_train_valid = X_train[valid_idx]
                y_train_valid = [int(y_dirs[k]) for k in valid_idx]

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
                    if y_depths[k] < 0:
                        continue
                    side = inv_dir_map[int(y_dirs[k])]
                    if side == "long":
                        emp_long[int(y_depths[k])] += 1
                    elif side == "short":
                        emp_short[int(y_depths[k])] += 1

                models_cache[h.name] = {
                    "clf": clf,
                    # Column j of predict_proba corresponds to present_classes[j].
                    "classes": present_classes if clf is not None else [],
                    "calibrator": calibrator,
                    "depth_long": _to_prob_dict(emp_long, depth_labels),
                    "depth_short": _to_prob_dict(emp_short, depth_labels),
                }

            if month_slices:
                pool_contributions = describe_pool(
                    month_slices, pool, forward_steps_by_horizon
                )

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

    result = {
        "symbol": symbol,
        "trained_on": list(pool.symbols),
        "pool_contributions": pool_contributions,
        "start": start,
        "end": end,
        "interval": interval,
        "predictions": len(predictions),
        "untrained_horizons": untrained_counts,
        "training_windows": windows.describe(scored_bars, pool_power),
        "run_dir": str(out_dir),
    }
    write_run_summary(out_dir, result)
    return result


def _to_prob_dict(counts: list[int], labels: list[str]) -> dict[str, float]:
    total = sum(counts)
    if total == 0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: count / total for label, count in zip(labels, counts, strict=True)}
