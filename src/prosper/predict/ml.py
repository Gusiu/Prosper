from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    DIRECTION_CLASSES,
    IDX_TO_DIR,
    direction_from_return,
    effective_symbols,
    parse_depth_bins,
)
from prosper.predict.calibration import calibration_split, fit_calibrator_from_model
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


def _direction_from_return_safe(r: float) -> str | None:
    """Wrapper that also handles NaN values from sklearn."""
    if r is None or math.isnan(r):
        return None
    return direction_from_return(r)


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
    """
    ML predictions using HistGradientBoostingClassifier.

    Output JSONL format (per bar):
      {open_time, date, symbol, short:{...}, medium:{...}, long:{...}}

    ``train_window_days`` and the horizons are calendar spans; both are
    converted to bar counts for *interval*, so a 1h run forecasts the same
    real-world durations as a 1d run and the two stay comparable.
    """
    if settings is None:
        settings = get_settings()

    horizons = list(DEFAULT_HORIZONS)
    selected = set(parse_horizons(horizons_selected))
    window_overrides = parse_window_overrides(train_window_by_horizon)

    # ── Seed & deterministic mode (research) ─────────────────────────────────
    effective_seed = settings.seed if settings.seed is not None else 42
    if settings.deterministic or settings.seed is not None:
        import random

        random.seed(effective_seed)
        np.random.seed(effective_seed)
        if settings.deterministic:
            print(f"[research] Deterministic mode ON (seed={effective_seed})")
        else:
            print(f"[research] Seed set to {effective_seed}")

    depth_bins, depth_labels = parse_depth_bins(depth_bins_str)

    start_dt = parse_date(start)
    end_dt = parse_date(end)

    # 1. Load features for every pooled symbol. One path whether pooling or
    # not, so "which rows may this model see" has a single implementation.
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
    dates = [value.date() for value in open_times]
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

    # Label <-> index mapping comes from the domain, not a local literal.
    inv_dir_map = IDX_TO_DIR

    # Retrain once per calendar month to balance freshness against runtime.
    current_train_month = None
    models_cache: dict[str, Any] = {}

    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt.date() or row_date > end_dt.date():
            continue

        out: dict[str, Any] = {
            "open_time": open_times[i].isoformat(),
            "date": row_date.isoformat(),
            "symbol": symbol,
        }

        train_month = row_date.replace(day=1)

        if current_train_month != train_month:
            current_train_month = train_month
            models_cache = {}

            month_slices: dict[str, Any] = {}
            for h in horizons:
                if h.name not in selected:
                    models_cache[h.name] = None
                    continue
                # Bounded by instants, not row indices: a label at instant t
                # needs the close at t + horizon, and symbols that launched at
                # different times share no common row numbering.
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

                # Reserve the most recent slice of the *training* region to
                # calibrate on. Holding out later bars instead would break
                # invariant 6: their labels need closes the forecast cannot see.
                split, _ = calibration_split(0, len(valid_idx))
                fit_idx = list(range(split))
                cal_idx = list(range(split, len(valid_idx)))

                # Direction classifier
                clf_dir = HistGradientBoostingClassifier(
                    random_state=effective_seed, max_iter=50, max_leaf_nodes=15
                )
                # Need at least 2 classes
                y_fit = [y_train_valid[k] for k in fit_idx] if fit_idx else y_train_valid
                X_fit = X_train_valid[fit_idx] if fit_idx else X_train_valid
                if len(set(y_fit)) > 1:
                    clf_dir.fit(X_fit, y_fit)
                elif len(set(y_train_valid)) > 1:
                    # The held-out split left a single-class fit set; train on
                    # everything and skip calibration rather than lose the bar.
                    clf_dir.fit(X_train_valid, y_train_valid)
                    cal_idx = []
                else:
                    clf_dir = None  # fallback

                calibrator = fit_calibrator_from_model(
                    clf_dir,
                    X_train_valid[cal_idx] if cal_idx else np.empty((0, X_train_valid.shape[1])),
                    [y_train_valid[k] for k in cal_idx],
                    present_classes=list(clf_dir.classes_) if clf_dir is not None else None,
                )

                # Depth classifiers (long and short separately)
                idx_long = [k for k in valid_idx if inv_dir_map[int(y_dirs[k])] == "long"]
                idx_short = [k for k in valid_idx if inv_dir_map[int(y_dirs[k])] == "short"]

                clf_depth_long = None
                if len(idx_long) > 5 and len({y_depths[k] for k in idx_long}) > 1:
                    clf_depth_long = HistGradientBoostingClassifier(
                        random_state=effective_seed, max_iter=30, max_leaf_nodes=10
                    )
                    clf_depth_long.fit(X_train[idx_long], [int(y_depths[k]) for k in idx_long])

                clf_depth_short = None
                if len(idx_short) > 5 and len({y_depths[k] for k in idx_short}) > 1:
                    clf_depth_short = HistGradientBoostingClassifier(
                        random_state=effective_seed, max_iter=30, max_leaf_nodes=10
                    )
                    clf_depth_short.fit(X_train[idx_short], [int(y_depths[k]) for k in idx_short])

                # Fallback empirics for depth
                emp_long = [0] * len(depth_labels)
                for k in idx_long:
                    if y_depths[k] >= 0:
                        emp_long[int(y_depths[k])] += 1
                emp_short = [0] * len(depth_labels)
                for k in idx_short:
                    if y_depths[k] >= 0:
                        emp_short[int(y_depths[k])] += 1

                def to_prob_dict(counts: list[int]) -> dict[str, float]:
                    s = sum(counts)
                    if s == 0:
                        return {lbl: 1.0 / len(depth_labels) for lbl in depth_labels}
                    return {lbl: c / s for lbl, c in zip(depth_labels, counts, strict=True)}

                models_cache[h.name] = {
                    "clf_dir": clf_dir,
                    "clf_depth_long": clf_depth_long,
                    "clf_depth_short": clf_depth_short,
                    "fallback_depth_long": to_prob_dict(emp_long),
                    "fallback_depth_short": to_prob_dict(emp_short),
                    "calibrator": calibrator,
                    "classes_dir": clf_dir.classes_ if clf_dir else [],
                    "classes_long": clf_depth_long.classes_ if clf_depth_long else [],
                    "classes_short": clf_depth_short.classes_ if clf_depth_short else [],
                }

            if month_slices:
                pool_contributions = describe_pool(
                    month_slices, pool, forward_steps_by_horizon
                )

        # Inference for current row
        X_test = X_all[i : i + 1]

        for h in horizons:
            cache = models_cache.get(h.name)
            if not cache or not cache["clf_dir"]:
                # Not enough realised history yet: emit an explicitly untrained
                # placeholder so evaluation can exclude it instead of scoring it.
                out[h.name] = untrained_horizon_payload(depth_labels)
                untrained_counts[h.name] += 1
                continue

            probs_dir = cache["clf_dir"].predict_proba(X_test)[0]
            # map correctly
            p_dict = dict.fromkeys(DIRECTION_CLASSES, 0.0)
            for cls_idx, prob in zip(cache["classes_dir"], probs_dir, strict=True):
                p_dict[inv_dir_map[cls_idx]] = prob
            p_dict = cache["calibrator"].apply_to_mapping(p_dict)

            # Long depth
            if cache["clf_depth_long"]:
                probs_dl = cache["clf_depth_long"].predict_proba(X_test)[0]
                dl_dict = {label: 0.0 for label in depth_labels}
                for cls_idx, prob in zip(cache["classes_long"], probs_dl, strict=True):
                    dl_dict[depth_labels[cls_idx]] = prob
            else:
                dl_dict = cache["fallback_depth_long"]

            # Short depth
            if cache["clf_depth_short"]:
                probs_ds = cache["clf_depth_short"].predict_proba(X_test)[0]
                ds_dict = {label: 0.0 for label in depth_labels}
                for cls_idx, prob in zip(cache["classes_short"], probs_ds, strict=True):
                    ds_dict[depth_labels[cls_idx]] = prob
            else:
                ds_dict = cache["fallback_depth_short"]

            out[h.name] = {
                **{f"P_{d}": p_dict[d] for d in DIRECTION_CLASSES},
                "depth_long_bins": dl_dict,
                "depth_short_bins": ds_dict,
                "trained": True,
            }

        predictions.append(out)

    if not predictions:
        return {"error": "No predictions generated for the requested date range", "symbol": symbol}


    # Also write a consolidated predictions.jsonl into a versioned folder for UI convenience
    run_dir = write_versioned_predictions(
        predictions, symbol, "ml", settings, interval=interval
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
        "run_dir": str(run_dir),
    }
    write_run_summary(run_dir, result, settings)
    return result
