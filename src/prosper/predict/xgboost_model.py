from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from prosper.config import Settings, get_settings
from prosper.labels.depth import HorizonSpec, direction_from_return, parse_depth_bins
from prosper.storage.layout import get_features_parquet_path
from prosper.storage.predictions import write_predictions_jsonl
from prosper.utils.time import parse_date


def predict_xgboost(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    train_window_days: int = 150,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    flat_threshold: float = 0.01,
    short_forward_days: int = 28,
    medium_forward_days: int = 182,
    long_forward_days: int = 365,
) -> dict[str, Any]:
    """
    XGBoost probabilistic direction predictions (short/medium/long).

    Lightweight implementation: trains `XGBClassifier` per horizon and
    writes monthly JSONL prediction files. Also exports `feature_importances_`
    to a versioned folder under reports/predictions/{symbol}/xgboost_{timestamp}/
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

    effective_seed = settings.seed if settings.seed is not None else 42
    if settings.deterministic or settings.seed is not None:
        import random

        random.seed(effective_seed)
        np.random.seed(effective_seed)

    depth_bins, depth_labels = parse_depth_bins("1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+")

    start_dt = parse_date(start)
    end_dt = parse_date(end)

    feat_path = get_features_parquet_path(symbol, "1d", settings=settings)
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

    horizons = [
        HorizonSpec("short", short_forward_days),
        HorizonSpec("medium", medium_forward_days),
        HorizonSpec("long", long_forward_days),
    ]

    dates = df_feat["_date"].to_list()
    closes = df_feat["close"].to_list()
    n = len(df_feat)

    X_all = df_feat.select(feature_cols).to_numpy()

    # Pre-calc direction targets
    horizon_targets = {}
    for h in horizons:
        directions = [None] * n
        for i in range(n):
            j = i + h.forward_days
            if j < n:
                r = closes[j] / closes[i] - 1.0
                directions[i] = direction_from_return(r, flat_threshold)
        horizon_targets[h.name] = {"direction": directions}

    predictions: list[dict[str, Any]] = []

    dir_map = {"short": 0, "flat": 1, "long": 2}
    inv_dir_map = {0: "short", 1: "flat", 2: "long"}

    current_train_month = None
    models_cache: dict[str, Any] = {}

    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt.date() or row_date > end_dt.date():
            continue

        train_end_idx = i - 1
        train_start_idx = max(0, i - train_window_days)

        if train_end_idx < 10:
            continue

        train_month = row_date.replace(day=1)

        if current_train_month != train_month:
            current_train_month = train_month
            models_cache = {}

            for h in horizons:
                y_dirs = horizon_targets[h.name]["direction"][train_start_idx:train_end_idx]
                X_train = X_all[train_start_idx:train_end_idx]

                valid_idx = [k for k, d in enumerate(y_dirs) if d is not None]

                if len(valid_idx) < 10:
                    models_cache[h.name] = None
                    continue

                X_train_valid = X_train[valid_idx]
                y_train_valid = [dir_map[y_dirs[k]] for k in valid_idx]

                try:
                    clf = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=effective_seed,
                        eval_metric="mlogloss",
                    )
                    if len(set(y_train_valid)) > 1:
                        clf.fit(X_train_valid, y_train_valid)
                    else:
                        clf = None
                except Exception:
                    clf = None

                models_cache[h.name] = {"clf": clf, "classes": clf.classes_ if clf else []}

    # Inference: use latest trained models to produce probabilities per date
    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt.date() or row_date > end_dt.date():
            continue

        row_date_str = row_date.isoformat()
        out: dict[str, Any] = {"date": row_date_str, "symbol": symbol}

        X_test = X_all[i : i + 1]
        for h in horizons:
            cache = models_cache.get(h.name)
            if not cache or not cache.get("clf"):
                out[h.name] = {
                    "P_long": 0.33,
                    "P_flat": 0.34,
                    "P_short": 0.33,
                    "depth_long_bins": {label: 1.0 / len(depth_labels) for label in depth_labels},
                    "depth_short_bins": {label: 1.0 / len(depth_labels) for label in depth_labels},
                }
                continue

            probs = cache["clf"].predict_proba(X_test)[0]
            p_dict = {"short": 0.0, "flat": 0.0, "long": 0.0}
            for cls_idx, prob in zip(cache["classes"], probs):
                p_dict[inv_dir_map[cls_idx]] = prob

            out[h.name] = {
                "P_long": p_dict["long"],
                "P_flat": p_dict["flat"],
                "P_short": p_dict["short"],
                "depth_long_bins": {label: 1.0 / len(depth_labels) for label in depth_labels},
                "depth_short_bins": {label: 1.0 / len(depth_labels) for label in depth_labels},
            }

        predictions.append(out)

    # Persist predictions grouped by month
    write_predictions_jsonl(predictions, symbol, settings)

    # Export feature importances for the last trained models (if available)
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        out_dir = settings.reports_predictions_dir / symbol / f"xgboost_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fi: dict[str, dict[str, float]] = {}
        for h in horizons:
            cache = models_cache.get(h.name)
            if cache and cache.get("clf") and hasattr(cache["clf"], "feature_importances_"):
                importances = cache["clf"].feature_importances_
                fi[h.name] = {col: float(imp) for col, imp in zip(feature_cols, importances)}
        if fi:
            (out_dir / "feature_importances.json").write_text(
                json.dumps(fi, indent=2), encoding="utf-8"
            )
        # Also write a consolidated predictions.jsonl inside the versioned folder for UI convenience
        if predictions:
            with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
                for r in predictions:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
    except Exception:
        pass

    return {"symbol": symbol, "start": start, "end": end, "predictions": len(predictions)}
