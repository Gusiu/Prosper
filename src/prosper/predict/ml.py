from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_features_parquet_path, get_prediction_report_path
from prosper.storage.parquet import load_parquet
from prosper.utils.time import parse_date


def parse_depth_bins(bins_str: str) -> tuple[list[tuple[float, float | None]], list[str]]:
    """Parse '1-2,2-3,...,34+' into ranges and stable labels."""
    bins: list[tuple[float, float | None]] = []
    labels: list[str] = []
    parts = [p.strip() for p in bins_str.split(",") if p.strip()]
    for part in parts:
        if part.endswith("+"):
            min_val = float(part[:-1])
            bins.append((min_val, None))
            labels.append(f"{int(min_val)}+")
        else:
            lo, hi = part.split("-", 1)
            min_val = float(lo)
            max_val = float(hi)
            bins.append((min_val, max_val))
            labels.append(f"{int(min_val)}-{int(max_val)}")
    return bins, labels


def assign_depth_bin(value_pct: float, bins: list[tuple[float, float | None]]) -> int | None:
    for idx, (min_val, max_val) in enumerate(bins):
        if max_val is None:
            if value_pct >= min_val:
                return idx
        else:
            if min_val <= value_pct < max_val:
                return idx
    return None


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    forward_days: int


def _direction_from_return(r: float, flat_threshold: float) -> str | None:
    if r is None or math.isnan(r):
        return None
    if abs(r) <= flat_threshold:
        return "flat"
    if r > flat_threshold:
        return "long"
    return "short"


def predict_ml(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    train_window_days: int = 150,
    flat_threshold: float = 0.01,
    depth_bins_str: str = "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+",
    short_forward_days: int = 28,
    medium_forward_days: int = 182,
    long_forward_days: int = 365,
) -> dict[str, Any]:
    """
    ML predictions using HistGradientBoostingClassifier.
    Output JSONL format (per day):
      {date, symbol, short:{...}, medium:{...}, long:{...}}
    """
    if settings is None:
        settings = get_settings()

    depth_bins, depth_labels = parse_depth_bins(depth_bins_str)
    
    start_dt = parse_date(start)
    end_dt = parse_date(end)

    # 1. Load Features
    feat_path = get_features_parquet_path(symbol, "1d", settings=settings)
    if not feat_path.exists():
        return {"error": f"No features parquet found for {symbol}. Run features build.", "symbol": symbol}
    
    # Disable hive partitioning to avoid duplicate schema errors if 'symbol' is already a column
    df_feat = pl.read_parquet(feat_path, hive_partitioning=False).sort("open_time")
    
    if df_feat.is_empty():
        return {"error": "Empty features DataFrame", "symbol": symbol}
        
    df_feat = df_feat.with_columns(pl.col("open_time").dt.date().alias("_date"))
    
    # 2. Extract features matrix
    # Exclude non-feature columns
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
    
    # Pre-calculate horizon targets
    horizon_targets = {}
    for h in horizons:
        directions = [None] * n
        returns = [None] * n
        depths = [None] * n
        for i in range(n):
            j = i + h.forward_days
            if j < n:
                r = closes[j] / closes[i] - 1.0
                returns[i] = r
                d = _direction_from_return(r, flat_threshold)
                directions[i] = d
                if d in ("long", "short"):
                    depths[i] = assign_depth_bin(abs(r) * 100.0, depth_bins)
        horizon_targets[h.name] = {
            "direction": directions,
            "depth": depths
        }
        
    predictions: list[dict[str, Any]] = []
    
    # Helper to map labels to int for classifier
    dir_map = {"short": 0, "flat": 1, "long": 2}
    inv_dir_map = {0: "short", 1: "flat", 2: "long"}

    # Evaluate sequentially (expanding window or rolling)
    # To keep it simple and fast, we'll train one model per month
    current_train_month = None
    models_cache = {}

    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt.date() or row_date > end_dt.date():
            continue
            
        row_date_str = row_date.isoformat()
        out: dict[str, Any] = {"date": row_date_str, "symbol": symbol}
        
        train_end_idx = i - 1
        train_start_idx = max(0, i - train_window_days)
        
        # If we don't have enough data, skip or output uniform
        if train_end_idx < 10:
            for h in horizons:
                out[h.name] = {
                    "P_long": 0.33, "P_flat": 0.34, "P_short": 0.33,
                    "depth_long_bins": {label: 1.0/len(depth_labels) for label in depth_labels},
                    "depth_short_bins": {label: 1.0/len(depth_labels) for label in depth_labels},
                }
            predictions.append(out)
            continue
            
        train_month = row_date.replace(day=1)
        
        # Retrain models once a month to save time
        if current_train_month != train_month:
            current_train_month = train_month
            models_cache = {}
            
            # Train model for each horizon
            for h in horizons:
                y_dirs = horizon_targets[h.name]["direction"][train_start_idx:train_end_idx]
                y_depths = horizon_targets[h.name]["depth"][train_start_idx:train_end_idx]
                X_train = X_all[train_start_idx:train_end_idx]
                
                # Filter valid labels (no None)
                valid_idx = [k for k, d in enumerate(y_dirs) if d is not None]
                
                if len(valid_idx) < 10:
                    models_cache[h.name] = None
                    continue
                    
                X_train_valid = X_train[valid_idx]
                y_train_valid = [dir_map[y_dirs[k]] for k in valid_idx]
                
                # Direction classifier
                clf_dir = HistGradientBoostingClassifier(random_state=42, max_iter=50, max_leaf_nodes=15)
                # Need at least 2 classes
                if len(set(y_train_valid)) > 1:
                    clf_dir.fit(X_train_valid, y_train_valid)
                else:
                    clf_dir = None # fallback
                
                # Depth classifiers (long and short separately)
                idx_long = [k for k in valid_idx if y_dirs[k] == "long"]
                idx_short = [k for k in valid_idx if y_dirs[k] == "short"]
                
                clf_depth_long = None
                if len(idx_long) > 5 and len(set(y_depths[k] for k in idx_long)) > 1:
                    clf_depth_long = HistGradientBoostingClassifier(random_state=42, max_iter=30, max_leaf_nodes=10)
                    clf_depth_long.fit(X_train[idx_long], [y_depths[k] for k in idx_long])
                    
                clf_depth_short = None
                if len(idx_short) > 5 and len(set(y_depths[k] for k in idx_short)) > 1:
                    clf_depth_short = HistGradientBoostingClassifier(random_state=42, max_iter=30, max_leaf_nodes=10)
                    clf_depth_short.fit(X_train[idx_short], [y_depths[k] for k in idx_short])
                
                # Fallback empirics for depth
                emp_long = [0]*len(depth_labels)
                for k in idx_long: emp_long[y_depths[k]] += 1
                emp_short = [0]*len(depth_labels)
                for k in idx_short: emp_short[y_depths[k]] += 1
                
                def to_prob_dict(counts):
                    s = sum(counts)
                    if s == 0: return {lbl: 1.0/len(depth_labels) for lbl in depth_labels}
                    return {lbl: c/s for lbl, c in zip(depth_labels, counts)}
                
                models_cache[h.name] = {
                    "clf_dir": clf_dir,
                    "clf_depth_long": clf_depth_long,
                    "clf_depth_short": clf_depth_short,
                    "fallback_depth_long": to_prob_dict(emp_long),
                    "fallback_depth_short": to_prob_dict(emp_short),
                    "classes_dir": clf_dir.classes_ if clf_dir else [],
                    "classes_long": clf_depth_long.classes_ if clf_depth_long else [],
                    "classes_short": clf_depth_short.classes_ if clf_depth_short else []
                }
                
        # Inference for current row
        X_test = X_all[i:i+1]
        
        for h in horizons:
            cache = models_cache.get(h.name)
            if not cache or not cache["clf_dir"]:
                out[h.name] = {
                    "P_long": 0.33, "P_flat": 0.34, "P_short": 0.33,
                    "depth_long_bins": {label: 1.0/len(depth_labels) for label in depth_labels},
                    "depth_short_bins": {label: 1.0/len(depth_labels) for label in depth_labels},
                }
                continue
                
            probs_dir = cache["clf_dir"].predict_proba(X_test)[0]
            # map correctly
            p_dict = {"short": 0.0, "flat": 0.0, "long": 0.0}
            for cls_idx, prob in zip(cache["classes_dir"], probs_dir):
                p_dict[inv_dir_map[cls_idx]] = prob
                
            # Long depth
            if cache["clf_depth_long"]:
                probs_dl = cache["clf_depth_long"].predict_proba(X_test)[0]
                dl_dict = {label: 0.0 for label in depth_labels}
                for cls_idx, prob in zip(cache["classes_long"], probs_dl):
                    dl_dict[depth_labels[cls_idx]] = prob
            else:
                dl_dict = cache["fallback_depth_long"]
                
            # Short depth
            if cache["clf_depth_short"]:
                probs_ds = cache["clf_depth_short"].predict_proba(X_test)[0]
                ds_dict = {label: 0.0 for label in depth_labels}
                for cls_idx, prob in zip(cache["classes_short"], probs_ds):
                    ds_dict[depth_labels[cls_idx]] = prob
            else:
                ds_dict = cache["fallback_depth_short"]
                
            out[h.name] = {
                "P_long": p_dict["long"],
                "P_flat": p_dict["flat"],
                "P_short": p_dict["short"],
                "depth_long_bins": dl_dict,
                "depth_short_bins": ds_dict,
            }
            
        predictions.append(out)

    # Write per-month JSONL (idempotent overwrite for this date range)
    out_dir = settings.reports_predictions_dir / symbol / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_month: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        month_key = row["date"][:7]
        by_month.setdefault(month_key, []).append(row)

    for month_key, rows in by_month.items():
        year = int(month_key[:4])
        month = int(month_key[5:7])
        report_path = get_prediction_report_path(symbol, year, month, settings=settings)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "predictions": len(predictions),
    }
