from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import polars as pl

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_parquet_file_path, get_prediction_report_path
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
    """Assign a positive percentage value to a depth-bin index."""
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
    if r is None:
        return None
    if abs(r) <= flat_threshold:
        return "flat"
    if r > flat_threshold:
        return "long"
    return "short"


def _laplace_prob(count: int, total: int, k: int, alpha: float) -> float:
    return (count + alpha) / (total + alpha * k) if total >= 0 else 0.0


def predict_baseline(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    rolling_window_days: int | None = None,
    flat_threshold: float = 0.01,
    depth_bins_str: str = "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+",
    short_forward_days: int = 28,
    medium_forward_days: int = 182,
    long_forward_days: int = 365,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """
    Baseline probabilistic predictions using rolling empirical frequencies + Laplace smoothing.

    Output JSONL format (per day):
      {date, symbol, short:{...}, medium:{...}, long:{...}}
    """
    if settings is None:
        settings = get_settings()
    if rolling_window_days is None:
        rolling_window_days = settings.baseline_rolling_window_days

    depth_bins, depth_labels = parse_depth_bins(depth_bins_str)
    depth_k = len(depth_bins)

    start_dt = parse_date(start)
    end_dt = parse_date(end)

    # Load daily klines across months in [start, end]
    months: list[tuple[int, int]] = []
    y, m = start_dt.year, start_dt.month
    y2, m2 = end_dt.year, end_dt.month
    while (y, m) <= (y2, m2):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1

    daily_parts: list[pl.DataFrame] = []
    for yy, mm in months:
        path = get_parquet_file_path(symbol, "1d", yy, month=mm, settings=settings)
        if path.exists():
            daily_parts.append(load_parquet(path))

    if not daily_parts:
        return {"error": f"No daily parquet found for {symbol} at 1d", "symbol": symbol}

    df_daily = pl.concat(daily_parts).sort("open_time").unique(subset=["open_time"], keep="first")
    df_daily = df_daily.with_columns(pl.col("open_time").dt.date().alias("_date"))
    df_daily = df_daily.filter(pl.col("_date") >= start_dt.date()).filter(pl.col("_date") <= end_dt.date())

    if df_daily.is_empty():
        return {"error": "No daily rows for requested date range", "symbol": symbol}

    # Prepare horizon labels once per horizon (direction + depth_bin)
    horizons = [
        HorizonSpec("short", short_forward_days),
        HorizonSpec("medium", medium_forward_days),
        HorizonSpec("long", long_forward_days),
    ]

    # Convert to python lists for fast rolling-window loops.
    # We avoid datetime->Python conversion issues by keeping _date as string.
    df_daily = df_daily.with_columns(pl.col("_date").dt.strftime("%Y-%m-%d").alias("_date_str"))
    dates = df_daily["_date_str"].to_list()
    closes = df_daily["close"].to_list()

    n = len(df_daily)

    def horizon_labels(forward_days: int) -> dict[str, list[Any]]:
        directions: list[str | None] = [None] * n
        depth_bins_idx: list[int | None] = [None] * n
        returns: list[float | None] = [None] * n

        for i in range(n):
            j = i + forward_days
            if j >= n:
                continue
            r = closes[j] / closes[i] - 1.0
            returns[i] = r
            directions[i] = _direction_from_return(r, flat_threshold)
            if directions[i] in ("long", "short"):
                depth_pct = abs(r) * 100.0
                depth_bins_idx[i] = assign_depth_bin(depth_pct, depth_bins)
        return {"direction": directions, "depth_bin_idx": depth_bins_idx, "return": returns}

    horizon_data = {h.name: horizon_labels(h.forward_days) for h in horizons}

    # Rolling window is index-based: use at most `rolling_window_days` previous as-of days.
    # (For dense daily data this is equivalent to date-based.)
    def window_slice(i: int) -> range:
        j0 = max(0, i - rolling_window_days)
        return range(j0, i)

    predictions: list[dict[str, Any]] = []
    for i in range(n):
        date_str = dates[i]
        out: dict[str, Any] = {"date": date_str, "symbol": symbol}

        for h in horizons:
            labels = horizon_data[h.name]
            dirs = labels["direction"]
            d_idx = labels["depth_bin_idx"]

            window_idx = list(window_slice(i))
            # Direction counts among available label rows in the window
            train_dirs = [dirs[j] for j in window_idx if dirs[j] is not None]
            total_dir = len(train_dirs)
            c_long = sum(1 for d in train_dirs if d == "long")
            c_flat = sum(1 for d in train_dirs if d == "flat")
            c_short = sum(1 for d in train_dirs if d == "short")

            # Laplace smoothing for direction
            k_dir = 3
            p_long = _laplace_prob(c_long, total_dir, k_dir, alpha)
            p_flat = _laplace_prob(c_flat, total_dir, k_dir, alpha)
            p_short = _laplace_prob(c_short, total_dir, k_dir, alpha)

            # Depth conditional probabilities
            def depth_probs_for(direction: str) -> dict[str, float]:
                # counts per bin
                idxs = [
                    d_idx[j]
                    for j in window_idx
                    if dirs[j] == direction and d_idx[j] is not None
                ]
                total = len(idxs)
                counts = [0] * depth_k
                for bi in idxs:
                    if bi is not None:
                        counts[int(bi)] += 1

                probs: dict[str, float] = {}
                for bi, label in enumerate(depth_labels):
                    probs[label] = _laplace_prob(counts[bi], total, depth_k, alpha)
                return probs

            depth_long_bins = depth_probs_for("long")
            depth_short_bins = depth_probs_for("short")

            out[h.name] = {
                "P_long": p_long,
                "P_flat": p_flat,
                "P_short": p_short,
                "depth_long_bins": depth_long_bins,
                "depth_short_bins": depth_short_bins,
            }

        predictions.append(out)

    # Write per-month JSONL (idempotent overwrite for this date range)
    out_dir = settings.reports_predictions_dir / symbol / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by month string
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

