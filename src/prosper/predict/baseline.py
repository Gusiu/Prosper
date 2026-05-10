from __future__ import annotations

from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.labels.depth import (
    HorizonSpec,
    assign_depth_bin,
    direction_from_return,
    parse_depth_bins,
)
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import load_parquet
from prosper.storage.predictions import write_predictions_jsonl
from prosper.utils.time import generate_month_range, parse_date


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
    """Baseline probabilistic predictions using rolling empirical frequencies + Laplace smoothing.

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
    months = generate_month_range(start_dt.date(), end_dt.date())

    daily_parts: list[pl.DataFrame] = []
    for yy, mm in months:
        path = get_parquet_file_path(symbol, "1d", yy, month=mm, settings=settings)
        if path.exists():
            daily_parts.append(load_parquet(path))

    if not daily_parts:
        return {"error": f"No daily parquet found for {symbol} at 1d", "symbol": symbol}

    df_daily = pl.concat(daily_parts).sort("open_time").unique(subset=["open_time"], keep="first")
    df_daily = df_daily.with_columns(pl.col("open_time").dt.date().alias("_date"))
    df_daily = df_daily.filter(pl.col("_date") >= start_dt.date()).filter(
        pl.col("_date") <= end_dt.date()
    )

    if df_daily.is_empty():
        return {"error": "No daily rows for requested date range", "symbol": symbol}

    # Prepare horizon labels once per horizon (direction + depth_bin)
    horizons = [
        HorizonSpec("short", short_forward_days),
        HorizonSpec("medium", medium_forward_days),
        HorizonSpec("long", long_forward_days),
    ]

    # Convert to python lists for fast rolling-window loops.
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
            directions[i] = direction_from_return(r, flat_threshold)
            if directions[i] in ("long", "short"):
                depth_pct = abs(r) * 100.0
                depth_bins_idx[i] = assign_depth_bin(depth_pct, depth_bins)
        return {"direction": directions, "depth_bin_idx": depth_bins_idx, "return": returns}

    horizon_data = {h.name: horizon_labels(h.forward_days) for h in horizons}

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
            train_dirs = [dirs[j] for j in window_idx if dirs[j] is not None]
            total_dir = len(train_dirs)
            c_long = sum(1 for d in train_dirs if d == "long")
            c_flat = sum(1 for d in train_dirs if d == "flat")
            c_short = sum(1 for d in train_dirs if d == "short")

            k_dir = 3
            p_long = _laplace_prob(c_long, total_dir, k_dir, alpha)
            p_flat = _laplace_prob(c_flat, total_dir, k_dir, alpha)
            p_short = _laplace_prob(c_short, total_dir, k_dir, alpha)

            def depth_probs_for(direction: str) -> dict[str, float]:
                idxs = [
                    d_idx[j] for j in window_idx if dirs[j] == direction and d_idx[j] is not None
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

    write_predictions_jsonl(predictions, symbol, settings)

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "predictions": len(predictions),
    }
