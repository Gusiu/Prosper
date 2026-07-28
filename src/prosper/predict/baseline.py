from __future__ import annotations

from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_HORIZONS,
    assign_depth_bin,
    direction_from_return,
    parse_depth_bins,
)
from prosper.predict.window import (
    MIN_TRAIN_SAMPLES,
    days_to_steps,
    training_bounds,
    untrained_horizon_payload,
    validate_train_window,
)
from prosper.storage.parquet import load_parquet
from prosper.storage.predictions import write_versioned_predictions
from prosper.utils.time import parse_date


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
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Baseline probabilistic predictions using rolling empirical frequencies + Laplace smoothing.

    Output JSONL format (per day):
      {open_time, date, symbol, short:{...}, medium:{...}, long:{...}}

    The rolling window only counts outcomes that had already been realised at
    the prediction bar. Counting the most recent ``forward_days`` bars would
    mean reading returns that had not happened yet, which is exactly the
    advantage a benchmark must not have.
    """
    if settings is None:
        settings = get_settings()
    if rolling_window_days is None:
        rolling_window_days = settings.baseline_rolling_window_days

    # The baseline reads daily klines directly, so its bar size is fixed at 1d.
    interval = "1d"
    horizons = list(DEFAULT_HORIZONS)
    steps_by_horizon = validate_train_window(rolling_window_days, interval, horizons)
    window_steps = days_to_steps(rolling_window_days, interval)

    depth_bins, depth_labels = parse_depth_bins(depth_bins_str)
    depth_k = len(depth_bins)

    start_dt = parse_date(start)
    end_dt = parse_date(end)

    # Load every available month, not just the requested range: the rolling
    # window looks *back* from the first predicted bar. Loading only [start,
    # end] left the early rows with almost no history, so the benchmark got
    # weaker the narrower the range you asked for — while the ML models, which
    # read all features and filter on output, did not.
    base_path = settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
    daily_parts = [
        load_parquet(path)
        for path in sorted(base_path.glob("**/*.parquet"))
        if path.is_file() and path.stat().st_size > 0
    ]

    if not daily_parts:
        return {"error": f"No daily parquet found for {symbol} at {interval}", "symbol": symbol}

    df_daily = pl.concat(daily_parts).sort("open_time").unique(subset=["open_time"], keep="first")
    df_daily = df_daily.with_columns(pl.col("open_time").dt.date().alias("_date"))

    if df_daily.is_empty():
        return {"error": f"No daily rows found for {symbol}", "symbol": symbol}

    open_times = df_daily["open_time"].to_list()
    dates = df_daily["_date"].to_list()
    closes = df_daily["close"].to_list()
    n = len(df_daily)

    def horizon_labels(forward_steps: int) -> dict[str, list[Any]]:
        directions: list[str | None] = [None] * n
        depth_bins_idx: list[int | None] = [None] * n

        for i in range(n):
            j = i + forward_steps
            if j >= n:
                continue
            r = closes[j] / closes[i] - 1.0
            directions[i] = direction_from_return(r, flat_threshold)
            if directions[i] in ("long", "short"):
                depth_bins_idx[i] = assign_depth_bin(abs(r) * 100.0, depth_bins)
        return {"direction": directions, "depth_bin_idx": depth_bins_idx}

    horizon_data = {h.name: horizon_labels(steps_by_horizon[h.name]) for h in horizons}

    predictions: list[dict[str, Any]] = []
    untrained_counts: dict[str, int] = {h.name: 0 for h in horizons}

    for i in range(n):
        if dates[i] < start_dt.date() or dates[i] > end_dt.date():
            continue

        out: dict[str, Any] = {
            "open_time": open_times[i].isoformat(),
            "date": dates[i].isoformat(),
            "symbol": symbol,
        }

        for h in horizons:
            labels = horizon_data[h.name]
            dirs = labels["direction"]
            d_idx = labels["depth_bin_idx"]

            # Only outcomes realised strictly before bar i are observable.
            train_start, train_end = training_bounds(
                i, window_steps, steps_by_horizon[h.name]
            )
            window_idx = [j for j in range(train_start, train_end) if dirs[j] is not None]

            if len(window_idx) < MIN_TRAIN_SAMPLES:
                out[h.name] = untrained_horizon_payload(depth_labels)
                untrained_counts[h.name] += 1
                continue

            total_dir = len(window_idx)
            c_long = sum(1 for j in window_idx if dirs[j] == "long")
            c_flat = sum(1 for j in window_idx if dirs[j] == "flat")
            c_short = sum(1 for j in window_idx if dirs[j] == "short")

            k_dir = 3
            p_long = _laplace_prob(c_long, total_dir, k_dir, alpha)
            p_flat = _laplace_prob(c_flat, total_dir, k_dir, alpha)
            p_short = _laplace_prob(c_short, total_dir, k_dir, alpha)

            def depth_probs_for(direction: str, window_idx: list[int] = window_idx) -> dict[str, float]:
                idxs = [
                    d_idx[j] for j in window_idx if dirs[j] == direction and d_idx[j] is not None
                ]
                total = len(idxs)
                counts = [0] * depth_k
                for bi in idxs:
                    counts[int(bi)] += 1

                return {
                    label: _laplace_prob(counts[bi], total, depth_k, alpha)
                    for bi, label in enumerate(depth_labels)
                }

            out[h.name] = {
                "P_long": p_long,
                "P_flat": p_flat,
                "P_short": p_short,
                "depth_long_bins": depth_probs_for("long"),
                "depth_short_bins": depth_probs_for("short"),
                "trained": True,
            }

        predictions.append(out)


    run_dir = write_versioned_predictions(
        predictions, symbol, "baseline", settings, interval=interval
    )

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "predictions": len(predictions),
        "untrained_horizons": untrained_counts,
        "run_dir": str(run_dir),
    }
