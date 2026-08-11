from __future__ import annotations

from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    DIRECTION_CLASSES,
    assign_depth_bin,
    direction_from_return,
    parse_depth_bins,
)
from prosper.predict.defaults import parse_horizons, parse_window_overrides
from prosper.predict.window import (
    MIN_TRAIN_SAMPLES,
    count_scored_bars,
    resolve_train_windows,
    untrained_horizon_payload,
    warn_low_power,
)
from prosper.storage.parquet import load_parquet
from prosper.storage.predictions import write_run_summary, write_versioned_predictions
from prosper.utils.time import parse_date


def _laplace_prob(count: int, total: int, k: int, alpha: float) -> float:
    return (count + alpha) / (total + alpha * k) if total >= 0 else 0.0


def predict_baseline(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    horizons_selected: str | None = None,
    rolling_window_days: int | None = None,
    train_window_by_horizon: str | None = None,
    depth_bins_str: str = DEFAULT_DEPTH_BINS_STR,
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

    # The benchmark has to be available wherever the models are. Everything
    # below is already expressed in bars of *interval*, so pinning it to 1d
    # only meant a 1h comparison had no buy-and-hold to be measured against.
    horizons = list(DEFAULT_HORIZONS)
    selected = set(parse_horizons(horizons_selected))
    window_overrides = parse_window_overrides(train_window_by_horizon)

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
    kline_parts = [
        load_parquet(path)
        for path in sorted(base_path.glob("**/*.parquet"))
        if path.is_file() and path.stat().st_size > 0
    ]

    if not kline_parts:
        return {"error": f"No {interval} parquet found for {symbol}", "symbol": symbol}

    df_bars = pl.concat(kline_parts).sort("open_time").unique(subset=["open_time"], keep="first")
    df_bars = df_bars.with_columns(pl.col("open_time").dt.date().alias("_date"))

    if df_bars.is_empty():
        return {"error": f"No {interval} rows found for {symbol}", "symbol": symbol}

    open_times = df_bars["open_time"].to_list()
    dates = df_bars["_date"].to_list()
    closes = df_bars["close"].to_list()
    n = len(df_bars)

    # After the load, not before: the window a horizon should get depends on how
    # much history there is to divide between training it and scoring it. The
    # benchmark counts frequencies over the same window the models train on, or
    # it would not be answering the same question.
    windows = resolve_train_windows(
        rolling_window_days, interval, horizons, window_overrides, history_bars=n
    )
    steps_by_horizon = windows.forward_steps
    scored_bars = count_scored_bars(dates, start_dt.date(), end_dt.date(), windows)
    warn_low_power(windows, scored_bars)

    def horizon_labels(forward_steps: int) -> dict[str, list[Any]]:
        directions: list[str | None] = [None] * n
        depth_bins_idx: list[int | None] = [None] * n

        for i in range(n):
            j = i + forward_steps
            if j >= n:
                continue
            r = closes[j] / closes[i] - 1.0
            directions[i] = direction_from_return(r)
            if directions[i] is not None:
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
            if h.name not in selected:
                out[h.name] = untrained_horizon_payload(depth_labels)
                untrained_counts[h.name] += 1
                continue
            labels = horizon_data[h.name]
            dirs = labels["direction"]
            d_idx = labels["depth_bin_idx"]

            # Only outcomes realised strictly before bar i are observable.
            train_start, train_end = windows.bounds(i, h.name)
            window_idx = [j for j in range(train_start, train_end) if dirs[j] is not None]

            if len(window_idx) < MIN_TRAIN_SAMPLES:
                out[h.name] = untrained_horizon_payload(depth_labels)
                untrained_counts[h.name] += 1
                continue

            total_dir = len(window_idx)
            k_dir = len(DIRECTION_CLASSES)
            direction_probs = {
                direction: _laplace_prob(
                    sum(1 for j in window_idx if dirs[j] == direction),
                    total_dir,
                    k_dir,
                    alpha,
                )
                for direction in DIRECTION_CLASSES
            }

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
                **{f"P_{d}": prob for d, prob in direction_probs.items()},
                "depth_long_bins": depth_probs_for("long"),
                "depth_short_bins": depth_probs_for("short"),
                "trained": True,
            }

        predictions.append(out)


    run_dir = write_versioned_predictions(
        predictions, symbol, "baseline", settings, interval=interval
    )

    result = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "predictions": len(predictions),
        "untrained_horizons": untrained_counts,
        "training_windows": windows.describe(scored_bars),
        "run_dir": str(run_dir),
    }
    write_run_summary(run_dir, result, settings)
    return result
