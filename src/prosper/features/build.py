from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_features_parquet_path, get_parquet_file_path
from prosper.storage.parquet import load_parquet, save_parquet
from prosper.utils.time import parse_date


def _month_iter(start_py_date: Any, end_py_date: Any) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    y, m = start_py_date.year, start_py_date.month
    y2, m2 = end_py_date.year, end_py_date.month
    while (y, m) <= (y2, m2):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def _rsi_ewm(close: pl.Expr, period: int = 14) -> pl.Expr:
    delta = close.diff(1)
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)

    # Wilder-like smoothing approximation via ewm (span=period).
    avg_gain = gain.ewm_mean(span=period, adjust=False)
    avg_loss = loss.ewm_mean(span=period, adjust=False)

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # avg_loss == 0 => RSI=100
    return pl.when(avg_loss == 0).then(100.0).otherwise(rsi)


def _ema(expr: pl.Expr, span: int) -> pl.Expr:
    return expr.ewm_mean(span=span, adjust=False)


def build_features(
    symbol: str,
    base_interval: str,
    start: str,
    end: str,
    jump_threshold: float = 0.02,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Build minimal daily+intraday features parquet.

    Notes:
    - We intentionally keep rolling-window warmup as nulls.
    - Intraday features use 1m minute returns aggregated per UTC day.
    """

    if settings is None:
        settings = get_settings()
    if base_interval != "1d":
        raise ValueError("MVP supports base_interval=1d only")

    start_dt = parse_date(start)
    end_dt = parse_date(end)
    # inclusive filtering by date (UTC)
    start_date = start_dt.date()
    end_date = end_dt.date()

    # Load daily klines
    months = _month_iter(start_date, end_date)
    daily_parts: list[pl.DataFrame] = []
    for y, m in months:
        path = get_parquet_file_path(symbol, "1d", y, month=m, settings=settings)
        if path.exists():
            daily_parts.append(load_parquet(path))

    if not daily_parts:
        out_path = get_features_parquet_path(symbol, base_interval, settings=settings)
        return {"error": f"No daily parquet found for {symbol} in {base_interval}", "path": str(out_path)}

    df_daily = pl.concat(daily_parts).sort("open_time").unique(subset=["open_time"], keep="first")
    df_daily = df_daily.with_columns(pl.col("open_time").dt.date().alias("_date"))
    df_daily = (
        df_daily.filter(pl.col("_date") >= start_date)
        .filter(pl.col("_date") <= end_date)
        .drop("_date")
    )

    # Daily returns/indicators
    log_close = pl.col("close").log()
    df_feat = df_daily.with_columns(
        [
            (log_close - log_close.shift(1)).alias("log_return_1d"),
            (log_close - log_close.shift(3)).alias("log_return_3d"),
            (log_close - log_close.shift(7)).alias("log_return_7d"),
            (log_close - log_close.shift(14)).alias("log_return_14d"),
        ]
    )

    df_feat = df_feat.with_columns(
        [
            pl.col("log_return_1d").rolling_std(window_size=7).alias("rolling_vol_7"),
            pl.col("log_return_1d").rolling_std(window_size=14).alias("rolling_vol_14"),
            pl.col("log_return_1d").rolling_std(window_size=30).alias("rolling_vol_30"),
        ]
    )

    df_feat = df_feat.with_columns(
        [
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("daily_range"),
            pl.col("close").rolling_max(window_size=30).alias("_roll_max_30"),
            pl.col("close").rolling_max(window_size=90).alias("_roll_max_90"),
        ]
    )
    df_feat = df_feat.with_columns(
        [
            (pl.col("close") / pl.col("_roll_max_30") - 1.0).alias("rolling_drawdown_30"),
            (pl.col("close") / pl.col("_roll_max_90") - 1.0).alias("rolling_drawdown_90"),
        ]
    ).drop(["_roll_max_30", "_roll_max_90"])

    df_feat = df_feat.with_columns(
        [
            pl.col("close").rolling_mean(window_size=10).alias("ma_10"),
            pl.col("close").rolling_mean(window_size=30).alias("ma_30"),
        ]
    )
    df_feat = df_feat.with_columns(
        [
            (pl.col("ma_10") - pl.col("ma_30")).alias("ma_diff"),
            (pl.col("ma_10") - pl.col("ma_10").shift(1)).alias("ma_slope"),
        ]
    )

    df_feat = df_feat.with_columns([_rsi_ewm(pl.col("close"), 14).alias("rsi_14")])
    # Clamp for numeric safety
    df_feat = df_feat.with_columns(pl.col("rsi_14").clip(0.0, 100.0).alias("rsi_14"))

    prev_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        [
            (pl.col("high") - pl.col("low")).abs(),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        ]
    )
    df_feat = df_feat.with_columns(
        [
            true_range.alias("_true_range"),
            true_range.ewm_mean(span=14, adjust=False).alias("atr_14"),
        ]
    ).drop(["_true_range"])

    ema12 = _ema(pl.col("close"), 12)
    ema26 = _ema(pl.col("close"), 26)
    macd = (ema12 - ema26).alias("_macd")
    signal = _ema(pl.col("_macd"), 9).alias("_signal")
    df_feat = df_feat.with_columns([macd]).with_columns([signal]).with_columns(
        [(pl.col("_macd") - pl.col("_signal")).alias("macd_hist")]
    ).drop(["_macd", "_signal"])

    # Intraday derived features
    intraday_parts: list[pl.DataFrame] = []
    for y, m in months:
        path = get_parquet_file_path(symbol, "1m", y, month=m, settings=settings)
        if path.exists():
            intraday_parts.append(load_parquet(path))
    if not intraday_parts:
        raise FileNotFoundError(f"No 1m parquet found for intraday features: {symbol}")

    df_1m = pl.concat(intraday_parts).sort("open_time").unique(subset=["open_time"], keep="first")
    df_1m = df_1m.with_columns(pl.col("open_time").dt.date().alias("_date"))
    df_1m = df_1m.filter(pl.col("_date") >= start_date).filter(pl.col("_date") <= end_date)

    df_1m = df_1m.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1).over("_date") - 1.0).alias("minute_return"),
        ]
    )
    df_1m = df_1m.with_columns(
        [
            pl.col("minute_return").pow(2).alias("_minute_return_sq"),
            (pl.col("close") / pl.col("close").cummax().over("_date") - 1.0).alias("_intraday_drawdown"),
        ]
    )

    df_intraday = df_1m.group_by("_date").agg(
        pl.col("_minute_return_sq").sum().alias("realized_vol_1d"),
        pl.col("minute_return").abs().gt(jump_threshold).sum().alias("jump_count"),
        pl.col("_intraday_drawdown").min().alias("max_intraday_drawdown"),
    )

    # Join daily + intraday on date (avoids datetime tz mismatches)
    df_feat = df_feat.with_columns(pl.col("open_time").dt.date().alias("_date"))
    df_feat = df_feat.join(df_intraday, on="_date", how="left").drop("_date").sort("open_time")

    out_path = get_features_parquet_path(symbol, base_interval, settings=settings)
    save_parquet(df_feat, out_path)
    return {"symbol": symbol, "rows": int(len(df_feat)), "path": str(out_path)}

