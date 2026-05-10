from __future__ import annotations

from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_features_parquet_path
from prosper.storage.parquet import load_parquet, save_parquet


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
    Build engineered features parquet for any base interval.
    """
    if settings is None:
        settings = get_settings()

    # Load klines for selected interval
    # We scan the entire symbol directory for the interval to be robust (works for both months and weeks)
    base_path = settings.processed_binance_spot_klines_dir / base_interval / f"symbol={symbol}"
    if not base_path.exists():
        out_path = get_features_parquet_path(symbol, base_interval, settings=settings)
        return {
            "error": f"No data directory found for {symbol} at {base_interval}",
            "path": str(out_path),
        }

    parts: list[pl.DataFrame] = []
    for p in base_path.glob("**/*.parquet"):
        if p.is_file() and p.stat().st_size > 0:
            parts.append(load_parquet(p))

    if not parts:
        out_path = get_features_parquet_path(symbol, base_interval, settings=settings)
        return {
            "error": f"No {base_interval} parquet files found for {symbol}",
            "path": str(out_path),
        }

    df = pl.concat(parts).sort("open_time").unique(subset=["open_time"], keep="first")

    # Timeframe-agnostic features (based on rows/candles)
    log_close = pl.col("close").log()
    df_feat = df.with_columns(
        [
            (log_close - log_close.shift(1)).alias("log_return_1b"),
            (log_close - log_close.shift(3)).alias("log_return_3b"),
            (log_close - log_close.shift(7)).alias("log_return_7b"),
            (log_close - log_close.shift(14)).alias("log_return_14b"),
        ]
    )

    df_feat = df_feat.with_columns(
        [
            pl.col("log_return_1b").rolling_std(window_size=7).alias("rolling_vol_7"),
            pl.col("log_return_1b").rolling_std(window_size=14).alias("rolling_vol_14"),
            pl.col("log_return_1b").rolling_std(window_size=30).alias("rolling_vol_30"),
        ]
    )

    df_feat = df_feat.with_columns(
        [
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("candle_range"),
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
    df_feat = (
        df_feat.with_columns([macd])
        .with_columns([signal])
        .with_columns([(pl.col("_macd") - pl.col("_signal")).alias("macd_hist")])
        .drop(["_macd", "_signal"])
    )

    # Intraday derived features (only when base interval is 1d and 1m data exists)
    intraday_base = settings.processed_binance_spot_klines_dir / "1m" / f"symbol={symbol}"
    if base_interval == "1d" and intraday_base.exists():
        intraday_files = list(intraday_base.glob("**/*.parquet"))
        intraday_parts = [load_parquet(p) for p in intraday_files if p.stat().st_size > 0]

        if intraday_parts:
            df_1m = (
                pl.concat(intraday_parts)
                .sort("open_time")
                .unique(subset=["open_time"], keep="first")
            )
            df_1m = df_1m.with_columns(pl.col("open_time").dt.date().alias("_date"))

            df_1m = df_1m.with_columns(
                [
                    (pl.col("close") / pl.col("close").shift(1).over("_date") - 1.0).alias(
                        "minute_return"
                    ),
                ]
            )
            df_1m = df_1m.with_columns(
                [
                    pl.col("minute_return").pow(2).alias("_minute_return_sq"),
                    (pl.col("close") / pl.col("close").cummax().over("_date") - 1.0).alias(
                        "_intraday_drawdown"
                    ),
                ]
            )

            df_intraday = df_1m.group_by("_date").agg(
                pl.col("_minute_return_sq").sum().alias("realized_vol_1d"),
                pl.col("minute_return").abs().gt(jump_threshold).sum().alias("jump_count"),
                pl.col("_intraday_drawdown").min().alias("max_intraday_drawdown"),
            )

            # Join intraday features to main features
            df_feat = df_feat.with_columns(pl.col("open_time").dt.date().alias("_date"))
            df_feat = (
                df_feat.join(df_intraday, on="_date", how="left").drop("_date").sort("open_time")
            )
        else:
            # No 1m data - just add empty intraday columns for schema consistency
            df_feat = df_feat.with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("realized_vol_1d"),
                    pl.lit(None).cast(pl.Int64).alias("jump_count"),
                    pl.lit(None).cast(pl.Float64).alias("max_intraday_drawdown"),
                ]
            )
    else:
        # Non-daily intervals don't need intraday features
        pass

    cols_to_drop = [c for c in ["symbol", "year", "month", "week"] if c in df_feat.columns]
    if cols_to_drop:
        df_feat = df_feat.drop(cols_to_drop)

    out_path = get_features_parquet_path(symbol, base_interval, settings=settings)
    save_parquet(df_feat, out_path)
    return {"symbol": symbol, "rows": int(len(df_feat)), "path": str(out_path)}
