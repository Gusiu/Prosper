"""Aggregation pipeline for resampling klines to different intervals."""

from datetime import timedelta, timezone
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import load_parquet, save_parquet
from prosper.utils.time import get_year_week, timestamp_to_utc_datetime


def aggregate_klines(df: pl.DataFrame, interval: str) -> pl.DataFrame:
    """
    Aggregate klines to a target interval (e.g. '15m', '4h', '1d', '1w').

    Args:
        df: DataFrame with klines
        interval: Polars compatible interval string

    Returns:
        DataFrame with aggregated klines
    """
    if df.is_empty():
        return df

    # Ensure open_time is datetime
    if df["open_time"].dtype != pl.Datetime:
        df = df.with_columns(
            pl.from_epoch(pl.col("open_time"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("open_time")
        )

    # Convert to Python datetime for safe timezone-naive truncation
    df_python = df.with_columns(
        pl.col("open_time").dt.replace_time_zone(None).alias("open_time_py")
    )

    agg_exprs = [
        pl.first("open").alias("open"),
        pl.max("high").alias("high"),
        pl.min("low").alias("low"),
        pl.last("close").alias("close"),
        pl.sum("volume").alias("volume"),
    ]
    if "close_time" in df.columns:
        agg_exprs.append(pl.last("close_time").alias("close_time"))
    for opt_col in ["quote_asset_volume", "num_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        if opt_col in df.columns:
            agg_exprs.append(pl.sum(opt_col).alias(opt_col))

    df_agg = (
        df_python.with_columns(
            pl.col("open_time_py").dt.truncate(interval).alias("bucket_start"),
        )
        .group_by("bucket_start")
        .agg(agg_exprs)
        .rename({"bucket_start": "open_time"})
        .with_columns(
            pl.col("open_time").dt.replace_time_zone("UTC").alias("open_time")
        )
        .sort("open_time")
    )

    return df_agg


def aggregate(
    symbol: str,
    from_interval: str,
    to_intervals: list[str],
    start: str,
    end: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Aggregate klines from one interval to others.

    Args:
        symbol: Trading symbol
        from_interval: Source interval (e.g., "1m")
        to_intervals: Target intervals (e.g., ["1h", "1d", "1w"])
        start: Start date in YYYY-MM format
        end: End date in YYYY-MM format
        settings: Settings instance (defaults to global)

    Returns:
        Dictionary with aggregation results
    """
    if settings is None:
        settings = get_settings()

    from prosper.pipeline.backfill import generate_month_range
    from prosper.utils.time import parse_year_month

    months = generate_month_range(start, end)
    results: dict[str, Any] = {
        "symbol": symbol,
        "from_interval": from_interval,
        "to_intervals": to_intervals,
        "processed_months": [],
        "errors": [],
    }

    # Removed static agg_functions dictionary

    for year, month in months:
        try:
            # Load source data
            source_path = get_parquet_file_path(symbol, from_interval, year, month, settings=settings)
            if not source_path.exists():
                results["errors"].append(f"{year}-{month:02d}: Source file not found")
                continue

            df_source = load_parquet(source_path)
            if df_source.is_empty():
                results["errors"].append(f"{year}-{month:02d}: Empty source data")
                continue

            # Aggregate to each target interval
            for to_interval in to_intervals:
                df_agg = aggregate_klines(df_source, to_interval)

                if df_agg.is_empty():
                    continue

                # Determine output path
                if to_interval == "1w":
                    # For weekly, need to determine year/week for each row
                    # Group by year/week and save separately
                    df_agg = df_agg.with_columns(
                        pl.col("open_time").dt.year().alias("year"),
                        pl.col("open_time").dt.week().alias("week"),
                    )

                    for (year_w, week_w), group_df in df_agg.group_by(["year", "week"]):
                        group_df_clean = group_df.drop(["year", "week"])
                        output_path = get_parquet_file_path(
                            symbol, to_interval, int(year_w), week=int(week_w), settings=settings
                        )
                        save_parquet(group_df_clean, output_path)
                else:
                    output_path = get_parquet_file_path(
                        symbol, to_interval, year, month, settings=settings
                    )
                    save_parquet(df_agg, output_path)

            results["processed_months"].append(f"{year}-{month:02d}")

        except Exception as e:
            results["errors"].append(f"{year}-{month:02d}: {str(e)}")

    return results
