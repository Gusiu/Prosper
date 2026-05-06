"""Aggregation pipeline for resampling klines."""

from collections.abc import Iterable
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import load_parquet, save_parquet

SUPPORTED_TARGET_INTERVALS = {
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
}


def normalize_target_intervals(intervals: str | Iterable[str]) -> list[str]:
    """Normalize comma/space separated interval inputs into a deduplicated list."""
    raw_items = [intervals] if isinstance(intervals, str) else list(intervals)
    normalized: list[str] = []
    for item in raw_items:
        for part in str(item).replace(",", " ").split():
            interval = part.strip()
            if interval:
                normalized.append(interval)

    seen: set[str] = set()
    return [interval for interval in normalized if not (interval in seen or seen.add(interval))]


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

    if df["open_time"].dtype != pl.Datetime:
        df = df.with_columns(
            pl.from_epoch(pl.col("open_time"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("open_time")
        )

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
    to_intervals: str | Iterable[str],
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

    from prosper.utils.time import generate_month_range

    to_intervals = normalize_target_intervals(to_intervals)
    months = generate_month_range(start, end)
    results: dict[str, Any] = {
        "symbol": symbol,
        "from_interval": from_interval,
        "to_intervals": to_intervals,
        "processed_months": [],
        "written_files": {},
        "errors": [],
    }

    unsupported = sorted(set(to_intervals) - SUPPORTED_TARGET_INTERVALS)
    if unsupported:
        results["errors"].append(f"Unsupported target intervals: {', '.join(unsupported)}")
        return results

    source_parts: list[pl.DataFrame] = []
    for year, month in months:
        source_path = get_parquet_file_path(symbol, from_interval, year, month, settings=settings)
        if not source_path.exists():
            results["errors"].append(f"{year}-{month:02d}: Source file not found")
            continue

        try:
            df_source = load_parquet(source_path)
        except Exception as e:
            results["errors"].append(f"{year}-{month:02d}: {e}")
            continue

        if df_source.is_empty():
            results["errors"].append(f"{year}-{month:02d}: Empty source data")
            continue

        source_parts.append(df_source)
        results["processed_months"].append(f"{year}-{month:02d}")

    if not source_parts:
        return results

    df_all = (
        pl.concat(source_parts, how="diagonal_relaxed")
        .sort("open_time")
        .unique(subset=["open_time"], keep="first")
        .sort("open_time")
    )

    for to_interval in to_intervals:
        try:
            df_agg = aggregate_klines(df_all, to_interval)
            results["written_files"][to_interval] = _save_aggregated_partitions(
                df_agg,
                symbol,
                to_interval,
                settings,
            )
        except Exception as e:
            results["errors"].append(f"{to_interval}: {e}")

    return results


def _save_aggregated_partitions(
    df: pl.DataFrame,
    symbol: str,
    interval: str,
    settings: Settings,
) -> list[str]:
    if df.is_empty():
        return []

    written_paths: list[str] = []
    if interval == "1w":
        partitioned = df.with_columns(
            pl.col("open_time").dt.iso_year().alias("_year"),
            pl.col("open_time").dt.week().alias("_week"),
        )
        for (year, week), group_df in partitioned.group_by(["_year", "_week"]):
            output_path = get_parquet_file_path(symbol, interval, int(year), week=int(week), settings=settings)
            save_parquet(group_df.drop(["_year", "_week"]).sort("open_time"), output_path)
            written_paths.append(str(output_path))
        return sorted(written_paths)

    partitioned = df.with_columns(
        pl.col("open_time").dt.year().alias("_year"),
        pl.col("open_time").dt.month().alias("_month"),
    )
    for (year, month), group_df in partitioned.group_by(["_year", "_month"]):
        output_path = get_parquet_file_path(symbol, interval, int(year), month=int(month), settings=settings)
        save_parquet(group_df.drop(["_year", "_month"]).sort("open_time"), output_path)
        written_paths.append(str(output_path))

    return sorted(written_paths)
