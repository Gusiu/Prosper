"""Parquet storage utilities using Polars."""

from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from prosper.config import Settings, get_settings
from prosper.utils.time import timestamp_to_utc_datetime

console = Console()

# Schema for 1m klines
KLINES_SCHEMA = {
    "open_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "quote_asset_volume": pl.Float64,
    "num_trades": pl.Int64,
    "taker_buy_base_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64,
}

# Minimal schema (required fields only)
KLINES_MINIMAL_SCHEMA = {
    "open_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def candles_to_dataframe(
    candles: list[dict[str, float | int]], schema: dict[str, pl.DataType] | None = None
) -> pl.DataFrame:
    """
    Convert list of candle dictionaries to Polars DataFrame.

    Args:
        candles: List of candle dictionaries
        schema: Optional schema dictionary (defaults to minimal schema)

    Returns:
        Polars DataFrame
    """
    if not candles:
        # Return empty DataFrame with schema
        if schema is None:
            schema = KLINES_MINIMAL_SCHEMA
        return pl.DataFrame({}, schema=schema)

    if schema is None:
        schema = KLINES_MINIMAL_SCHEMA

    # Convert to DataFrame
    df = pl.DataFrame(candles, schema=schema, strict=False)

    # Ensure open_time is datetime
    if "open_time" in df.columns:
        if df["open_time"].dtype == pl.Int64:
            df = df.with_columns(
                pl.from_epoch(pl.col("open_time"), time_unit="ms")
                .dt.replace_time_zone("UTC")
                .alias("open_time")
            )

    # Sort by open_time
    if "open_time" in df.columns:
        df = df.sort("open_time")

    # Remove duplicates on open_time
    df = df.unique(subset=["open_time"], keep="first")

    return df


def save_parquet(
    df: pl.DataFrame,
    output_path: Path,
    partition_by: list[str] | None = None,
    settings: Settings | None = None,
) -> None:
    """
    Save DataFrame to Parquet file.

    Args:
        df: Polars DataFrame
        output_path: Path to output Parquet file
        partition_by: Optional list of columns to partition by
        settings: Settings instance (defaults to global)
    """
    if settings is None:
        settings = get_settings()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if partition_by:
        # Partitioned write
        df.write_parquet(
            output_path.parent,
            partition_by=partition_by,
            compression="snappy",
        )
    else:
        # Single file write
        df.write_parquet(
            output_path,
            compression="snappy",
        )


def load_parquet(
    path: Path,
    filters: list[tuple[str, str, Any]] | None = None,
    schema: dict[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    """
    Load Parquet file(s) into DataFrame.

    Args:
        path: Path to Parquet file or directory
        filters: Optional list of (column, operator, value) tuples for filtering
        schema: Optional schema to enforce

    Returns:
        Polars DataFrame
    """
    if path.is_file():
        df = pl.read_parquet(path)
    elif path.is_dir():
        df = pl.read_parquet(path)
    else:
        raise FileNotFoundError(f"Parquet path not found: {path}")

    # Apply filters if provided
    if filters:
        for col, op, value in filters:
            if op == "==":
                df = df.filter(pl.col(col) == value)
            elif op == ">=":
                df = df.filter(pl.col(col) >= value)
            elif op == "<=":
                df = df.filter(pl.col(col) <= value)
            elif op == ">":
                df = df.filter(pl.col(col) > value)
            elif op == "<":
                df = df.filter(pl.col(col) < value)

    # Enforce schema if provided
    if schema:
        # Cast columns to match schema
        for col, dtype in schema.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(dtype))

    # Sort by open_time if present
    if "open_time" in df.columns:
        df = df.sort("open_time")

    return df


def merge_parquet_files(
    output_path: Path,
    input_paths: list[Path],
    deduplicate: bool = True,
    sort_by: str = "open_time",
) -> pl.DataFrame:
    """
    Merge multiple Parquet files into one.

    Args:
        output_path: Path to output Parquet file
        input_paths: List of input Parquet file paths
        deduplicate: Whether to remove duplicates
        sort_by: Column to sort by

    Returns:
        Merged DataFrame
    """
    if not input_paths:
        raise ValueError("No input paths provided")

    dataframes: list[pl.DataFrame] = []
    for path in input_paths:
        if path.exists():
            df = load_parquet(path)
            dataframes.append(df)

    if not dataframes:
        raise ValueError("No valid Parquet files found")

    # Concatenate
    merged = pl.concat(dataframes)

    # Deduplicate
    if deduplicate and sort_by in merged.columns:
        merged = merged.unique(subset=[sort_by], keep="first")
        merged = merged.sort(sort_by)

    # Save
    save_parquet(merged, output_path)

    return merged


def check_parquet_exists(
    symbol: str,
    interval: str,
    year: int,
    month: int | None = None,
    week: int | None = None,
    settings: Settings | None = None,
) -> bool:
    """
    Check if Parquet file exists for given parameters.

    Args:
        symbol: Trading symbol
        interval: Time interval (1m, 1h, 1d, 1w)
        year: Year
        month: Month (1-12) - required for 1m, 1h, 1d
        week: Week number - required for 1w
        settings: Settings instance (defaults to global)

    Returns:
        True if file exists, False otherwise
    """
    from prosper.storage.layout import get_parquet_file_path

    try:
        parquet_path = get_parquet_file_path(symbol, interval, year, month, week, settings)
        return parquet_path.exists()
    except ValueError:
        return False
