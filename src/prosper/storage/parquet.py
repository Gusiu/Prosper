"""Parquet storage utilities using Polars."""

from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from prosper.config import Settings

console = Console()

# Minimal schema (required fields only)
KLINES_MINIMAL_SCHEMA = {
    "open_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

# Everything Binance ships in a kline row. `extract_csv_from_zip` already parses
# these; the minimal schema above silently dropped them, so the order-flow
# columns were paid for on every download and then thrown away.
KLINES_FULL_SCHEMA = {
    **KLINES_MINIMAL_SCHEMA,
    "close_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "quote_asset_volume": pl.Float64,
    "num_trades": pl.Int64,
    "taker_buy_base_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64,
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
    if schema is None:
        # Keep whatever the source actually provided. Pinning the minimal schema
        # here is what discarded num_trades and the taker-buy columns.
        available = set(candles[0]) if candles else set()
        schema = {
            name: dtype
            for name, dtype in KLINES_FULL_SCHEMA.items()
            if name in available or name in KLINES_MINIMAL_SCHEMA
        }

    if not candles:
        return pl.DataFrame({}, schema=schema)

    # Convert to DataFrame
    df = pl.DataFrame(candles, schema=schema, strict=False)

    # Ensure epoch-millisecond timestamps become datetimes
    for column in ("open_time", "close_time"):
        if column in df.columns and df[column].dtype == pl.Int64:
            df = df.with_columns(
                pl.from_epoch(pl.col(column), time_unit="ms")
                .dt.replace_time_zone("UTC")
                .alias(column)
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
) -> None:
    """
    Save DataFrame to Parquet file.

    Args:
        df: Polars DataFrame
        output_path: Path to output Parquet file
        partition_by: Optional list of columns to partition by
    """
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
        df = pl.read_parquet(path, hive_partitioning=False)
    elif path.is_dir():
        df = pl.read_parquet(path, hive_partitioning=False)
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
