"""QA checks for data quality validation."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_qa_report_path
from prosper.storage.parquet import load_parquet
from prosper.utils.time import timestamp_to_utc_datetime


def check_sorted(df: pl.DataFrame, time_col: str = "open_time") -> tuple[bool, str]:
    """
    Check if DataFrame is sorted by time column.

    Args:
        df: DataFrame to check
        time_col: Name of time column

    Returns:
        Tuple of (is_sorted, message)
    """
    if df.is_empty():
        return (True, "Empty DataFrame")

    if time_col not in df.columns:
        return (False, f"Time column '{time_col}' not found")

    sorted_df = df.sort(time_col)
    is_sorted = df.equals(sorted_df)

    if is_sorted:
        return (True, "Data is sorted")
    else:
        return (False, "Data is not sorted")


def check_duplicates(df: pl.DataFrame, time_col: str = "open_time") -> tuple[bool, int, list[Any]]:
    """
    Check for duplicate timestamps.

    Args:
        df: DataFrame to check
        time_col: Name of time column

    Returns:
        Tuple of (has_no_duplicates, duplicate_count, duplicate_times)
    """
    if df.is_empty():
        return (True, 0, [])

    if time_col not in df.columns:
        return (False, 0, [])

    duplicates = df.filter(pl.col(time_col).is_duplicated())
    duplicate_count = len(duplicates)

    if duplicate_count == 0:
        return (True, 0, [])
    else:
        # Convert to string to avoid timezone serialization issues
        duplicate_times = (
            duplicates[time_col].unique().dt.strftime("%Y-%m-%dT%H:%M:%S%.3fZ").to_list()
        )
        return (False, duplicate_count, duplicate_times)


def check_gaps(
    df: pl.DataFrame,
    time_col: str = "open_time",
    expected_interval_seconds: int = 60,
    tolerance_seconds: int = 120,
) -> tuple[bool, int, list[dict[str, Any]]]:
    """
    Check for gaps in time series.

    Args:
        df: DataFrame to check
        time_col: Name of time column
        expected_interval_seconds: Expected interval between records in seconds
        tolerance_seconds: Tolerance for gap detection in seconds

    Returns:
        Tuple of (has_no_gaps, gap_count, list_of_gaps)
    """
    if df.is_empty() or len(df) < 2:
        return (True, 0, [])

    if time_col not in df.columns:
        return (False, 0, [])

    # Sort by time
    df_sorted = df.sort(time_col)

    # Calculate time differences
    time_diffs = df_sorted[time_col].diff().dt.total_seconds()

    # Find gaps larger than expected + tolerance
    threshold = expected_interval_seconds + tolerance_seconds
    gap_mask = time_diffs > threshold

    if not gap_mask.any():
        return (True, 0, [])

    # Get gap details using Polars only (avoid datetime->Python conversion issues)
    df_with_gaps = df_sorted.with_columns(
        time_diffs.alias("diff"),
        pl.col(time_col).shift(1).dt.strftime("%Y-%m-%dT%H:%M:%S").alias("gap_start_str"),
        pl.col(time_col).dt.strftime("%Y-%m-%dT%H:%M:%S").alias("gap_end_str"),
    )
    gap_rows = df_with_gaps.filter(pl.col("diff") > threshold).select(
        ["gap_start_str", "gap_end_str", "diff"]
    )

    gaps: list[dict[str, Any]] = []
    for i in range(len(gap_rows)):
        row = gap_rows.row(i, named=True)
        gaps.append(
            {
                "gap_start": row["gap_start_str"],
                "gap_end": row["gap_end_str"],
                "gap_seconds": float(row["diff"]),
                "expected_interval": expected_interval_seconds,
            }
        )

    return (False, len(gaps), gaps)


def check_ohlc_invariants(df: pl.DataFrame) -> tuple[bool, int, list[dict[str, Any]]]:
    """
    Check OHLC invariants:
    - No negative values
    - high >= max(open, close)
    - low <= min(open, close)

    Args:
        df: DataFrame with OHLC columns

    Returns:
        Tuple of (all_valid, violation_count, list_of_violations)
    """
    if df.is_empty():
        return (True, 0, [])

    violations: list[dict[str, Any]] = []

    # Convert open_time to string early to avoid timezone serialization issues
    df_work = df.with_columns(
        pl.col("open_time").dt.strftime("%Y-%m-%d %H:%M:%S").alias("_open_time_str")
    )

    # Check for negative values - select only needed cols to avoid datetime conversion
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            negative_rows = df_work.filter(pl.col(col) < 0).select([col, "_open_time_str"])
            for i in range(len(negative_rows)):
                row = negative_rows.row(i, named=True)
                violations.append(
                    {
                        "type": "negative_value",
                        "column": col,
                        "value": float(row[col]),
                        "open_time": row.get("_open_time_str"),
                    }
                )

    # Check high >= max(open, close) - select only needed cols to avoid datetime conversion
    if all(col in df.columns for col in ["high", "open", "close"]):
        invalid_high_rows = df_work.filter(
            pl.col("high") < pl.max_horizontal(["open", "close"])
        ).select(["high", "open", "close", "_open_time_str"])
        for i in range(len(invalid_high_rows)):
            row = invalid_high_rows.row(i, named=True)
            violations.append(
                {
                    "type": "high_invariant",
                    "high": float(row["high"]),
                    "open": float(row["open"]),
                    "close": float(row["close"]),
                    "open_time": row.get("_open_time_str"),
                }
            )

    # Check low <= min(open, close)
    if all(col in df.columns for col in ["low", "open", "close"]):
        invalid_low_rows = df_work.filter(
            pl.col("low") > pl.min_horizontal(["open", "close"])
        ).select(["low", "open", "close", "_open_time_str"])
        for i in range(len(invalid_low_rows)):
            row = invalid_low_rows.row(i, named=True)
            violations.append(
                {
                    "type": "low_invariant",
                    "low": float(row["low"]),
                    "open": float(row["open"]),
                    "close": float(row["close"]),
                    "open_time": row.get("_open_time_str"),
                }
            )

    return (len(violations) == 0, len(violations), violations)


def run_qa_checks(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Run all QA checks on a dataset.

    Args:
        symbol: Trading symbol
        interval: Time interval (1m, 1h, 1d)
        year: Year
        month: Month (1-12)
        settings: Settings instance (defaults to global)

    Returns:
        Dictionary with QA results
    """
    if settings is None:
        settings = get_settings()

    from prosper.storage.layout import get_parquet_file_path

    parquet_path = get_parquet_file_path(symbol, interval, year, month, settings=settings)

    if not parquet_path.exists():
        return {
            "symbol": symbol,
            "interval": interval,
            "year": year,
            "month": month,
            "error": "Parquet file not found",
        }

    # Load data
    df = load_parquet(parquet_path)

    if df.is_empty():
        return {
            "symbol": symbol,
            "interval": interval,
            "year": year,
            "month": month,
            "error": "Empty DataFrame",
        }

    # Determine expected interval
    expected_interval_map = {"1m": 60, "1h": 3600, "1d": 86400}
    expected_interval = expected_interval_map.get(interval, 60)

    # Run checks
    is_sorted, sort_msg = check_sorted(df)
    has_no_duplicates, dup_count, dup_times = check_duplicates(df)
    has_no_gaps, gap_count, gaps = check_gaps(df, expected_interval_seconds=expected_interval)
    all_valid_ohlc, violation_count, violations = check_ohlc_invariants(df)

    # Compile results
    results = {
        "symbol": symbol,
        "interval": interval,
        "year": year,
        "month": month,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": len(df),
        "checks": {
            "sorted": {"passed": is_sorted, "message": sort_msg},
            "duplicates": {
                "passed": has_no_duplicates,
                "count": dup_count,
                "duplicate_times": dup_times[:10] if dup_times else [],  # Limit to first 10
            },
            "gaps": {
                "passed": has_no_gaps,
                "count": gap_count,
                "gaps": gaps[:10] if gaps else [],  # Limit to first 10
            },
            "ohlc_invariants": {
                "passed": all_valid_ohlc,
                "violation_count": violation_count,
                "violations": violations[:10] if violations else [],  # Limit to first 10
            },
        },
        "overall_passed": is_sorted and has_no_duplicates and has_no_gaps and all_valid_ohlc,
    }

    # Save report
    report_path = get_qa_report_path(symbol, interval, year, month, settings=settings)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)

    return results
