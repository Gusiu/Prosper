"""Tests for QA checks."""

from datetime import UTC, datetime, timedelta

import polars as pl
from prosper.qa.checks import (
    check_duplicates,
    check_gaps,
    check_ohlc_invariants,
    check_sorted,
)


def create_sample_data() -> pl.DataFrame:
    """Create sample klines data."""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    timestamps = [
        int((base_time + timedelta(minutes=i)).timestamp() * 1000) for i in range(10)
    ]

    data = {
        "open_time": timestamps,
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.5] * 10,
        "volume": [1000.0] * 10,
    }

    df = pl.DataFrame(data)
    df = df.with_columns(
        pl.from_epoch(pl.col("open_time"), time_unit="ms").dt.replace_time_zone("UTC").alias("open_time")
    )
    return df


def test_check_sorted() -> None:
    """Test sorted check."""
    df = create_sample_data()
    is_sorted, msg = check_sorted(df)
    assert is_sorted is True

    # Reverse order
    df_reversed = df.sort("open_time", descending=True)
    is_sorted, msg = check_sorted(df_reversed)
    assert is_sorted is False


def test_check_duplicates() -> None:
    """Test duplicate check."""
    df = create_sample_data()
    has_no_duplicates, dup_count, dup_times = check_duplicates(df)
    assert has_no_duplicates is True
    assert dup_count == 0

    # Add duplicate
    df_dup = pl.concat([df, df.head(1)])
    has_no_duplicates, dup_count, dup_times = check_duplicates(df_dup)
    assert has_no_duplicates is False
    assert dup_count > 0


def test_check_gaps() -> None:
    """Test gap detection."""
    df = create_sample_data()
    has_no_gaps, gap_count, gaps = check_gaps(df, expected_interval_seconds=60)
    assert has_no_gaps is True
    assert gap_count == 0

    # Create gap: drop row 5 (minute 5) so we have 0-4, 6-9 with 120s gap between 4 and 6
    df_gap = pl.concat([df.head(5), df.tail(4)])
    has_no_gaps, gap_count, gaps = check_gaps(
        df_gap, expected_interval_seconds=60, tolerance_seconds=0
    )
    assert has_no_gaps is False
    assert gap_count > 0


def test_check_ohlc_invariants() -> None:
    """Test OHLC invariant checks."""
    df = create_sample_data()
    all_valid, violation_count, violations = check_ohlc_invariants(df)
    assert all_valid is True
    assert violation_count == 0

    # Create invalid data (high < open)
    df_invalid = df.with_columns(pl.col("high").mul(0.5))
    all_valid, violation_count, violations = check_ohlc_invariants(df_invalid)
    assert all_valid is False
    assert violation_count > 0
