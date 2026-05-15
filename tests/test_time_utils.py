"""Tests for time utilities."""

from datetime import UTC

import pytest
from prosper.utils.time import (
    normalize_timestamp_ms,
    parse_date,
    parse_year_month,
    timestamp_to_utc_datetime,
)


def test_normalize_timestamp_ms_milliseconds() -> None:
    """Test normalization of millisecond timestamp."""
    ts_ms = 1609459200000  # 2021-01-01 00:00:00 UTC in milliseconds
    assert normalize_timestamp_ms(ts_ms) == ts_ms


def test_normalize_timestamp_ms_microseconds() -> None:
    """Test normalization of microsecond timestamp."""
    ts_us = 1609459200000000  # 2021-01-01 00:00:00 UTC in microseconds
    ts_ms_expected = 1609459200000
    assert normalize_timestamp_ms(ts_us) == ts_ms_expected


def test_timestamp_to_utc_datetime() -> None:
    """Test timestamp to UTC datetime conversion."""
    ts_ms = 1609459200000  # 2021-01-01 00:00:00 UTC
    dt = timestamp_to_utc_datetime(ts_ms)
    assert dt.year == 2021
    assert dt.month == 1
    assert dt.day == 1
    assert dt.tzinfo == UTC



def test_parse_year_month() -> None:
    """Test parsing YYYY-MM string."""
    year, month = parse_year_month("2024-03")
    assert year == 2024
    assert month == 3


def test_parse_year_month_invalid() -> None:
    """Test parsing invalid YYYY-MM string."""
    with pytest.raises(ValueError):
        parse_year_month("2024/03")

    with pytest.raises(ValueError):
        parse_year_month("invalid")


def test_parse_date() -> None:
    """Test parsing YYYY-MM-DD string."""
    dt = parse_date("2024-03-15")
    assert dt.year == 2024
    assert dt.month == 3
    assert dt.day == 15
    assert dt.tzinfo == UTC
