"""Time utilities for UTC datetime handling and timestamp normalization."""

from datetime import datetime, timezone
from typing import Any

import polars as pl


def normalize_timestamp_ms(ts: int | float) -> int:
    """
    Normalize timestamp to milliseconds.

    Detects if timestamp is in microseconds (>= 1e12) and converts to milliseconds.
    Otherwise assumes milliseconds.

    Args:
        ts: Timestamp in milliseconds or microseconds

    Returns:
        Timestamp in milliseconds
    """
    if ts >= 1e15:
        # Likely microseconds (e.g. 2021 = ~1.6e15 us), convert to milliseconds
        return int(ts / 1000)
    return int(ts)


def timestamp_to_utc_datetime(ts_ms: int) -> datetime:
    """
    Convert milliseconds timestamp to UTC datetime.

    Args:
        ts_ms: Timestamp in milliseconds

    Returns:
        UTC datetime object
    """
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def datetime_to_timestamp_ms(dt: datetime) -> int:
    """
    Convert UTC datetime to milliseconds timestamp.

    Args:
        dt: Datetime object (assumed UTC if timezone-naive)

    Returns:
        Timestamp in milliseconds
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_year_month(year_month: str) -> tuple[int, int]:
    """
    Parse YYYY-MM string to (year, month).

    Args:
        year_month: String in format YYYY-MM

    Returns:
        Tuple of (year, month)

    Raises:
        ValueError: If format is invalid
    """
    try:
        parts = year_month.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid year-month format: {year_month}")
        year = int(parts[0])
        month = int(parts[1])
        if not (1 <= month <= 12):
            raise ValueError(f"Invalid month: {month}")
        return (year, month)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid year-month format: {year_month}") from e


def parse_date(date_str: str) -> datetime:
    """
    Parse YYYY-MM-DD string to UTC datetime.

    Args:
        date_str: String in format YYYY-MM-DD

    Returns:
        UTC datetime object
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def get_week_number(dt: datetime) -> int:
    """
    Get ISO week number for a datetime.

    Args:
        dt: Datetime object (assumed UTC if timezone-naive)

    Returns:
        ISO week number (1-53)
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isocalendar()[1]


def get_year_week(dt: datetime) -> tuple[int, int]:
    """
    Get (year, week) tuple using ISO week numbering.

    Args:
        dt: Datetime object (assumed UTC if timezone-naive)

    Returns:
        Tuple of (ISO year, ISO week number)
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    iso_year, iso_week, _ = dt.isocalendar()
    return (iso_year, iso_week)


def ensure_utc_datetime(dt: datetime | Any) -> datetime:
    """
    Ensure datetime is UTC-aware.

    Args:
        dt: Datetime object or Polars datetime

    Returns:
        UTC-aware datetime
    """
    if isinstance(dt, pl.Datetime):
        # Convert Polars datetime to Python datetime
        dt = dt.to_python()
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise TypeError(f"Expected datetime, got {type(dt)}")
