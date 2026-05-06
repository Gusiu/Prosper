"""Time utilities for UTC datetime handling and timestamp normalization."""

from __future__ import annotations

import datetime as _dt
from datetime import UTC, datetime


def normalize_timestamp_ms(ts: int | float) -> int:
    """Normalize timestamp to milliseconds.

    Detects if timestamp is in microseconds (>= 1e15) and converts to
    milliseconds.  Otherwise assumes milliseconds.
    """
    if ts >= 1e15:
        return int(ts / 1000)
    return int(ts)


def timestamp_to_utc_datetime(ts_ms: int) -> datetime:
    """Convert milliseconds timestamp to UTC datetime."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)


def parse_year_month(year_month: str) -> tuple[int, int]:
    """Parse ``YYYY-MM`` string to ``(year, month)``."""
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
    """Parse ``YYYY-MM-DD`` string to UTC datetime."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(tzinfo=UTC)


def get_year_week(dt: datetime) -> tuple[int, int]:
    """Get ``(ISO year, ISO week)`` tuple for a datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    iso_year, iso_week, _ = dt.isocalendar()
    return (iso_year, iso_week)


# ── Month iteration ─────────────────────────────────────────────────────────

def generate_month_range(
    start: str | _dt.date,
    end: str | _dt.date,
) -> list[tuple[int, int]]:
    """Generate ``[(year, month), ...]`` from *start* to *end* inclusive.

    Accepts either ``YYYY-MM`` strings **or** ``date`` / ``datetime`` objects.
    """
    if isinstance(start, str):
        sy, sm = parse_year_month(start)
    else:
        sy, sm = start.year, start.month

    if isinstance(end, str):
        ey, em = parse_year_month(end)
    else:
        ey, em = end.year, end.month

    months: list[tuple[int, int]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months

