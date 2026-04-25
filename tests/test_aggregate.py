"""Tests for aggregation functions."""

from datetime import datetime, timedelta, timezone

import polars as pl

from prosper.pipeline.aggregate import aggregate_1m_to_1d, aggregate_1m_to_1h, aggregate_1m_to_1w


def create_sample_1m_data() -> pl.DataFrame:
    """Create sample 1m klines data."""
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [int((base_time + timedelta(minutes=i)).timestamp() * 1000) for i in range(120)]

    data = {
        "open_time": timestamps,
        "open": [100.0 + i * 0.1 for i in range(120)],
        "high": [101.0 + i * 0.1 for i in range(120)],
        "low": [99.0 + i * 0.1 for i in range(120)],
        "close": [100.5 + i * 0.1 for i in range(120)],
        "volume": [1000.0] * 120,
    }

    df = pl.DataFrame(data)
    df = df.with_columns(
        pl.from_epoch(pl.col("open_time"), time_unit="ms").dt.replace_time_zone("UTC").alias("open_time")
    )
    return df


def test_aggregate_1m_to_1h() -> None:
    """Test aggregation from 1m to 1h."""
    df_1m = create_sample_1m_data()
    df_1h = aggregate_1m_to_1h(df_1m)

    assert not df_1h.is_empty()
    assert len(df_1h) == 2  # 120 minutes = 2 hours
    assert "open_time" in df_1h.columns
    assert "open" in df_1h.columns
    assert "high" in df_1h.columns
    assert "low" in df_1h.columns
    assert "close" in df_1h.columns
    assert "volume" in df_1h.columns

    # Check OHLC invariants
    assert (df_1h["high"] >= df_1h[["open", "close"]].max_horizontal()).all()
    assert (df_1h["low"] <= df_1h[["open", "close"]].min_horizontal()).all()


def test_aggregate_1m_to_1d() -> None:
    """Test aggregation from 1m to 1d."""
    df_1m = create_sample_1m_data()
    df_1d = aggregate_1m_to_1d(df_1m)

    assert not df_1d.is_empty()
    assert len(df_1d) == 1  # All data in one day
    assert "open_time" in df_1d.columns
    assert "open" in df_1d.columns
    assert "high" in df_1d.columns
    assert "low" in df_1d.columns
    assert "close" in df_1d.columns
    assert "volume" in df_1d.columns


def test_aggregate_1m_to_1w() -> None:
    """Test aggregation from 1m to 1w."""
    df_1m = create_sample_1m_data()
    df_1w = aggregate_1m_to_1w(df_1m)

    assert not df_1w.is_empty()
    assert "open_time" in df_1w.columns
    assert "open" in df_1w.columns
    assert "high" in df_1w.columns
    assert "low" in df_1w.columns
    assert "close" in df_1w.columns
    assert "volume" in df_1w.columns


def test_aggregate_empty_dataframe() -> None:
    """Test aggregation with empty DataFrame."""
    empty_df = pl.DataFrame(
        {
            "open_time": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
    )

    assert aggregate_1m_to_1h(empty_df).is_empty()
    assert aggregate_1m_to_1d(empty_df).is_empty()
    assert aggregate_1m_to_1w(empty_df).is_empty()
