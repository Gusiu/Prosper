"""Tests for aggregation."""

from datetime import UTC, datetime, timedelta

import polars as pl
from prosper.config import Settings
from prosper.pipeline.aggregate import aggregate, aggregate_klines, normalize_target_intervals
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import load_parquet, save_parquet


def create_sample_1m_data() -> pl.DataFrame:
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
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
    df_1m = create_sample_1m_data()
    df_1h = aggregate_klines(df_1m, "1h")

    assert not df_1h.is_empty()
    assert len(df_1h) == 2
    assert "open_time" in df_1h.columns
    assert "open" in df_1h.columns
    assert "high" in df_1h.columns
    assert "low" in df_1h.columns
    assert "close" in df_1h.columns
    assert "volume" in df_1h.columns

    assert (df_1h["high"] >= df_1h[["open", "close"]].max_horizontal()).all()
    assert (df_1h["low"] <= df_1h[["open", "close"]].min_horizontal()).all()


def test_aggregate_1m_to_1d() -> None:
    df_1m = create_sample_1m_data()
    df_1d = aggregate_klines(df_1m, "1d")

    assert not df_1d.is_empty()
    assert len(df_1d) == 1
    assert "open_time" in df_1d.columns
    assert "open" in df_1d.columns
    assert "high" in df_1d.columns
    assert "low" in df_1d.columns
    assert "close" in df_1d.columns
    assert "volume" in df_1d.columns


def test_aggregate_1m_to_1w() -> None:
    df_1m = create_sample_1m_data()
    df_1w = aggregate_klines(df_1m, "1w")

    assert not df_1w.is_empty()
    assert "open_time" in df_1w.columns
    assert "open" in df_1w.columns
    assert "high" in df_1w.columns
    assert "low" in df_1w.columns
    assert "close" in df_1w.columns
    assert "volume" in df_1w.columns


def test_aggregate_empty_dataframe() -> None:
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

    assert aggregate_klines(empty_df, "1h").is_empty()
    assert aggregate_klines(empty_df, "1d").is_empty()
    assert aggregate_klines(empty_df, "1w").is_empty()


def test_aggregate_pipeline_writes_requested_intervals(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    symbol = "TESTUSDT"
    source = create_sample_1m_data()
    save_parquet(source, get_parquet_file_path(symbol, "1m", 2024, month=1, settings=settings))

    result = aggregate(symbol, "1m", ["1h", "1d"], "2024-01", "2024-01", settings=settings)

    assert result["errors"] == []
    assert get_parquet_file_path(symbol, "1h", 2024, month=1, settings=settings).exists()
    assert get_parquet_file_path(symbol, "1d", 2024, month=1, settings=settings).exists()
    assert len(load_parquet(get_parquet_file_path(symbol, "1h", 2024, month=1, settings=settings))) == 2


def test_normalize_target_intervals_accepts_cli_shapes() -> None:
    assert normalize_target_intervals("1h,1d") == ["1h", "1d"]
    assert normalize_target_intervals(["1h", "1d", "1h"]) == ["1h", "1d"]
    assert normalize_target_intervals(["1h,1d", "1w"]) == ["1h", "1d", "1w"]
