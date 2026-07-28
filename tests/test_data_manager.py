"""Tests for DataManager inventory behavior."""

from datetime import UTC, datetime, timedelta

import polars as pl
from prosper.config import Settings
from prosper.inventory import DataManager
from prosper.storage.layout import (
    get_features_parquet_path,
    get_labels_parquet_path,
    get_parquet_file_path,
)
from prosper.storage.parquet import save_parquet


def _sample_daily_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        }
    )


def _sample_chart_frame() -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start, start + timedelta(days=1), start + timedelta(days=3)]
    return pl.DataFrame(
        {
            "open_time": dates,
            "open": [100.0, 101.0, 103.0],
            "high": [102.0, 103.0, 105.0],
            "low": [99.0, 100.0, 102.0],
            "close": [101.0, 102.0, 104.0],
            "volume": [10.0, 11.0, 13.0],
        }
    )


def _sample_features_frame() -> pl.DataFrame:
    df = _sample_chart_frame()
    return df.with_columns(
        [
            pl.Series("ma_10", [100.5, 101.5, 103.5]),
            pl.Series("ma_30", [100.1, 101.1, 103.1]),
            pl.Series("rsi_14", [50.0, 55.0, 61.0]),
            pl.Series("macd_hist", [0.1, -0.05, 0.2]),
            pl.Series("atr_14", [1.0, 1.2, 1.4]),
        ]
    )


def _sample_labels_frame() -> pl.DataFrame:
    df = _sample_chart_frame()
    return df.select("open_time", "close").with_columns(
        [
            pl.Series("return_fwd", [0.02, -0.02, None]),
            pl.Series("direction", ["long", "short", "flat"]),
            pl.Series("depth_bin", [0, 1, None]),
        ]
    )


def test_inventory_reports_intervals_quality_and_cache_refresh(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    symbol = "TESTUSDT"
    data_manager = DataManager(settings)

    save_parquet(_sample_daily_frame(), get_parquet_file_path(symbol, "1m", 2024, month=1, settings=settings))
    first_inventory = data_manager.get_inventory(refresh=True)

    assert [agg["interval"] for agg in first_inventory[0]["aggregations"]] == ["1m"]
    assert first_inventory[0]["quality"]["status"] == "warning"

    save_parquet(_sample_daily_frame(), get_parquet_file_path(symbol, "1d", 2024, month=1, settings=settings))
    for interval in ["1m", "1d"]:
        save_parquet(
            _sample_daily_frame(),
            get_features_parquet_path(symbol, interval, settings=settings),
        )
        save_parquet(
            _sample_daily_frame(),
            get_labels_parquet_path(symbol, interval, settings=settings),
        )
    refreshed_inventory = data_manager.get_inventory(refresh=True)

    assert [agg["interval"] for agg in refreshed_inventory[0]["aggregations"]] == ["1m", "1d"]
    assert refreshed_inventory[0]["quality"]["status"] == "complete"
    assert data_manager.cache_path.exists()


def test_chart_data_joins_features_labels_and_detects_gaps(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    symbol = "TESTUSDT"
    data_manager = DataManager(settings)

    save_parquet(_sample_chart_frame(), get_parquet_file_path(symbol, "1d", 2024, month=1, settings=settings))
    save_parquet(_sample_features_frame(), get_features_parquet_path(symbol, "1d", settings=settings))
    save_parquet(_sample_labels_frame(), get_labels_parquet_path(symbol, "1d", settings=settings))

    payload = data_manager.get_chart_data(symbol, "1d", limit=1000)

    assert payload["rows"] == 3
    assert payload["gaps"] == [{"time": 1704240000, "from": 1704240000, "to": 1704240000, "missing": 1}]
    assert "ma_10" in payload["columns"]["features"]
    assert "direction" in payload["columns"]["labels"]
    assert payload["candles"][0]["time"] == 1704067200
    assert payload["candles"][0]["ma_10"] == 100.5
    assert payload["candles"][0]["direction"] == "long"
    assert payload["candles"][1]["depth_bin"] == 1
