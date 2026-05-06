"""Tests for DataManager inventory behavior."""

from datetime import UTC, datetime

import polars as pl
from prosper.config import Settings
from prosper.core import DataManager
from prosper.storage.layout import get_parquet_file_path
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


def test_inventory_reports_intervals_quality_and_cache_refresh(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    symbol = "TESTUSDT"
    data_manager = DataManager(settings)

    save_parquet(_sample_daily_frame(), get_parquet_file_path(symbol, "1m", 2024, month=1, settings=settings))
    first_inventory = data_manager.get_inventory(refresh=True)

    assert first_inventory[0]["aggregations"] == ["1m"]
    assert first_inventory[0]["quality"]["status"] == "incomplete"

    save_parquet(_sample_daily_frame(), get_parquet_file_path(symbol, "1d", 2024, month=1, settings=settings))
    refreshed_inventory = data_manager.get_inventory(refresh=True)

    assert refreshed_inventory[0]["aggregations"] == ["1d", "1m"]
    assert refreshed_inventory[0]["quality"]["status"] == "healthy"
    assert data_manager.cache_path.exists()
