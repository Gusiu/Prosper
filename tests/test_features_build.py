"""Tests for feature engineering."""

from datetime import UTC, datetime, timedelta

import polars as pl
from prosper.config import Settings
from prosper.features.build import build_features
from prosper.storage.layout import get_features_parquet_path, get_parquet_file_path
from prosper.storage.parquet import load_parquet, save_parquet


def _daily_rows(start: datetime, days: int) -> pl.DataFrame:
    rows = []
    for i in range(days):
        ts = start + timedelta(days=i)
        close = 100.0 + i
        rows.append(
            {
                "open_time": ts,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0 + i,
            }
        )
    return pl.DataFrame(rows)


def _minute_rows(start: datetime, days: int) -> pl.DataFrame:
    rows = []
    for i in range(days):
        ts = start + timedelta(days=i)
        close = 100.0 + i
        rows.append(
            {
                "open_time": ts,
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close + 0.1,
                "volume": 10.0,
            }
        )
    return pl.DataFrame(rows)


def test_build_features_uses_daily_warmup_before_requested_start(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    symbol = "TESTUSDT"
    start = datetime(2024, 1, 1, tzinfo=UTC)

    save_parquet(
        _daily_rows(start, 12),
        get_parquet_file_path(symbol, "1d", 2024, month=1, settings=settings),
    )
    save_parquet(
        _minute_rows(start + timedelta(days=9), 3),
        get_parquet_file_path(symbol, "1m", 2024, month=1, settings=settings),
    )

    result = build_features(
        symbol,
        "1d",
        "2024-01-10",
        "2024-01-12",
        settings=settings,
    )

    assert result["rows"] == 3
    features = load_parquet(get_features_parquet_path(symbol, "1d", settings=settings))
    assert len(features) == 3
    assert features["open_time"].dt.strftime("%Y-%m-%d").to_list() == [
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
    ]
    assert features["ma_10"][0] is not None
