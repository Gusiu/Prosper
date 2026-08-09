"""Binance ships more than OHLCV per kline; the pipeline used to discard it.

`extract_csv_from_zip` parsed `num_trades`, `quote_asset_volume` and the
taker-buy columns on every download, then `candles_to_dataframe` pinned a
six-column schema that dropped them. The aggregation step even had code to sum
them, which could never run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from prosper.config import Settings
from prosper.features.build import build_features
from prosper.pipeline.aggregate import aggregate_klines
from prosper.storage.layout import get_features_parquet_path, get_parquet_file_path
from prosper.storage.parquet import (
    KLINES_FULL_SCHEMA,
    KLINES_MINIMAL_SCHEMA,
    candles_to_dataframe,
    save_parquet,
)

SYMBOL = "TESTUSDT"
EXTRA = ("close_time", "quote_asset_volume", "num_trades", "taker_buy_base_volume",
         "taker_buy_quote_volume")


def _candle(ts_ms: int, *, full: bool) -> dict:
    row = {
        "open_time": ts_ms,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 50.0,
    }
    if full:
        row |= {
            "close_time": ts_ms + 59_999,
            "quote_asset_volume": 5050.0,
            "num_trades": 25,
            "taker_buy_base_volume": 30.0,
            "taker_buy_quote_volume": 3030.0,
        }
    return row


def test_full_kline_columns_survive_conversion() -> None:
    df = candles_to_dataframe([_candle(1704067200000, full=True)])

    assert set(EXTRA) <= set(df.columns)
    assert df["num_trades"][0] == 25
    assert df["taker_buy_base_volume"][0] == 30.0
    # Epoch milliseconds become timestamps, same as open_time.
    assert df["close_time"].dtype == KLINES_FULL_SCHEMA["close_time"]


def test_ohlcv_only_sources_still_work() -> None:
    df = candles_to_dataframe([_candle(1704067200000, full=False)])
    assert list(df.columns) == list(KLINES_MINIMAL_SCHEMA)


def test_empty_input_keeps_the_minimal_schema() -> None:
    assert list(candles_to_dataframe([]).columns) == list(KLINES_MINIMAL_SCHEMA)


def test_aggregation_sums_the_order_flow_columns() -> None:
    """The summing code existed but could never run — the columns never arrived."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        _candle(int((start + timedelta(minutes=i)).timestamp() * 1000), full=True)
        for i in range(120)
    ]
    df = candles_to_dataframe(rows)

    hourly = aggregate_klines(df, "1h")

    assert len(hourly) == 2
    assert hourly["num_trades"][0] == 25 * 60
    assert hourly["taker_buy_base_volume"][0] == pytest.approx(30.0 * 60)
    assert hourly["quote_asset_volume"][0] == pytest.approx(5050.0 * 60)


def _write_daily(settings: Settings, *, full: bool, days: int = 90) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(days):
        ts = start + timedelta(days=i)
        row = {
            "open_time": ts,
            "open": 100.0 + i,
            "high": 102.0 + i,
            "low": 99.0 + i,
            "close": 101.0 + i,
            "volume": 50.0 + i,
        }
        if full:
            row |= {
                "quote_asset_volume": (101.0 + i) * (50.0 + i),
                "num_trades": 20 + i,
                "taker_buy_base_volume": (50.0 + i) * 0.6,
                "taker_buy_quote_volume": (101.0 + i) * (50.0 + i) * 0.6,
            }
        rows.append(row)

    df = pl.DataFrame(rows).with_columns(pl.col("open_time").dt.month().alias("_m"))
    for (month,), group in df.group_by(["_m"]):
        save_parquet(
            group.drop("_m").sort("open_time"),
            get_parquet_file_path(SYMBOL, "1d", 2024, month=int(month), settings=settings),
        )


ORDER_FLOW_FEATURES = {
    "taker_buy_ratio",
    "avg_trade_size",
    "flow_imbalance",
    "taker_buy_ratio_14",
    "trade_count_rel_30",
    # `avg_trade_size` is in base units, so it is kept for display and its
    # scale-free counterpart is what a model trains on.
    "avg_trade_size_rel_30",
}


def test_order_flow_features_appear_when_the_columns_exist(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    _write_daily(settings, full=True)

    build_features(SYMBOL, "1d", "2024-01-01", "2024-03-30", settings=settings)
    features = pl.read_parquet(
        get_features_parquet_path(SYMBOL, "1d", settings=settings), hive_partitioning=False
    )

    assert ORDER_FLOW_FEATURES <= set(features.columns)
    # `trade_count` was a float cast of `num_trades` kept as working state for
    # `trade_count_rel_30`; saving it stored the same quantity twice.
    assert "trade_count" not in features.columns
    # 60% of volume lifted the ask in the fixture.
    assert features["taker_buy_ratio"][0] == pytest.approx(0.6)
    assert features["flow_imbalance"][0] == pytest.approx(0.1)


def test_features_still_build_without_the_order_flow_columns(tmp_path) -> None:
    """Klines written before the schema widened must keep working."""
    settings = Settings(data_root=tmp_path)
    _write_daily(settings, full=False)

    result = build_features(SYMBOL, "1d", "2024-01-01", "2024-03-30", settings=settings)

    assert "error" not in result
    features = pl.read_parquet(
        get_features_parquet_path(SYMBOL, "1d", settings=settings), hive_partitioning=False
    )
    assert not (ORDER_FLOW_FEATURES & set(features.columns))
    assert "rsi_14" in features.columns
