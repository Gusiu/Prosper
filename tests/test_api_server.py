"""Tests for local API command safety helpers."""

import json
from datetime import UTC, datetime

import polars as pl
import pytest
from prosper.api.server import (
    delete_symbol_data,
    get_kline_chart_data,
    get_symbol_metadata,
    parse_safe_command_chain,
)
from prosper.config import Settings
from prosper.inventory import DataManager
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import save_parquet


def test_parse_safe_command_chain_accepts_prosper_segments() -> None:
    commands = parse_safe_command_chain(
        "python -m prosper.cli backfill --symbol BTCUSDT --start 2024-01 --end 2024-01"
        " && python -m prosper.cli labels build --symbol BTCUSDT"
    )

    assert len(commands) == 2
    assert commands[0][:4] == ["python", "-m", "prosper.cli", "backfill"]
    assert commands[1][:5] == ["python", "-m", "prosper.cli", "labels", "build"]


def test_parse_safe_command_chain_rejects_shell_injection() -> None:
    with pytest.raises(ValueError):
        parse_safe_command_chain("python -m prosper.cli backfill --symbol BTCUSDT --start 2024-01 --end 2024-01; whoami")

    with pytest.raises(ValueError):
        parse_safe_command_chain("python -m prosper.cli backfill --symbol BTCUSDT && echo hacked")


def test_delete_symbol_data_removes_feature_and_label_partitions(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.server.DataManager", lambda: DataManager(settings))

    features_dir = settings.processed_data_dir / "binance" / "spot" / "features" / "1d" / "symbol=TESTUSDT"
    labels_dir = settings.processed_binance_spot_labels_dir / "1d" / "symbol=TESTUSDT"
    features_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (features_dir / "features.parquet").write_text("placeholder", encoding="utf-8")
    (labels_dir / "labels.parquet").write_text("placeholder", encoding="utf-8")

    result = delete_symbol_data("TESTUSDT")

    assert result["paths_removed"] >= 2
    assert not features_dir.exists()
    assert not labels_dir.exists()


def test_get_kline_chart_data_returns_lightweight_payload(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.server.DataManager", lambda: DataManager(settings))

    symbol = "TESTUSDT"
    save_parquet(
        pl.DataFrame(
            {
                "open_time": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [10.0],
            }
        ),
        get_parquet_file_path(symbol, "1d", 2024, month=1, settings=settings),
    )

    payload = get_kline_chart_data(symbol, "1d", limit=1000)

    assert payload["symbol"] == symbol
    assert payload["interval"] == "1d"
    assert payload["candles"][0]["time"] == 1704067200
    assert payload["candles"][0]["close"] == 100.5


def test_get_symbol_metadata_returns_base_and_quote_assets(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.server.get_settings", lambda: settings)
    symbols_path = settings.meta_dir / "binance_spot_symbols.json"
    symbols_path.write_text(
        json.dumps(
            {
                "symbols": [
                    "BTCUSDT",
                    "ETHUSDT",
                    {"symbol": "SOLFDUSD", "baseAsset": "SOL", "quoteAsset": "FDUSD"},
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = get_symbol_metadata()

    assert payload["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLFDUSD"]
    assert payload["base_assets"] == ["BTC", "ETH", "SOL"]
    assert payload["quote_assets"] == ["FDUSD", "USDT"]
