"""Tests for the structured API: command builders and data endpoints."""

import json
from datetime import UTC, datetime

import polars as pl
import pytest
from prosper.api.server import (
    AggregateRequest,
    PipelineRequest,
    RepairRequest,
    TrainRequest,
    build_aggregate_command,
    build_pipeline_command,
    build_repair_commands,
    build_train_command,
    delete_symbol_data,
    get_kline_chart_data,
    get_symbol_metadata,
)
from prosper.config import Settings
from prosper.inventory import DataManager
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import save_parquet


def _argv_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_build_pipeline_command_covers_every_selected_interval() -> None:
    commands = build_pipeline_command(
        PipelineRequest(
            symbol="btcusdt", start="2024-01-01", end="2024-03-31", intervals=["1h", "1d"]
        )
    )

    verbs = [cmd[3] for cmd in commands]
    assert verbs[0] == "backfill"
    assert verbs[1] == "aggregate"
    # features + labels for 1m and each requested interval
    assert verbs.count("features") == 3
    assert verbs.count("labels") == 3
    assert all(_argv_value(cmd, "--symbol") == "BTCUSDT" for cmd in commands)


def test_build_train_command_passes_interval_and_research_flags() -> None:
    argv = build_train_command(
        TrainRequest(
            symbol="BTCUSDT",
            model_type="gru",
            interval="1h",
            start="2024-01-01",
            end="2024-06-01",
            epochs=7,
            flags={"strict": True, "deterministic": True, "seed": 7},
        )
    )[0]

    assert argv[:5] == ["python", "-m", "prosper.cli", "predict", "gru"]
    assert _argv_value(argv, "--interval") == "1h"
    assert _argv_value(argv, "--epochs") == "7"
    assert _argv_value(argv, "--seed") == "7"
    assert "--deterministic" in argv and "--strict" in argv


def test_command_builders_reject_a_bad_symbol() -> None:
    with pytest.raises(ValueError, match="Invalid symbol"):
        build_train_command(
            TrainRequest(symbol="not a symbol!", start="2024-01-01", end="2024-06-01")
        )
    with pytest.raises(ValueError, match="Invalid symbol"):
        build_aggregate_command(
            AggregateRequest(
                symbol="../etc", start="2024-01-01", end="2024-03-31", intervals=["1d"]
            )
        )


def test_build_aggregate_command_requires_an_interval() -> None:
    with pytest.raises(ValueError, match="at least one interval"):
        build_aggregate_command(
            AggregateRequest(symbol="BTCUSDT", start="2024-01-01", end="2024-03-31")
        )


def test_repair_redownloads_when_the_1m_base_has_gaps() -> None:
    inventory = [
        {
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "aggregations": [
                {"interval": "1m", "status": "incomplete", "gaps": 3},
                {"interval": "1d", "status": "complete", "gaps": 0},
            ],
        }
    ]

    commands, steps = build_repair_commands(RepairRequest(symbol="BTCUSDT"), inventory)

    assert commands[0][3] == "backfill"
    assert any("re-download 1m" in step for step in steps)


def test_repair_reaggregates_and_rebuilds_only_the_broken_intervals() -> None:
    inventory = [
        {
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "aggregations": [
                {"interval": "1m", "status": "complete", "gaps": 0},
                {"interval": "1h", "status": "incomplete", "gaps": 2},
                {"interval": "1d", "status": "warning", "gaps": 0},
                {"interval": "1w", "status": "complete", "gaps": 0},
            ],
        }
    ]

    commands, steps = build_repair_commands(RepairRequest(symbol="BTCUSDT"), inventory)

    assert all(cmd[3] != "backfill" for cmd in commands), "1m is intact, do not re-download"
    aggregate = next(cmd for cmd in commands if cmd[3] == "aggregate")
    assert _argv_value(aggregate, "--to") == "1h"

    rebuilt = {
        _argv_value(cmd, "--base-interval") for cmd in commands if cmd[3] == "features"
    }
    # The gapped interval and the one missing features; not the healthy 1w.
    assert rebuilt == {"1h", "1d"}
    assert len(steps) == 2


def test_repair_returns_nothing_for_a_healthy_symbol() -> None:
    inventory = [
        {
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "aggregations": [{"interval": "1m", "status": "complete", "gaps": 0}],
        }
    ]

    commands, steps = build_repair_commands(RepairRequest(symbol="BTCUSDT"), inventory)

    assert commands == []
    assert steps == []


def test_repair_rejects_an_unknown_symbol() -> None:
    with pytest.raises(ValueError, match="not present in the data inventory"):
        build_repair_commands(RepairRequest(symbol="NOPEUSDT"), [])


def test_delete_symbol_data_removes_feature_and_label_partitions(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.routes.data.DataManager", lambda: DataManager(settings))

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
    monkeypatch.setattr("prosper.api.routes.data.DataManager", lambda: DataManager(settings))

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
    monkeypatch.setattr("prosper.api.routes.meta.get_settings", lambda: settings)
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
