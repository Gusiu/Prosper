"""Tests for local API command safety helpers."""

import pytest
from prosper.api.server import delete_symbol_data, parse_safe_command_chain
from prosper.config import Settings
from prosper.core import DataManager


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
