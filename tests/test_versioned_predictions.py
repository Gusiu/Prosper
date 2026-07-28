"""Tests for versioned prediction folder helpers and AI model API."""

import json

import pytest
from fastapi import HTTPException
from prosper.api.server import (
    EvalRequest,
    TrainRequest,
    delete_ai_model,
    get_ai_models,
    get_ai_predictions,
    get_feature_importances,
    run_evaluation,
    run_training,
    runner,
)
from prosper.config import Settings
from prosper.storage.layout import (
    parse_versioned_prediction_folder,
    resolve_versioned_prediction_dir,
    versioned_prediction_folder_name,
)


def test_versioned_prediction_folder_name_with_interval() -> None:
    assert versioned_prediction_folder_name("ml", "20240501123000", "1d") == "ml_1d_20240501123000"


def test_parse_versioned_prediction_folder_new_format() -> None:
    assert parse_versioned_prediction_folder("xgboost_1h_20240501123000") == (
        "xgboost",
        "1h",
        "20240501123000",
    )


def test_parse_versioned_prediction_folder_legacy_format() -> None:
    assert parse_versioned_prediction_folder("xgboost_20240501123000") == (
        "xgboost",
        "1d",
        "20240501123000",
    )


def test_parse_versioned_prediction_folder_embedded_model_type() -> None:
    assert parse_versioned_prediction_folder("ml_1d_20240501123000") == (
        "ml",
        "1d",
        "20240501123000",
    )


def test_get_ai_models_parses_interval(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.server.get_settings", lambda: settings)

    run_dir = (
        settings.reports_predictions_dir
        / "BTCUSDT"
        / "ml_1d_20240501123000"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text(
        json.dumps({"date": "2024-05-01", "symbol": "BTCUSDT"}) + "\n",
        encoding="utf-8",
    )

    payload = get_ai_models()

    assert len(payload["models"]) == 1
    assert payload["models"][0]["model_type"] == "ml"
    assert payload["models"][0]["interval"] == "1d"
    assert payload["models"][0]["timestamp"] == "20240501123000"


def test_delete_ai_model_removes_run_folder(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.server.get_settings", lambda: settings)

    run_dir = resolve_versioned_prediction_dir(
        "BTCUSDT", "gru", "20240501123000", settings=settings, interval="1d"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text("{}", encoding="utf-8")

    result = delete_ai_model("BTCUSDT", "gru", "20240501123000", interval="1d")

    assert result["status"] == "deleted"
    assert not run_dir.exists()


def test_delete_ai_model_not_found(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.server.get_settings", lambda: settings)

    with pytest.raises(HTTPException) as exc:
        delete_ai_model("BTCUSDT", "tft", "20990101123000", interval="1d")

    assert exc.value.status_code == 404


def test_get_ai_predictions_versioned_run(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.server.get_settings", lambda: settings)

    run_dir = resolve_versioned_prediction_dir(
        "BTCUSDT", "xgboost", "20240501123000", settings=settings, interval="1h"
    )
    run_dir.mkdir(parents=True)
    row = {"date": "2024-05-01", "symbol": "BTCUSDT", "short": {"P_long": 0.5}}
    (run_dir / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    payload = get_ai_predictions(
        symbol="BTCUSDT",
        model_type="xgboost",
        timestamp="20240501123000",
        interval="1h",
    )

    assert len(payload["predictions"]) == 1
    assert payload["predictions"][0]["date"] == "2024-05-01"


def test_get_feature_importances_returns_json(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.server.get_settings", lambda: settings)

    run_dir = resolve_versioned_prediction_dir(
        "BTCUSDT", "xgboost", "20240501123000", settings=settings, interval="1d"
    )
    run_dir.mkdir(parents=True)
    fi = {"short": {"feat_a": 0.8, "feat_b": 0.2}}
    (run_dir / "feature_importances.json").write_text(json.dumps(fi), encoding="utf-8")

    payload = get_feature_importances(
        symbol="BTCUSDT",
        model_type="xgboost",
        timestamp="20240501123000",
        interval="1d",
    )

    assert payload["short"]["feat_a"] == 0.8


def test_run_training_includes_interval_in_command(monkeypatch) -> None:
    started: list[list[list[str]]] = []

    def fake_start(cmd: str, commands: list[list[str]]) -> None:
        started.append(commands)

    monkeypatch.setattr(runner, "start", fake_start)

    result = run_training(
        TrainRequest(
            symbol="BTCUSDT",
            model_type="ml",
            start="2024-01-01",
            end="2024-06-01",
            interval="1h",
        )
    )

    assert result["status"] == "started"
    assert len(started) == 1
    argv = started[0][0]
    assert argv[:5] == ["python", "-m", "prosper.cli", "predict", "ml"]
    assert argv[argv.index("--interval") + 1] == "1h"


def test_evaluate_model_predictions_builds_safe_command(monkeypatch) -> None:
    started: list[str] = []

    def fake_start(cmd: str, commands: list[list[str]]) -> None:
        started.append(cmd)
        assert commands[0][:5] == ["python", "-m", "prosper.cli", "eval", "predictions"]

    monkeypatch.setattr(runner, "start", fake_start)

    result = run_evaluation(
        EvalRequest(
            symbol="BTCUSDT",
            model_type="xgboost",
            timestamp="20240501123000",
            interval="1d",
        )
    )

    assert result["status"] == "started"
    assert len(started) == 1
    assert "--model-type xgboost" in started[0]
    assert "--timestamp 20240501123000" in started[0]
