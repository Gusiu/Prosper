"""Tests for versioned prediction folder helpers and AI model API."""

import json
from types import SimpleNamespace

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
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

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
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

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
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

    with pytest.raises(HTTPException) as exc:
        delete_ai_model("BTCUSDT", "tft", "20990101123000", interval="1d")

    assert exc.value.status_code == 404


def test_get_ai_predictions_versioned_run(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

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
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

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

    def fake_submit(name: str, commands: list[list[str]]):
        started.append(commands)
        return SimpleNamespace(to_dict=lambda: {"id": 1, "name": name})

    monkeypatch.setattr(runner, "submit", fake_submit)
    monkeypatch.setattr(runner, "snapshot", lambda: {"queued": 0, "pending": []})

    result = run_training(
        TrainRequest(
            symbol="BTCUSDT",
            model_type="ml",
            start="2024-01-01",
            end="2024-06-01",
            interval="1h",
        )
    )

    assert result["status"] == "queued"
    assert len(started) == 1
    argv = started[0][0]
    assert argv[:5] == ["python", "-m", "prosper.cli", "predict", "ml"]
    assert argv[argv.index("--interval") + 1] == "1h"


def test_evaluate_model_predictions_builds_safe_command(monkeypatch) -> None:
    started: list[list[str]] = []

    def fake_submit(name: str, commands: list[list[str]]):
        started.append(commands[0])
        return SimpleNamespace(to_dict=lambda: {"id": 1, "name": name})

    monkeypatch.setattr(runner, "submit", fake_submit)
    monkeypatch.setattr(runner, "snapshot", lambda: {"queued": 0, "pending": []})

    result = run_evaluation(
        EvalRequest(
            symbol="BTCUSDT",
            model_type="xgboost",
            timestamp="20240501123000",
            interval="1d",
        )
    )

    assert result["status"] == "queued"
    assert len(started) == 1
    argv = started[0]
    assert argv[:5] == ["python", "-m", "prosper.cli", "eval", "predictions"]
    assert argv[argv.index("--model-type") + 1] == "xgboost"
    assert argv[argv.index("--timestamp") + 1] == "20240501123000"


def test_deleting_a_run_removes_its_derived_artifacts(tmp_path, monkeypatch) -> None:
    """Invariant 5: an artifact names its source run, so it cannot outlive it.

    A deleted run used to leave its backtest and recommendations on disk, and
    any reader globbing those directories reported the dead run's numbers next
    to current ones.
    """
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

    slug = "xgboost_1d_20240501123000"
    run_dir = resolve_versioned_prediction_dir(
        "BTCUSDT", "xgboost", "20240501123000", settings=settings, interval="1d"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text("{}", encoding="utf-8")

    derived = [
        settings.reports_dir / "backtests" / "BTCUSDT" / slug,
        settings.reports_dir / "evaluations" / "BTCUSDT" / slug,
        settings.reports_recommendations_dir / "BTCUSDT" / slug,
        settings.reports_eval_dir / "BTCUSDT" / slug,
    ]
    for path in derived:
        path.mkdir(parents=True)
        (path / "artifact.json").write_text("{}", encoding="utf-8")

    result = delete_ai_model("BTCUSDT", "xgboost", "20240501123000", interval="1d")

    assert result["status"] == "deleted"
    assert not run_dir.exists()
    for path in derived:
        assert not path.exists(), f"{path} outlived its run"


def test_deleting_a_run_without_derived_artifacts_still_works(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

    run_dir = resolve_versioned_prediction_dir(
        "BTCUSDT", "ml", "20240501123000", settings=settings, interval="1d"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text("{}", encoding="utf-8")

    result = delete_ai_model("BTCUSDT", "ml", "20240501123000", interval="1d")

    assert result["status"] == "deleted"
    assert len(result["removed"]) == 1


def test_another_runs_artifacts_are_left_alone(tmp_path, monkeypatch) -> None:
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

    for ts in ("20240501123000", "20240601123000"):
        d = resolve_versioned_prediction_dir(
            "BTCUSDT", "ml", ts, settings=settings, interval="1d"
        )
        d.mkdir(parents=True)
        (d / "predictions.jsonl").write_text("{}", encoding="utf-8")
        bt = settings.reports_dir / "backtests" / "BTCUSDT" / f"ml_1d_{ts}"
        bt.mkdir(parents=True)
        (bt / "backtest.json").write_text("{}", encoding="utf-8")

    delete_ai_model("BTCUSDT", "ml", "20240501123000", interval="1d")

    survivor = settings.reports_dir / "backtests" / "BTCUSDT" / "ml_1d_20240601123000"
    assert survivor.exists()
