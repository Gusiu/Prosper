"""No request may name a path outside the data lake, and no other site may send one.

Both halves were broken at once. The CORS middleware granted every origin every
method, so any page open in the same browser could call the API; and only the
symbol of a run was checked, so `?interval=1d/../../..` - or, on Windows, a
`%5C` in the timestamp segment - resolved the delete endpoint's `shutil.rmtree`
to a directory outside the lake. Reproduced against a temporary lake before the
fix: the directory was gone.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlencode

import pytest
from fastapi import HTTPException

from prosper.api.commands import build_eval_command, build_train_command
from prosper.api.models import EvalRequest, TrainRequest
from prosper.api.routes.runs import delete_ai_model
from prosper.api.server import app
from prosper.api.validation import require_inside
from prosper.config import Settings
from prosper.storage.layout import (
    resolve_versioned_prediction_dir,
    versioned_prediction_folder_name,
)

TIMESTAMP = "20240501123000"


def _call(method: str, path: str, headers: dict[str, str], query: dict | None = None):
    """Drive the ASGI app directly; returns (status, headers)."""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": urlencode(query or {}).encode(), "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8000),
    }
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}


@pytest.fixture
def lake(tmp_path, monkeypatch) -> tuple[Settings, Path]:
    """A lake with one real run, and a directory beside it that must survive."""
    settings = Settings(data_root=tmp_path / "data")
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)
    run_dir = resolve_versioned_prediction_dir(
        "BTCUSDT", "xgboost", TIMESTAMP, settings=settings, interval="1d"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text("{}", encoding="utf-8")

    victim = tmp_path / "victim_dir"
    victim.mkdir()
    (victim / "keep.txt").write_text("must survive", encoding="utf-8")
    return settings, victim


# ── other origins ────────────────────────────────────────────────────────────

def test_a_preflight_from_another_site_is_not_granted() -> None:
    status, headers = _call(
        "OPTIONS", "/api/data/BTCUSDT",
        {"host": "127.0.0.1:8000", "origin": "https://evil.example",
         "access-control-request-method": "DELETE"},
    )
    assert "access-control-allow-origin" not in headers
    assert "DELETE" not in headers.get("access-control-allow-methods", "")


def test_a_request_naming_another_host_is_refused() -> None:
    """DNS rebinding makes a hostile domain same-origin; its Host header gives it away."""
    status, _ = _call("GET", "/api/task/queue", {"host": "evil.example"})
    assert status == 400


def test_the_dashboard_host_is_still_served() -> None:
    status, _ = _call("GET", "/api/task/queue", {"host": "127.0.0.1:8000"})
    assert status == 200


# ── paths out of the lake ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("model_type", "timestamp", "interval"),
    [
        ("xgboost", "x\\..\\..\\..\\..\\..\\victim_dir", "1d"),  # through the path segment
        ("xgboost", "dir", "1d/../../../../../victim"),  # through the query string
        ("xgboost\\..\\..", TIMESTAMP, "1d"),  # through the model segment
        ("..", TIMESTAMP, "1d"),
    ],
)
def test_delete_refuses_a_run_that_names_another_directory(
    lake, model_type, timestamp, interval
) -> None:
    _, victim = lake
    with pytest.raises(HTTPException) as exc:
        delete_ai_model("BTCUSDT", model_type, timestamp, interval=interval)
    assert exc.value.status_code == 400
    assert (victim / "keep.txt").exists()


def test_delete_refuses_an_unknown_model_or_interval(lake) -> None:
    for model_type, interval in (("svm", "1d"), ("xgboost", "7d")):
        with pytest.raises(HTTPException) as exc:
            delete_ai_model("BTCUSDT", model_type, TIMESTAMP, interval=interval)
        assert exc.value.status_code == 400


def test_delete_still_removes_a_real_run(lake) -> None:
    settings, victim = lake
    result = delete_ai_model("btcusdt", "XGBoost", TIMESTAMP, interval="1d")
    assert result["status"] == "deleted"
    run_dir = resolve_versioned_prediction_dir(
        "BTCUSDT", "xgboost", TIMESTAMP, settings=settings, interval="1d"
    )
    assert not run_dir.exists()
    assert victim.exists()


@pytest.mark.parametrize("bad", ["..", "a/b", "a\\b", "x..y", ""])
def test_the_storage_layer_refuses_a_run_name_that_is_a_path(bad) -> None:
    """The last line: code that forgets to validate still cannot build the path."""
    with pytest.raises(ValueError):
        versioned_prediction_folder_name("xgboost", bad, "1d")
    with pytest.raises(ValueError):
        versioned_prediction_folder_name(bad or "", TIMESTAMP, "1d")
    if bad:
        with pytest.raises(ValueError):
            versioned_prediction_folder_name("xgboost", TIMESTAMP, bad)


def test_require_inside_refuses_a_path_that_resolves_outside(tmp_path) -> None:
    root = tmp_path / "lake"
    root.mkdir()
    assert require_inside(root / "a" / "b", root).is_relative_to(root.resolve())
    with pytest.raises(HTTPException):
        require_inside(root / ".." / "elsewhere", root)


# ── what the API hands to the CLI ────────────────────────────────────────────

def test_train_refuses_an_interval_that_is_not_one() -> None:
    request = TrainRequest(symbol="BTCUSDT", start="2024-01-01", end="2024-06-30",
                           interval="1d/../../x")
    with pytest.raises(ValueError):
        build_train_command(request)


def test_train_renders_numeric_parameters_itself() -> None:
    ok = TrainRequest(symbol="BTCUSDT", model_type="xgboost", start="2024-01-01",
                      end="2024-06-30", params={"max_depth": "4", "learning_rate": 0.05})
    argv = build_train_command(ok)[0]
    assert argv[argv.index("--max-depth") + 1] == "4"
    assert argv[argv.index("--learning-rate") + 1] == "0.05"

    bad = TrainRequest(symbol="BTCUSDT", model_type="xgboost", start="2024-01-01",
                       end="2024-06-30", params={"max_depth": "--root"})
    with pytest.raises(ValueError):
        build_train_command(bad)


@pytest.mark.parametrize(
    ("model_type", "timestamp", "interval"),
    [("svm", TIMESTAMP, "1d"), ("xgboost", "latest", "1d"), ("xgboost", TIMESTAMP, "1d/..")],
)
def test_eval_refuses_a_run_identity_that_is_not_one(model_type, timestamp, interval) -> None:
    request = EvalRequest(symbol="BTCUSDT", model_type=model_type, timestamp=timestamp,
                          interval=interval)
    with pytest.raises(ValueError):
        build_eval_command(request)
