"""Guards for the bounded-response and bounded-memory behaviour of the API.

Before these, opening the Evaluation tab for a 1h run pulled ~71 MB (the whole
recommendation set, so the UI could render fifty rows), a chart request scanned
every monthly parquet of a symbol regardless of the date range, and the task
log grew without limit for the lifetime of the process.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from prosper.api.server import (
    MAX_TASK_LOG_LINES,
    TaskRunner,
    get_ai_evaluation_detail,
    get_ai_evaluation_flags,
)
from prosper.config import Settings
from prosper.eval.predictions import evaluate_predictions
from prosper.inventory import DataManager
from prosper.storage.layout import get_parquet_file_path, resolve_versioned_prediction_dir
from prosper.storage.parquet import save_parquet

SYMBOL = "BTCUSDT"
TIMESTAMP = "20240501123000"


def _depth() -> dict[str, float]:
    return {
        label: 0.125
        for label in ("1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-34", "34+")
    }


def _evaluated_run(tmp_path, rows: int = 300) -> Settings:
    settings = Settings(data_root=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    save_parquet(
        pl.DataFrame(
            {
                "open_time": [start + timedelta(days=i) for i in range(rows + 40)],
                "open": [100.0 + i for i in range(rows + 40)],
                "high": [101.0 + i for i in range(rows + 40)],
                "low": [99.0 + i for i in range(rows + 40)],
                # A trend under the 7-day cycle. Without it close[i + 28] equals
                # close[i] exactly — 28 and 182 are both multiples of 7 — so
                # every short and medium outcome had a zero forward return and
                # no side to be scored against.
                "close": [100.0 + i * 0.5 + (i % 7) * 3 for i in range(rows + 40)],
                "volume": [1000.0 + i for i in range(rows + 40)],
            }
        ),
        get_parquet_file_path(SYMBOL, "1d", 2024, month=1, settings=settings),
    )

    run_dir = resolve_versioned_prediction_dir(
        SYMBOL, "ml", TIMESTAMP, settings=settings, interval="1d"
    )
    run_dir.mkdir(parents=True)
    lines = []
    for i in range(rows):
        ts = start + timedelta(days=i)
        horizon = {
            "P_long": 0.8 if i % 2 else 0.1,
            "P_short": 0.1 if i % 2 else 0.8,
            "depth_long_bins": _depth(),
            "depth_short_bins": _depth(),
            "trained": True,
        }
        lines.append(
            json.dumps(
                {
                    "open_time": ts.isoformat(),
                    "date": ts.date().isoformat(),
                    "symbol": SYMBOL,
                    "short": horizon,
                    "medium": horizon,
                    "long": horizon,
                }
            )
        )
    (run_dir / "predictions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    evaluate_predictions(
        symbol=SYMBOL, model_type="ml", timestamp=TIMESTAMP, interval="1d", settings=settings
    )
    return settings


def test_evaluation_detail_is_bounded_by_limit(tmp_path, monkeypatch) -> None:
    settings = _evaluated_run(tmp_path)
    monkeypatch.setattr("prosper.api.routes.evaluations.get_settings", lambda: settings)

    detail = get_ai_evaluation_detail(SYMBOL, "ml", TIMESTAMP, interval="1d", limit=25)

    assert len(detail["worst_predictions"]) == 25
    assert detail["worst_predictions_total"] > 25
    assert len(detail["recommendations"]) <= 25
    # The heavy nested quality set is no longer part of the response at all.
    assert "quality_rows" not in detail


def test_evaluation_detail_filters_by_flag_server_side(tmp_path, monkeypatch) -> None:
    settings = _evaluated_run(tmp_path)
    monkeypatch.setattr("prosper.api.routes.evaluations.get_settings", lambda: settings)

    flags = get_ai_evaluation_flags(SYMBOL, "ml", TIMESTAMP, interval="1d")["flags"]
    assert flags, "expected at least one quality flag in the fixture"
    target = flags[0]["flag"]

    detail = get_ai_evaluation_detail(
        SYMBOL, "ml", TIMESTAMP, interval="1d", limit=500, flag=target
    )
    assert detail["worst_predictions"]
    assert all(target in row["flags"] for row in detail["worst_predictions"])
    assert detail["worst_predictions_total"] == next(
        item["count"] for item in flags if item["flag"] == target
    )


def test_worst_predictions_are_sorted_by_error(tmp_path, monkeypatch) -> None:
    settings = _evaluated_run(tmp_path)
    monkeypatch.setattr("prosper.api.routes.evaluations.get_settings", lambda: settings)

    rows = get_ai_evaluation_detail(SYMBOL, "ml", TIMESTAMP, interval="1d", limit=50)[
        "worst_predictions"
    ]
    scores = [row["error_score"] for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_evaluation_detail_rejects_an_absurd_limit(tmp_path, monkeypatch) -> None:
    settings = _evaluated_run(tmp_path)
    monkeypatch.setattr("prosper.api.routes.evaluations.get_settings", lambda: settings)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        get_ai_evaluation_detail(SYMBOL, "ml", TIMESTAMP, interval="1d", limit=50_000)
    assert exc.value.status_code == 400


def test_chart_scan_prunes_partitions_outside_the_range(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    base = settings.processed_binance_spot_klines_dir / "1d" / f"symbol={SYMBOL}"
    for month in range(1, 13):
        start = datetime(2024, month, 1, tzinfo=UTC)
        save_parquet(
            pl.DataFrame(
                {
                    "open_time": [start + timedelta(days=i) for i in range(5)],
                    "open": [100.0] * 5,
                    "high": [101.0] * 5,
                    "low": [99.0] * 5,
                    "close": [100.0] * 5,
                    "volume": [10.0] * 5,
                }
            ),
            get_parquet_file_path(SYMBOL, "1d", 2024, month=month, settings=settings),
        )

    manager = DataManager(settings)
    everything = manager._scan_parquet_tree(base)
    assert everything is not None
    assert len(everything.collect()) == 60

    march_only = manager._scan_parquet_tree(
        base,
        datetime(2024, 3, 1, tzinfo=UTC),
        datetime(2024, 3, 31, tzinfo=UTC),
    )
    assert march_only is not None
    assert len(march_only.collect()) == 5


def test_partition_pruning_keeps_unparseable_names(tmp_path) -> None:
    """A malformed partition name must never hide data."""
    settings = Settings(data_root=tmp_path)
    manager = DataManager(settings)
    year_dir = tmp_path / "year=2024"
    assert manager._partition_in_range(
        year_dir,
        tmp_path / "month=not-a-number",
        datetime(2024, 3, 1, tzinfo=UTC),
        datetime(2024, 3, 31, tzinfo=UTC),
    )


def test_task_log_buffer_is_bounded_and_cursor_stays_absolute() -> None:
    runner = TaskRunner()
    total = MAX_TASK_LOG_LINES + 250
    for i in range(total):
        runner._append_log(f"line {i}")

    assert len(runner.logs) == MAX_TASK_LOG_LINES
    assert runner.dropped_log_lines == 250

    # A client that has read nothing gets the surviving window and a cursor
    # that already accounts for the trimmed lines.
    lines, next_idx = runner.read_logs(0)
    assert lines[0] == "line 250"
    assert next_idx == total

    # A client mid-stream resumes without re-reading or skipping.
    lines, next_idx = runner.read_logs(total - 10)
    assert lines == [f"line {i}" for i in range(total - 10, total)]
    assert next_idx == total


def test_prediction_responses_are_capped(tmp_path, monkeypatch) -> None:
    """A whole run must not be serialised into one response.

    Asking for every bar of a 1d run returned 5 MB; a 1h run over the same
    span is 24x that. The endpoint accepted no `limit` at all, so the only
    protection was a docstring asking callers to pass year/month.
    """
    from prosper.api.routes.runs import get_ai_predictions

    settings = _evaluated_run(tmp_path, rows=300)
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

    payload = get_ai_predictions(
        symbol=SYMBOL, model_type="ml", timestamp=TIMESTAMP, interval="1d", limit=5
    )

    assert len(payload["predictions"]) == 5
    assert payload["matched"] == 300
    assert payload["truncated"] is True


def test_the_cap_cannot_be_raised_from_the_query_string(tmp_path, monkeypatch) -> None:
    from prosper.api.routes.runs import MAX_PREDICTION_ROWS, get_ai_predictions

    settings = _evaluated_run(tmp_path, rows=50)
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

    payload = get_ai_predictions(
        symbol=SYMBOL, model_type="ml", timestamp=TIMESTAMP, interval="1d", limit=10**9
    )

    assert payload["limit"] == MAX_PREDICTION_ROWS


def test_an_untruncated_response_says_so(tmp_path, monkeypatch) -> None:
    from prosper.api.routes.runs import get_ai_predictions

    settings = _evaluated_run(tmp_path, rows=20)
    monkeypatch.setattr("prosper.api.routes.runs.get_settings", lambda: settings)

    payload = get_ai_predictions(symbol=SYMBOL, model_type="ml", timestamp=TIMESTAMP, interval="1d")

    assert payload["truncated"] is False
    assert payload["count"] == payload["matched"] == 20
