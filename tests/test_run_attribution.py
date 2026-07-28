"""Every derived artifact must name the prediction run it came from.

Before this, all models wrote into one shared `reports/predictions/{symbol}/daily/`
folder, so planner windows, walk-forward metrics and backtests silently
described whichever model had run last.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from prosper.config import Settings
from prosper.eval.backtest import run_backtest
from prosper.planner.windows import plan_windows
from prosper.storage.layout import get_parquet_file_path, get_recommendation_report_path
from prosper.storage.parquet import save_parquet
from prosper.storage.runs import list_runs, load_run_predictions, resolve_run

SYMBOL = "TESTUSDT"


def _horizon(p_long: float, p_short: float, trained: bool = True) -> dict:
    depth = {
        label: 0.125
        for label in ("1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-34", "34+")
    }
    return {
        "P_long": p_long,
        "P_flat": max(0.0, 1.0 - p_long - p_short),
        "P_short": p_short,
        "depth_long_bins": depth,
        "depth_short_bins": depth,
        "trained": trained,
    }


def _write_run(
    settings: Settings,
    model_type: str,
    timestamp: str,
    *,
    interval: str = "1d",
    p_long: float = 0.9,
    p_short: float = 0.05,
    days: int = 40,
    trained: bool = True,
) -> None:
    run_dir = settings.reports_predictions_dir / SYMBOL / f"{model_type}_{interval}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "open_time": (start + timedelta(days=i)).isoformat(),
            "date": (start + timedelta(days=i)).date().isoformat(),
            "symbol": SYMBOL,
            "short": _horizon(p_long, p_short, trained),
            "medium": _horizon(p_long, p_short, trained),
            "long": _horizon(p_long, p_short, trained),
        }
        for i in range(days)
    ]
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _write_klines(settings: Settings, days: int = 60) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "open_time": [start + timedelta(days=i) for i in range(days)],
            "open": [100.0 + i for i in range(days)],
            "high": [101.0 + i for i in range(days)],
            "low": [99.0 + i for i in range(days)],
            "close": [100.0 + i for i in range(days)],
            "volume": [1000.0 + i for i in range(days)],
        }
    ).with_columns(pl.col("open_time").dt.month().alias("_m"))
    for (month,), group in df.group_by(["_m"]):
        save_parquet(
            group.drop("_m").sort("open_time"),
            get_parquet_file_path(SYMBOL, "1d", 2024, month=int(month), settings=settings),
        )


@pytest.fixture
def lake(tmp_path):
    settings = Settings(data_root=tmp_path)
    _write_klines(settings)
    return settings


def test_list_runs_returns_newest_first(lake) -> None:
    _write_run(lake, "ml", "20240101000000")
    _write_run(lake, "gru", "20240301000000")
    _write_run(lake, "xgboost", "20240201000000")

    runs = list_runs(symbol=SYMBOL, settings=lake)

    assert [run.model_type for run in runs] == ["gru", "xgboost", "ml"]
    assert runs[0].slug == "gru_1d_20240301000000"


def test_list_runs_filters_by_model_and_interval(lake) -> None:
    _write_run(lake, "ml", "20240101000000", interval="1d")
    _write_run(lake, "ml", "20240102000000", interval="1h")

    assert len(list_runs(symbol=SYMBOL, model_type="ml", settings=lake)) == 2
    hourly = list_runs(symbol=SYMBOL, model_type="ml", interval="1h", settings=lake)
    assert [run.timestamp for run in hourly] == ["20240102000000"]


def test_resolve_run_defaults_to_latest_but_honours_an_explicit_pick(lake) -> None:
    _write_run(lake, "ml", "20240101000000")
    _write_run(lake, "gru", "20240301000000")

    assert resolve_run(SYMBOL, settings=lake).model_type == "gru"

    explicit = resolve_run(
        SYMBOL, model_type="ml", timestamp="20240101000000", interval="1d", settings=lake
    )
    assert explicit.model_type == "ml"
    assert len(load_run_predictions(explicit)) == 40


def test_resolve_run_lists_candidates_when_nothing_matches(lake) -> None:
    _write_run(lake, "ml", "20240101000000")

    with pytest.raises(FileNotFoundError) as exc:
        resolve_run(SYMBOL, model_type="tft", settings=lake)

    assert "ml_1d_20240101000000" in str(exc.value)


def test_planner_windows_are_written_under_the_source_run(lake) -> None:
    _write_run(lake, "ml", "20240101000000", p_long=0.9, p_short=0.05)
    _write_run(lake, "gru", "20240301000000", p_long=0.05, p_short=0.9)

    ml_result = plan_windows(
        SYMBOL,
        settings=lake,
        model_type="ml",
        timestamp="20240101000000",
        interval="1d",
    )
    gru_result = plan_windows(
        SYMBOL,
        settings=lake,
        model_type="gru",
        timestamp="20240301000000",
        interval="1d",
    )

    assert ml_result["run"]["slug"] == "ml_1d_20240101000000"
    assert gru_result["run"]["slug"] == "gru_1d_20240301000000"

    # The two models disagree, so their recommendations must not share a file.
    ml_path = get_recommendation_report_path(
        SYMBOL, 2024, 1, settings=lake, run_slug="ml_1d_20240101000000"
    )
    gru_path = get_recommendation_report_path(
        SYMBOL, 2024, 1, settings=lake, run_slug="gru_1d_20240301000000"
    )
    assert ml_path != gru_path
    assert ml_path.exists() and gru_path.exists()

    ml_payload = json.loads(ml_path.read_text(encoding="utf-8"))
    gru_payload = json.loads(gru_path.read_text(encoding="utf-8"))
    assert ml_payload["model_type"] == "ml"
    assert gru_payload["model_type"] == "gru"
    assert {w["recommendation"] for w in ml_payload["windows"]} != {
        w["recommendation"] for w in gru_payload["windows"]
    }


def test_planner_skips_untrained_horizons(lake) -> None:
    _write_run(lake, "ml", "20240101000000", trained=False)

    result = plan_windows(
        SYMBOL, settings=lake, model_type="ml", timestamp="20240101000000", interval="1d"
    )

    assert result["skipped_untrained_horizons"] > 0
    assert result["total_windows"] == 0


def test_backtest_reports_the_run_it_simulated(lake) -> None:
    _write_run(lake, "ml", "20240101000000", p_long=0.95, p_short=0.01)

    result = run_backtest(
        SYMBOL,
        start="2024-01-01",
        end="2024-02-05",
        settings=lake,
        model_type="ml",
        timestamp="20240101000000",
        interval="1d",
    )

    assert "error" not in result
    assert result["run"]["slug"] == "ml_1d_20240101000000"
    assert result["total_trades"] >= 1
    # The MVP execution model must stay visible to whoever reads the numbers.
    assert any("same bar's close" in note for note in result["limitations"])


def test_backtest_refuses_a_run_with_no_trained_rows(lake) -> None:
    _write_run(lake, "ml", "20240101000000", trained=False)

    result = run_backtest(
        SYMBOL,
        start="2024-01-01",
        end="2024-02-05",
        settings=lake,
        model_type="ml",
        timestamp="20240101000000",
        interval="1d",
    )

    assert "error" in result
    assert "untrained" in result["error"]


def test_backtest_without_any_run_explains_itself(lake) -> None:
    result = run_backtest(SYMBOL, start="2024-01-01", end="2024-02-05", settings=lake)

    assert "error" in result
    assert "No prediction runs found" in result["error"]
