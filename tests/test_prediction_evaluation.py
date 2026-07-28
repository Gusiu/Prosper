"""Tests for versioned prediction-quality evaluation."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from prosper.config import Settings
from prosper.eval.predictions import (
    brier_score,
    evaluate_predictions,
    expected_calibration_error,
    negative_log_likelihood,
    normalize_probabilities,
    probability_entropy,
)
from prosper.storage.layout import resolve_versioned_prediction_dir
from prosper.storage.parquet import save_parquet


def test_prediction_metric_functions_are_stable() -> None:
    probs = normalize_probabilities({"P_long": 0.8, "P_flat": 0.1, "P_short": 0.1})

    assert probs == {"short": 0.1, "flat": 0.1, "long": 0.8}
    assert brier_score(probs, "long") < brier_score(probs, "short")
    assert negative_log_likelihood(probs, "long") == -math.log(0.8)
    assert 0.0 < probability_entropy(probs) < math.log(3.0)

    ece, curve = expected_calibration_error(
        [
            {"confidence": 0.8, "correct": True},
            {"confidence": 0.8, "correct": False},
        ],
        bins=5,
    )
    assert ece == pytest.approx(0.3)
    assert sum(bucket["count"] for bucket in curve) == 2


def test_evaluate_predictions_writes_quality_artifacts(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    symbol = "BTCUSDT"
    timestamp = "20240501123000"

    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = 400
    df = pl.DataFrame(
        {
            "open_time": [start + timedelta(days=i) for i in range(rows)],
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.0 + i for i in range(rows)],
            "volume": [1000.0 + i for i in range(rows)],
        }
    )
    save_parquet(
        df,
        settings.processed_binance_spot_klines_dir
        / "1d"
        / f"symbol={symbol}"
        / "year=2024"
        / "month=01"
        / "part.parquet",
    )

    run_dir = resolve_versioned_prediction_dir(
        symbol,
        "baseline",
        timestamp,
        settings=settings,
        interval="1d",
    )
    run_dir.mkdir(parents=True)
    depth = {
        "1-2": 0.01,
        "2-3": 0.01,
        "3-5": 0.01,
        "5-8": 0.01,
        "8-13": 0.01,
        "13-21": 0.05,
        "21-34": 0.1,
        "34+": 0.8,
    }
    prediction = {
        "date": "2024-01-01",
        "symbol": symbol,
        "short": {
            "P_long": 0.8,
            "P_flat": 0.1,
            "P_short": 0.1,
            "depth_long_bins": depth,
            "depth_short_bins": depth,
        },
        "medium": {
            "P_long": 0.8,
            "P_flat": 0.1,
            "P_short": 0.1,
            "depth_long_bins": depth,
            "depth_short_bins": depth,
        },
        "long": {
            "P_long": 0.8,
            "P_flat": 0.1,
            "P_short": 0.1,
            "depth_long_bins": depth,
            "depth_short_bins": depth,
        },
    }
    (run_dir / "predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    result = evaluate_predictions(
        symbol=symbol,
        model_type="baseline",
        timestamp=timestamp,
        interval="1d",
        settings=settings,
    )

    artifacts = result["artifacts"]
    assert result["metrics"]["overall"]["samples"] == 3
    assert result["metrics"]["overall"]["accuracy"] == 1.0
    assert result["metrics"]["overall"]["model_score"] > 70.0
    assert len(result["worst_predictions"]) == 3
    assert len(result["recommendations"]) == 3

    for artifact_path in artifacts.values():
        assert artifact_path
        assert Path(artifact_path).exists()

    quality_path = Path(artifacts["predictions_quality"])
    assert quality_path.suffix == ".parquet"
    written = pl.read_parquet(quality_path)
    assert len(written) == 3
    # Flattened, not nested: the API scans this file lazily.
    assert {"pred_direction", "actual_direction", "quality_score"} <= set(written.columns)


def _write_klines(settings: Settings, symbol: str, rows: int) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    save_parquet(
        pl.DataFrame(
            {
                "open_time": [start + timedelta(days=i) for i in range(rows)],
                "open": [100.0 + i for i in range(rows)],
                "high": [101.0 + i for i in range(rows)],
                "low": [99.0 + i for i in range(rows)],
                "close": [100.0 + i for i in range(rows)],
                "volume": [1000.0 + i for i in range(rows)],
            }
        ),
        settings.processed_binance_spot_klines_dir
        / "1d"
        / f"symbol={symbol}"
        / "year=2024"
        / "month=01"
        / "part.parquet",
    )


def _horizon_payload(trained: bool) -> dict:
    depth = {label: 0.125 for label in ("1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-34", "34+")}
    if trained:
        return {
            "P_long": 0.8,
            "P_flat": 0.1,
            "P_short": 0.1,
            "depth_long_bins": depth,
            "depth_short_bins": depth,
            "trained": True,
        }
    return {
        "P_long": 1 / 3,
        "P_flat": 1 / 3,
        "P_short": 1 / 3,
        "depth_long_bins": depth,
        "depth_short_bins": depth,
        "trained": False,
    }


def test_untrained_horizons_are_excluded_from_metrics(tmp_path) -> None:
    """A placeholder horizon must not be scored as if it were a forecast."""
    settings = Settings(data_root=tmp_path)
    symbol = "BTCUSDT"
    timestamp = "20240501123000"
    _write_klines(settings, symbol, 400)

    run_dir = resolve_versioned_prediction_dir(
        symbol, "ml", timestamp, settings=settings, interval="1d"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "open_time": "2024-01-01T00:00:00+00:00",
                "date": "2024-01-01",
                "symbol": symbol,
                "short": _horizon_payload(trained=True),
                "medium": _horizon_payload(trained=False),
                "long": _horizon_payload(trained=False),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_predictions(
        symbol=symbol,
        model_type="ml",
        timestamp=timestamp,
        interval="1d",
        settings=settings,
    )

    metrics = result["metrics"]
    assert metrics["overall"]["samples"] == 1
    assert metrics["overall"]["untrained_rows"] == 2
    assert metrics["by_horizon"]["short"]["samples"] == 1
    assert metrics["by_horizon"]["medium"]["samples"] == 0
    assert metrics["by_horizon"]["medium"]["untrained_rows"] == 1
    assert metrics["by_horizon"]["long"]["untrained_rows"] == 1
    # The composite score must come from the trained horizon alone.
    assert metrics["by_horizon"]["medium"]["score"] is None


def test_sub_daily_predictions_score_against_their_own_bar(tmp_path) -> None:
    """Hourly rows share a calendar date; `open_time` must disambiguate them."""
    settings = Settings(data_root=tmp_path)
    symbol = "BTCUSDT"
    timestamp = "20240501123000"

    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = 900
    save_parquet(
        pl.DataFrame(
            {
                "open_time": [start + timedelta(hours=i) for i in range(bars)],
                "open": [100.0 + i for i in range(bars)],
                "high": [101.0 + i for i in range(bars)],
                "low": [99.0 + i for i in range(bars)],
                "close": [100.0 + i for i in range(bars)],
                "volume": [1000.0 + i for i in range(bars)],
            }
        ),
        settings.processed_binance_spot_klines_dir
        / "1h"
        / f"symbol={symbol}"
        / "year=2024"
        / "month=01"
        / "part.parquet",
    )

    run_dir = resolve_versioned_prediction_dir(
        symbol, "ml", timestamp, settings=settings, interval="1h"
    )
    run_dir.mkdir(parents=True)
    rows = [
        {
            "open_time": (start + timedelta(hours=hour)).isoformat(),
            "date": "2024-01-01",
            "symbol": symbol,
            "short": _horizon_payload(trained=True),
            "medium": _horizon_payload(trained=False),
            "long": _horizon_payload(trained=False),
        }
        for hour in range(3)
    ]
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    result = evaluate_predictions(
        symbol=symbol,
        model_type="ml",
        timestamp=timestamp,
        interval="1h",
        settings=settings,
    )

    quality_rows = pl.read_parquet(result["artifacts"]["predictions_quality"]).to_dicts()
    # Three distinct bars, each resolved to its own timestamp rather than all
    # collapsing onto the first bar of 2024-01-01.
    assert {row["open_time"] for row in quality_rows} == {row["open_time"] for row in rows}
    # The short horizon spans 28 calendar days = 672 hourly bars at this interval.
    assert {row["forward_steps"] for row in quality_rows} == {28 * 24}
