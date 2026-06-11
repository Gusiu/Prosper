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
    rows_written = quality_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows_written) == 3
