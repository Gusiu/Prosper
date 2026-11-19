"""Every calibrated forecast says what calibration did to it.

A run kept only the calibrated probabilities, so whether calibration helped,
and whether a temperature ran into the end of the grid, could be argued but
not measured. Each trained payload now carries the probabilities before the
calibrator, the temperature, where it came from, and how many held-out rows
chose it.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from prosper.config import Settings
from prosper.domain import DIRECTION_CLASSES, HORIZON_NAMES
from prosper.features.build import build_features
from prosper.predict.calibration import TEMPERATURE_GRID, TemperatureCalibrator
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import save_parquet

SYMBOL = "TESTUSDT"


def _write_history(settings: Settings, days: int) -> None:
    """A slow cycle plus noise, so every horizon sees both directions."""
    rng = random.Random(5)
    rows = []
    level = 100.0
    start = datetime(2018, 1, 1, tzinfo=UTC)
    for i in range(days):
        cycle = 1.0 + 0.6 * math.sin(2 * math.pi * i / 500.0)
        level *= 1.0 + rng.gauss(0.0, 0.015)
        price = level * cycle
        rows.append(
            {
                "open_time": start + timedelta(days=i),
                "open": price * 0.995,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "volume": 1000.0 + i,
            }
        )
    df = pl.DataFrame(rows).with_columns(
        pl.col("open_time").dt.year().alias("_y"),
        pl.col("open_time").dt.month().alias("_m"),
    )
    for (year, month), group in df.group_by(["_y", "_m"]):
        save_parquet(
            group.drop(["_y", "_m"]).sort("open_time"),
            get_parquet_file_path(SYMBOL, "1d", int(year), month=int(month), settings=settings),
        )


@pytest.fixture
def lake(tmp_path):
    settings = Settings(data_root=tmp_path)
    _write_history(settings, 2200)
    build_features(SYMBOL, "1d", "2018-01-01", "2024-01-10", settings=settings)
    return settings


def _payloads(settings: Settings, model: str) -> list[dict]:
    run_dir = next((settings.reports_predictions_dir / SYMBOL).glob(f"{model}_1d_*"))
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row[h] for row in rows for h in HORIZON_NAMES if row[h]["trained"]]


def _assert_recorded(payloads: list[dict]) -> None:
    assert payloads, "nothing was trained, so nothing was checked"
    fitted = 0
    for payload in payloads:
        before = payload["uncalibrated"]
        assert set(before) == set(DIRECTION_CLASSES)
        assert sum(before.values()) == pytest.approx(1.0)
        assert payload["temperature"] in TEMPERATURE_GRID
        if payload["calibrated_on"]:
            fitted += 1
            assert payload["temperature_source"] == "fitted"
        else:
            assert payload["temperature_source"] in ("carried", "prior")
        after = TemperatureCalibrator(payload["temperature"]).apply_to_mapping(before)
        for name in DIRECTION_CLASSES:
            assert payload[f"P_{name}"] == pytest.approx(after[name], abs=1e-9)
    assert fitted, "no payload was calibrated on a held-out slice"


def test_xgboost_records_what_calibration_did(lake) -> None:
    from prosper.predict.xgboost_model import predict_xgboost

    predict_xgboost(
        symbol=SYMBOL, start="2022-06-01", end="2022-07-31", settings=lake, interval="1d"
    )
    _assert_recorded(_payloads(lake, "xgboost"))


def test_gradient_boosting_records_what_calibration_did(lake) -> None:
    from prosper.predict.ml import predict_ml

    predict_ml(symbol=SYMBOL, start="2022-06-01", end="2022-07-31", settings=lake, interval="1d")
    _assert_recorded(_payloads(lake, "ml"))


@pytest.mark.slow
def test_gru_records_what_calibration_did(lake) -> None:
    from prosper.predict.gru import predict_gru

    predict_gru(
        symbol=SYMBOL, start="2022-06-01", end="2022-06-30", settings=lake,
        interval="1d", epochs=1, hidden_size=8,
    )
    _assert_recorded(_payloads(lake, "gru"))


@pytest.mark.slow
def test_tft_records_what_calibration_did(lake) -> None:
    from prosper.predict.tft import predict_tft

    predict_tft(
        symbol=SYMBOL, start="2022-06-01", end="2022-06-20", settings=lake, interval="1d",
        seq_len=30, max_epochs=1, hidden_size=8,
    )
    _assert_recorded(_payloads(lake, "tft"))
