"""Integration tests: predict CLI path, versioned output, and horizon training.

The regression test in here is the reason the module exists: an earlier
implementation shipped a training window narrower than the medium/long
horizons, so those horizons silently emitted a constant placeholder for every
row while the run still reported success.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from prosper.config import Settings
from prosper.features.build import build_features
from prosper.predict.ml import predict_ml
from prosper.storage.layout import get_parquet_file_path, parse_versioned_prediction_folder
from prosper.storage.parquet import save_parquet


def _write_daily_history(settings: Settings, symbol: str, start: datetime, days: int) -> None:
    """Write a contiguous daily OHLCV series, one parquet per calendar month."""
    rng = random.Random(7)
    rows = []
    price = 100.0
    for i in range(days):
        price *= 1.0 + rng.gauss(0.0004, 0.02)
        ts = start + timedelta(days=i)
        rows.append(
            {
                "open_time": ts,
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
            get_parquet_file_path(symbol, "1d", int(year), month=int(month), settings=settings),
        )


@pytest.fixture
def prepared_lake(tmp_path):
    """A data lake with enough daily history to train every default horizon."""
    settings = Settings(data_root=tmp_path)
    symbol = "TESTUSDT"
    # 730d window + 365d long horizon + prediction range needs ~4 years.
    _write_daily_history(settings, symbol, datetime(2019, 1, 1, tzinfo=UTC), 1500)
    build_features(symbol, "1d", "2019-01-01", "2023-02-28", settings=settings)
    return settings, symbol


def _read_versioned_rows(settings: Settings, symbol: str) -> tuple[str, list[dict]]:
    pred_root = settings.reports_predictions_dir / symbol
    versioned = [
        d for d in pred_root.iterdir() if d.is_dir() and parse_versioned_prediction_folder(d.name)
    ]
    assert len(versioned) == 1
    lines = (versioned[0] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return versioned[0].name, [json.loads(line) for line in lines if line.strip()]


def test_predict_ml_writes_interval_versioned_folder(prepared_lake) -> None:
    settings, symbol = prepared_lake

    result = predict_ml(
        symbol=symbol,
        start="2022-06-01",
        end="2022-08-31",
        settings=settings,
        interval="1d",
        train_window_days=730,
    )

    assert "error" not in result
    assert result["predictions"] > 0

    folder_name, rows = _read_versioned_rows(settings, symbol)
    parsed = parse_versioned_prediction_folder(folder_name)
    assert parsed == ("ml", "1d", parsed[2])
    assert len(rows) == result["predictions"]


def test_every_horizon_is_actually_trained(prepared_lake) -> None:
    """No horizon may fall back to the untrained placeholder for the whole run.

    This is the regression guard for the training-window collapse: with a
    window narrower than the horizon, `medium` and `long` used to be 100%
    placeholder while the run still looked successful.
    """
    settings, symbol = prepared_lake

    result = predict_ml(
        symbol=symbol,
        start="2022-06-01",
        end="2022-08-31",
        settings=settings,
        interval="1d",
        train_window_days=730,
    )

    total = result["predictions"]
    for horizon, untrained in result["untrained_horizons"].items():
        assert untrained == 0, f"{horizon}: {untrained}/{total} rows were never trained"

    _, rows = _read_versioned_rows(settings, symbol)
    uniform = 1.0 / 3.0
    for horizon in ("short", "medium", "long"):
        payloads = [row[horizon] for row in rows]
        assert all(p["trained"] for p in payloads), f"{horizon} emitted untrained payloads"

        # A trained classifier must not return the uniform prior everywhere.
        constant = sum(1 for p in payloads if math.isclose(p["P_long"], uniform, abs_tol=1e-9))
        assert constant / len(payloads) < 0.05, (
            f"{horizon}: {constant}/{len(payloads)} rows equal the uniform prior"
        )


def test_predict_ml_rejects_window_narrower_than_horizon(prepared_lake) -> None:
    """A window that cannot train every horizon is a hard error, not a silent fallback."""
    settings, symbol = prepared_lake

    with pytest.raises(ValueError, match="too small at interval 1d"):
        predict_ml(
            symbol=symbol,
            start="2022-06-01",
            end="2022-08-31",
            settings=settings,
            interval="1d",
            train_window_days=150,
        )


def test_predictions_carry_unique_open_time_per_row(prepared_lake) -> None:
    """`date` collides on sub-daily intervals, so `open_time` is the real key."""
    settings, symbol = prepared_lake

    predict_ml(
        symbol=symbol,
        start="2022-06-01",
        end="2022-07-31",
        settings=settings,
        interval="1d",
        train_window_days=730,
    )

    _, rows = _read_versioned_rows(settings, symbol)
    open_times = [row["open_time"] for row in rows]
    assert len(set(open_times)) == len(open_times)
    assert all(row["date"] == row["open_time"][:10] for row in rows)
