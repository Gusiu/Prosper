"""Edge cases where a predictor could quietly stop learning.

Both cases here produced runs that looked successful while a horizon emitted
nothing but the untrained placeholder.
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
from prosper.predict.baseline import predict_baseline
from prosper.predict.xgboost_model import predict_xgboost
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import save_parquet

SYMBOL = "TESTUSDT"


def _write_history(settings: Settings, days: int) -> None:
    """A slow cycle plus noise, so every horizon sees both directions.

    A monotone series would make the 365-day horizon single-class, which is a
    legitimate reason to skip training and would mask the bug under test.
    """
    rng = random.Random(11)
    rows = []
    level = 100.0
    start = datetime(2018, 1, 1, tzinfo=UTC)
    for i in range(days):
        cycle = 1.0 + 0.6 * math.sin(2 * math.pi * i / 500.0)
        level *= 1.0 + rng.gauss(0.0, 0.015)
        price = level * cycle
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
            get_parquet_file_path(SYMBOL, "1d", int(year), month=int(month), settings=settings),
        )


@pytest.fixture
def lake(tmp_path):
    settings = Settings(data_root=tmp_path)
    _write_history(settings, 2200)
    build_features(SYMBOL, "1d", "2018-01-01", "2024-01-10", settings=settings)
    return settings


def _run_rows(settings: Settings, model: str) -> list[dict]:
    run_dir = next(
        (settings.reports_predictions_dir / SYMBOL).glob(f"{model}_1d_*")
    )
    return [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_xgboost_trains_when_a_window_lacks_one_direction(lake) -> None:
    """XGBClassifier rejects label sets like {short, long} that skip a class.

    The failure was caught and turned into an untrained horizon, so a trending
    window silently disabled the model instead of fitting a two-class problem.
    """
    result = predict_xgboost(
        symbol=SYMBOL,
        start="2022-06-01",
        end="2022-09-30",
        settings=lake,
        interval="1d",
        train_window_days=730,
        n_estimators=20,
    )

    assert "error" not in result
    assert result["untrained_horizons"] == {"short": 0, "medium": 0, "long": 0}

    rows = _run_rows(lake, "xgboost")
    for horizon in ("short", "medium", "long"):
        payloads = [row[horizon] for row in rows]
        assert all(p["trained"] for p in payloads)
        # Probabilities stay a valid distribution after the class remapping.
        for p in payloads:
            total = p["P_long"] + p["P_flat"] + p["P_short"]
            assert total == pytest.approx(1.0, abs=1e-6)


def test_baseline_history_does_not_depend_on_the_requested_range(lake) -> None:
    """The rolling window looks back, so it must not be clipped to [start, end].

    Loading only the requested months left the first predictions with almost no
    history, which made the benchmark look worse the narrower the range.
    """
    narrow = predict_baseline(
        symbol=SYMBOL,
        start="2023-01-01",
        end="2023-03-31",
        settings=lake,
        rolling_window_days=730,
    )

    assert "error" not in narrow
    assert narrow["untrained_horizons"] == {"short": 0, "medium": 0, "long": 0}

    rows = _run_rows(lake, "baseline")
    first, last = rows[0], rows[-1]
    # Even the very first predicted bar has a full window behind it, so it is
    # not the uniform prior.
    for horizon in ("short", "medium", "long"):
        assert first[horizon]["trained"] is True
        assert first[horizon]["P_long"] != pytest.approx(1 / 3, abs=1e-9)
    assert first["date"] == "2023-01-01"
    assert last["date"] == "2023-03-31"
