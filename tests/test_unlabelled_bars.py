"""A bar without a label still belongs to the series (SPECIFICATION 2.2).

A forward return of exactly zero has no side, so its bar has no label. The
sequence models keep such a bar as input and leave it out only as a target:
dropped from the frame, it leaves a hole TimeSeriesDataSet refuses (TFT); kept
as a target, CrossEntropyLoss rejects its -1 (GRU). Six of the nine study
symbols close a horizon flat somewhere, while BTCUSDT, ETHUSDT and SOLUSDT never
do - so a test on those three alone cannot see either failure.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from prosper.config import Settings
from prosper.domain import HORIZON_NAMES
from prosper.features.build import build_features
from prosper.labels.forward import forward_returns
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import save_parquet

SYMBOL = "TESTUSDT"
FLAT = 882  # 2021-06-01, inside the training windows of June 2022


def _write_history_with_a_flat_week(settings: Settings, days: int = 1500) -> list[float]:
    """A noisy cycle whose bar FLAT has a 7-day return of exactly zero."""
    rng = random.Random(3)
    rows, closes = [], []
    level = 100.0
    start = datetime(2019, 1, 1, tzinfo=UTC)
    for i in range(days):
        level *= 1 + rng.gauss(0, 0.015)
        price = level * (1 + 0.5 * math.sin(2 * math.pi * i / 500))
        if i == FLAT + 7:
            price = closes[FLAT]
        closes.append(price)
        rows.append(
            {
                "open_time": start + timedelta(days=i),
                "open": price * 0.99,
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
    return closes


def _run_tft(settings: Settings) -> dict:
    pytest.importorskip("pytorch_forecasting")
    from prosper.predict.tft import predict_tft

    return predict_tft(
        symbol=SYMBOL, start="2022-06-01", end="2022-06-20", settings=settings, interval="1d",
        seq_len=30, train_window_days=730, max_epochs=1, hidden_size=8,
    )


def _run_gru(settings: Settings) -> dict:
    from prosper.predict.gru import predict_gru

    return predict_gru(
        symbol=SYMBOL, start="2022-06-01", end="2022-06-20", settings=settings, interval="1d",
        seq_len=30, train_window_days=730, epochs=1, hidden_size=8,
    )


@pytest.mark.slow
@pytest.mark.parametrize("run", [_run_tft, _run_gru], ids=["tft", "gru"])
def test_a_bar_without_a_label_does_not_stop_training(tmp_path, run) -> None:
    settings = Settings(data_root=tmp_path)
    closes = _write_history_with_a_flat_week(settings)
    build_features(SYMBOL, "1d", "2019-01-01", "2023-02-01", settings=settings)
    assert forward_returns(closes, 7)[FLAT] == 0.0, "the planted bar must have no side"

    result = run(settings)

    assert "error" not in result
    assert result["untrained_horizons"] == {name: 0 for name in HORIZON_NAMES}
