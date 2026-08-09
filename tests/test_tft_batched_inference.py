"""TFT inference is batched per retraining month, not rebuilt per bar.

The old loop built a pandas frame, a TimeSeriesDataSet and a DataLoader with
batch_size=1 for every bar and every horizon. That — not the model — is why a
full run took hours.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from prosper.config import Settings
from prosper.domain import DIR_TO_IDX, DIRECTION_CLASSES, HORIZON_NAMES
from prosper.features.build import build_features
from prosper.predict.tft import (
    _logits_to_probs,
    _predict_month,
    _unwrap_prediction,
)
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import save_parquet

SYMBOL = "TESTUSDT"


# Column layout the encoder actually produces: {'nan': 0, 'long': 1, 'short': 2}.
# Which column belongs to which class is asserted in test_tft_class_order.py.
ENCODER_COLUMNS = [2, 1]


def test_logits_become_a_valid_distribution() -> None:
    probs = _logits_to_probs(np.array([0.0, 0.5, 2.0]), ENCODER_COLUMNS)

    assert len(probs) == len(DIRECTION_CLASSES)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1], "column 2 (short) carries the larger logit"


def test_the_unknown_class_is_dropped_before_softmax() -> None:
    """NaNLabelEncoder(add_nan=True) adds a class that is not a direction."""
    with_unknown = _logits_to_probs(np.array([9.0, 0.5, 2.0]), ENCODER_COLUMNS)
    without = _logits_to_probs(np.array([0.5, 2.0]), [1, 0])

    # The "unknown" logit is ignored, however large it is.
    assert with_unknown == pytest.approx(without)


def test_logits_survive_a_large_magnitude() -> None:
    """Softmax is shifted by the max, so big logits must not overflow."""
    probs = _logits_to_probs(np.array([0.0, 999.0, 1000.0]), ENCODER_COLUMNS)
    assert sum(probs) == pytest.approx(1.0)
    assert all(math.isfinite(p) for p in probs)


@pytest.mark.parametrize(
    "raw",
    [
        {"prediction": "tensor"},
        ({"prediction": "tensor"}, "index"),
    ],
)
def test_prediction_is_unwrapped_from_every_known_shape(raw) -> None:
    assert _unwrap_prediction(raw) == "tensor"


class _FakeModel:
    """Records the loaders it was asked to predict on."""

    def __init__(self, time_indices: list[int]) -> None:
        self.time_indices = time_indices
        self.calls = 0

    def predict(self, loader, mode, return_index, trainer_kwargs=None):  # noqa: ARG002
        self.calls += 1
        # `predict` spins up its own Trainer; it must be told not to log, or it
        # writes lightning_logs/ into the working directory.
        assert trainer_kwargs is not None
        assert trainer_kwargs["logger"] is False
        import pandas as pd
        import torch

        rows = len(self.time_indices)
        output = {"prediction": torch.zeros((rows, 1, len(DIRECTION_CLASSES) + 1))}
        index = pd.DataFrame({"time_idx": self.time_indices})
        return output, index


class _FakeNormalizer:
    # The layout NaNLabelEncoder(add_nan=True) produces for these two labels.
    classes_ = {"nan": 0, "long": 1, "short": 2}


class _FakeDataset:
    target_normalizer = _FakeNormalizer()

    @staticmethod
    def from_dataset(ref, frame, predict, stop_randomization):  # noqa: ARG004
        return _FakeDataset()

    def to_dataloader(self, train, batch_size, num_workers):  # noqa: ARG002
        return "loader"


def test_a_month_is_predicted_in_one_call() -> None:
    bars = list(range(100, 120))
    model = _FakeModel(bars)

    result = _predict_month(
        model,
        dataset_ref=None,
        dataset_cls=_FakeDataset,
        X_norm=np.zeros((200, 3), dtype=np.float32),
        y_dir=np.zeros(200, dtype=np.int64),
        feature_cols=["a", "b", "c"],
        symbol=SYMBOL,
        bar_indices=bars,
        seq_len=30,
        forward_steps=28,
        cutoff=100,
        trainer_kwargs={"logger": False},
    )

    assert model.calls == 1, "one batched pass, not one per bar"
    assert sorted(result) == bars
    assert all(sum(v) == pytest.approx(1.0) for v in result.values())


def test_target_history_is_masked_at_the_month_cutoff() -> None:
    """Labels needing a close at or after the cutoff must not reach the encoder."""
    captured = {}

    class CapturingDataset(_FakeDataset):
        @staticmethod
        def from_dataset(ref, frame, predict, stop_randomization):  # noqa: ARG004
            captured["frame"] = frame
            return _FakeDataset()

    bars = [100]
    _predict_month(
        _FakeModel(bars),
        dataset_ref=None,
        dataset_cls=CapturingDataset,
        X_norm=np.zeros((200, 1), dtype=np.float32),
        y_dir=np.full(200, DIR_TO_IDX["long"], dtype=np.int64),
        feature_cols=["a"],
        symbol=SYMBOL,
        bar_indices=bars,
        seq_len=30,
        forward_steps=28,
        cutoff=100,
        trainer_kwargs={"logger": False},
    )

    frame = captured["frame"]
    known = frame[frame["target"].notna()]
    masked = frame[frame["target"].isna()]

    # cutoff=100, forward_steps=28 -> labels at k >= 72 need an unseen close.
    assert known["time_idx"].max() < 72
    assert masked["time_idx"].min() == 72


def test_the_caller_dict_survives_repeated_prediction() -> None:
    """pytorch-forecasting injects its collector into the dict it is handed.

    `predict` does `trainer_kwargs.setdefault("callbacks", ... + [callback])`,
    so a dict reused across calls keeps the *first* call's callback: every
    later Trainer collects into a stale object, the result comes back empty and
    the horizon degrades to untrained without raising. Reusing one dict for
    three horizons a month is exactly the shape that triggers it.
    """
    shared = {"logger": False}
    bars = [100]

    for _ in range(3):
        result = _predict_month(
            _FakeModel(bars),
            dataset_ref=None,
            dataset_cls=_FakeDataset,
            X_norm=np.zeros((200, 1), dtype=np.float32),
            y_dir=np.zeros(200, dtype=np.int64),
            feature_cols=["a"],
            symbol=SYMBOL,
            bar_indices=bars,
            seq_len=30,
            forward_steps=28,
            cutoff=100,
            trainer_kwargs=shared,
        )
        assert sorted(result) == bars

    assert shared == {"logger": False}, "the caller's dict must not be mutated"


def _write_cyclical_history(settings: Settings, days: int = 1500) -> None:
    rng = random.Random(3)
    rows = []
    level = 100.0
    start = datetime(2019, 1, 1, tzinfo=UTC)
    for i in range(days):
        level *= 1 + rng.gauss(0, 0.015)
        price = level * (1 + 0.5 * math.sin(2 * math.pi * i / 500))
        ts = start + timedelta(days=i)
        rows.append(
            {
                "open_time": ts,
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


@pytest.mark.slow
def test_tft_end_to_end_produces_trained_predictions(tmp_path) -> None:
    """A real (tiny) TFT run: every horizon trains and the output varies."""
    pytest.importorskip("pytorch_forecasting")
    from prosper.predict.tft import predict_tft

    settings = Settings(data_root=tmp_path)
    _write_cyclical_history(settings)
    build_features(SYMBOL, "1d", "2019-01-01", "2023-02-01", settings=settings)

    result = predict_tft(
        symbol=SYMBOL,
        start="2022-06-01",
        end="2022-06-20",
        settings=settings,
        interval="1d",
        seq_len=30,
        train_window_days=730,
        max_epochs=1,
        hidden_size=8,
    )

    assert "error" not in result
    assert result["untrained_horizons"] == {name: 0 for name in HORIZON_NAMES}
    assert result["predictions"] == 20
