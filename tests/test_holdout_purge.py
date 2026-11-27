"""The calibration slice must be out of sample for the model it calibrates.

A label is the return from a bar's close to the close F bars later
(`labels/forward.py`), so neighbouring labels share most of their price path.
Every model holds out the newest quarter of its training window to fit the
calibrator, and the networks also stop training on it. With the fitted rows
running right up to that slice, the last F fitted labels were computed from
closes inside it: the model was scored there on outcomes it had been trained
on, looked more skilful than on any later bar, and was calibrated accordingly.
At the yearly horizon the slice is shorter than the horizon, so every held-out
label overlapped a fitted one.

Each test records which rows a real run fits and which it holds out, and checks
that no close is shared: the newest fitted bar lies more than F bars before the
oldest held-out one - the separation `training_bounds` keeps from the bar being
predicted.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from prosper.config import Settings
from prosper.domain import DEFAULT_HORIZONS
from prosper.features.build import build_features
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import save_parquet

SYMBOL = "TESTUSDT"
FORWARD = {h.name: h.steps("1d") for h in DEFAULT_HORIZONS}
DAY = np.timedelta64(1, "D")


def _write_history(settings: Settings, days: int) -> None:
    """A slow cycle plus noise, so every horizon sees both directions."""
    rng = random.Random(11)
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


def _assert_out_of_sample(gaps: list[tuple[int, int]]) -> None:
    """Each pair is (forward steps, bars from the newest fitted to the oldest held-out row)."""
    assert gaps, "no window held anything out, so nothing was checked"
    for forward, gap in gaps:
        assert gap > forward, (
            f"the newest fitted bar is {gap} bars before the first held-out one, so at "
            f"a {forward}-bar horizon their labels share {forward - gap + 1} closes"
        )


# ── the split itself ─────────────────────────────────────────────────────────

def test_the_fitted_rows_end_a_horizon_before_the_held_out_slice() -> None:
    from prosper.predict.calibration import CALIBRATION_FRACTION, purged_holdout

    bars = list(range(1000))
    fit_end, cal_start = purged_holdout(bars, 91)

    assert cal_start == 1000 - int(1000 * CALIBRATION_FRACTION)
    assert bars[cal_start] - bars[fit_end - 1] == 92, "91 bars belong to neither part"


def test_a_pooled_instant_is_never_on_both_sides() -> None:
    """Three symbols per day: a quarter of the rows can start mid-instant, and
    rows from one day are the same market. A horizon is a span of days there,
    not a count of rows."""
    from prosper.predict.calibration import purged_holdout

    # 402 days: the newest quarter of the 1206 rows begins on the last row of a day.
    days = np.datetime64("2021-01-01", "D") + np.arange(402)
    times = np.repeat(days, 3).astype("datetime64[us]")
    fit_end, cal_start = purged_holdout(times, np.timedelta64(28, "D"))

    assert times[cal_start - 1] < times[cal_start], "the slice starts with a whole day"
    assert (times[cal_start] - times[fit_end - 1]) / DAY == 29
    assert times[fit_end] > times[fit_end - 1], "the fitted rows end with a whole day"


def test_a_gap_that_leaves_nothing_to_fit_holds_nothing_out() -> None:
    """The yearly horizon in a window a few months long: training on everything
    without a calibrator beats a calibrator with no model behind it."""
    from prosper.predict.calibration import purged_holdout

    assert purged_holdout(list(range(300)), 364) == (300, 300)


def test_a_rescued_split_is_judged_on_the_rows_actually_fitted() -> None:
    """Without the gap the oldest 100 rows look fittable; with it, only those
    up to 70 are fitted, and they hold one class."""
    from prosper.predict.calibration import usable_split

    labels = [0] * 70 + [1] * 30 + [0, 1] * 50
    assert usable_split(labels, 100, min_holdout=20) == 100
    split = usable_split(labels, 100, min_holdout=20, times=range(len(labels)), purge=30)

    assert split is not None and split > 100


# ── pooled tree models ───────────────────────────────────────────────────────

@dataclass
class _Window:
    """One trained window of a tabular model: its rows and how they were divided."""

    horizon: str
    times: np.ndarray  # open_time of each labelled training row, oldest first
    fitted: int | None = None  # leading rows the direction model was fitted on
    held_out: int = 0  # trailing rows the calibrator was fitted on


def _spy_on_tabular(monkeypatch, forecaster, module, classifier) -> list[_Window]:
    windows: list[_Window] = []
    fit_horizon, fit = forecaster.fit_horizon, classifier.fit
    calibrate = module.fit_calibrator_from_model

    def recording_fit_horizon(self, horizon, window, pooled, valid):
        windows.append(_Window(horizon, np.asarray(pooled.open_time)[valid]))
        return fit_horizon(self, horizon, window, pooled, valid)

    def recording_fit(self, X, y, *args, **kwargs):
        # The direction model is fitted first; depth models, if any, follow it.
        if windows and windows[-1].fitted is None:
            windows[-1].fitted = len(X)
        return fit(self, X, y, *args, **kwargs)

    def recording_calibrate(clf, X, y, **kwargs):
        windows[-1].held_out = len(X)
        return calibrate(clf, X, y, **kwargs)

    monkeypatch.setattr(forecaster, "fit_horizon", recording_fit_horizon)
    monkeypatch.setattr(classifier, "fit", recording_fit)
    monkeypatch.setattr(module, "fit_calibrator_from_model", recording_calibrate)
    return windows


def _tabular_gaps(windows: list[_Window]) -> list[tuple[int, int]]:
    gaps = []
    for window in windows:
        if not window.held_out or window.fitted is None:
            continue
        newest_fitted = window.times[window.fitted - 1]
        oldest_held_out = window.times[len(window.times) - window.held_out]
        gaps.append((FORWARD[window.horizon], int((oldest_held_out - newest_fitted) / DAY)))
    return gaps


def test_xgboost_calibrates_on_labels_it_was_not_fitted_on(lake, monkeypatch) -> None:
    import xgboost

    from prosper.predict import xgboost_model

    windows = _spy_on_tabular(
        monkeypatch, xgboost_model.XGBoostForecaster, xgboost_model, xgboost.XGBClassifier
    )
    xgboost_model.predict_xgboost(
        symbol=SYMBOL, start="2022-06-01", end="2022-07-31", settings=lake, interval="1d"
    )

    _assert_out_of_sample(_tabular_gaps(windows))


def test_gradient_boosting_calibrates_on_labels_it_was_not_fitted_on(lake, monkeypatch) -> None:
    from prosper.predict import ml

    windows = _spy_on_tabular(
        monkeypatch, ml.MLForecaster, ml, ml.HistGradientBoostingClassifier
    )
    ml.predict_ml(symbol=SYMBOL, start="2022-06-01", end="2022-07-31", settings=lake, interval="1d")

    _assert_out_of_sample(_tabular_gaps(windows))


# ── sequence models ──────────────────────────────────────────────────────────

@pytest.mark.slow
def test_gru_stops_and_calibrates_on_labels_it_was_not_fitted_on(lake, monkeypatch) -> None:
    from torch.utils.data import Subset

    from prosper.predict import gru

    current: dict[str, str] = {}
    gaps: list[tuple[int, int]] = []
    reseed, train = gru.GRUForecaster.reseed, gru._train_gru

    # `reseed` learns the window's horizon just before training; `_train_gru`
    # is never told it.
    def remember_horizon(self, horizon, month):
        current["horizon"] = horizon
        reseed(self, horizon, month)

    def recording_train(model, loader, *args, dataset, holdout, **kwargs):
        fitted = loader.dataset
        if holdout and fitted is not dataset:
            # Positions in `dataset` are sequences; a sequence's index is its
            # target bar less a constant, so differences are differences in bars.
            def sequence(position: int) -> int:
                return dataset.indices[position] if isinstance(dataset, Subset) else position

            gaps.append(
                (
                    FORWARD[current["horizon"]],
                    sequence(min(holdout)) - sequence(max(fitted.indices)),
                )
            )
        return train(model, loader, *args, dataset=dataset, holdout=holdout, **kwargs)

    monkeypatch.setattr(gru.GRUForecaster, "reseed", remember_horizon)
    monkeypatch.setattr(gru, "_train_gru", recording_train)
    gru.predict_gru(
        symbol=SYMBOL, start="2022-06-01", end="2022-06-30", settings=lake,
        interval="1d", epochs=1, hidden_size=8,
    )

    _assert_out_of_sample(gaps)


@pytest.mark.slow
def test_tft_stops_and_calibrates_on_labels_it_was_not_fitted_on(lake, monkeypatch) -> None:
    from prosper.predict import tft

    gaps: list[tuple[int, int]] = []
    calibrate = tft._fit_tft_calibrator

    def recording_calibrate(model, ds, *args, bar_indices, forward_steps, **kwargs):
        if bar_indices:
            newest_fitted = int(ds.decoded_index["time_idx_first_prediction"].max())
            gaps.append((forward_steps, min(bar_indices) - newest_fitted))
        return calibrate(
            model, ds, *args, bar_indices=bar_indices, forward_steps=forward_steps, **kwargs
        )

    monkeypatch.setattr(tft, "_fit_tft_calibrator", recording_calibrate)
    tft.predict_tft(
        symbol=SYMBOL, start="2022-06-01", end="2022-06-20", settings=lake, interval="1d",
        seq_len=30, max_epochs=1, hidden_size=8,
    )

    _assert_out_of_sample(gaps)


@pytest.mark.slow
def test_tft_stops_on_the_calibration_slice_alone(lake, monkeypatch) -> None:
    """The stopping rule scores the slice the calibrator is fitted on. Its frame
    reaches back `seq_len` bars so that the oldest held-out bar has a full
    encoder history; those older bars are fitted or purged ones, so they may be
    inputs but never targets."""
    from prosper.predict import tft

    windows: list[dict[str, set[int]]] = []
    reseed = tft.TFTForecaster.reseed
    callback, calibrate = tft._calibrated_val_loss_callback, tft._fit_tft_calibrator

    def new_window(self, horizon, month):
        windows.append({})
        reseed(self, horizon, month)

    def recording_callback(val_dataset):
        targets = val_dataset.decoded_index["time_idx_first_prediction"]
        windows[-1]["stopped"] = {int(k) for k in targets}
        return callback(val_dataset)

    def recording_calibrate(model, ds, *args, bar_indices, **kwargs):
        windows[-1]["calibrated"] = set(bar_indices)
        return calibrate(model, ds, *args, bar_indices=bar_indices, **kwargs)

    monkeypatch.setattr(tft.TFTForecaster, "reseed", new_window)
    monkeypatch.setattr(tft, "_calibrated_val_loss_callback", recording_callback)
    monkeypatch.setattr(tft, "_fit_tft_calibrator", recording_calibrate)
    tft.predict_tft(
        symbol=SYMBOL, start="2022-06-01", end="2022-06-20", settings=lake, interval="1d",
        seq_len=30, max_epochs=1, hidden_size=8,
    )

    stopped = [window for window in windows if "stopped" in window]
    assert stopped, "no window stopped early, so nothing was checked"
    for window in stopped:
        extra = sorted(window["stopped"] - window["calibrated"])
        assert not extra, f"stopped on {len(extra)} bars outside the slice, from {extra[0]}"
        assert window["stopped"] == window["calibrated"]


# ── one holdout policy for every model ───────────────────────────────────────

def _write_trend_then_cycle(settings: Settings) -> None:
    """A steady rise for 1300 days, then a 60-day cycle around where it ended.

    Every label older than about day 1240 is a rise, so in January 2022 the
    default fitted rows of the week and month windows hold one class while
    their newest rows hold both: the split has to move for the window to be
    calibrated at all. The quarter's first fall comes too late for any split
    to leave the calibrator its 40 rows, so that window rightly holds nothing out.
    """
    start = datetime(2018, 1, 1, tzinfo=UTC)
    top = 100.0 * 1.002**1300
    rows = []
    for i in range(1600):
        if i < 1300:
            price = 100.0 * 1.002**i
        else:
            price = top * (1.0 + 0.3 * math.sin(2 * math.pi * (i - 1300) / 60.0))
        rows.append(
            {
                "open_time": start + timedelta(days=i),
                "open": price * 0.999,
                "high": price * 1.01,
                "low": price * 0.99,
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


def test_tree_models_move_a_split_that_leaves_one_class_to_fit(tmp_path) -> None:
    """The networks already looked for a nearby usable split; the tree models
    trained on every row and dropped the calibrator, so on the same window one
    architecture was calibrated and another was not."""
    import json

    from prosper.domain import HORIZON_NAMES
    from prosper.predict.xgboost_model import predict_xgboost

    settings = Settings(data_root=tmp_path)
    _write_trend_then_cycle(settings)
    build_features(SYMBOL, "1d", "2018-01-01", "2022-05-01", settings=settings)
    predict_xgboost(
        symbol=SYMBOL, start="2022-01-01", end="2022-01-31", settings=settings, interval="1d"
    )

    run_dir = next((settings.reports_predictions_dir / SYMBOL).glob("xgboost_1d_*"))
    rows = [json.loads(line) for line in
            (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    for horizon in ("week", "month"):
        assert horizon in HORIZON_NAMES
        payloads = [row[horizon] for row in rows if row[horizon]["trained"]]
        assert payloads, f"{horizon} did not train"
        assert all(p["calibrated_on"] > 0 for p in payloads), f"{horizon} lost its calibrator"


def test_the_policy_keeps_the_default_split_when_it_is_usable() -> None:
    from prosper.predict.calibration import choose_holdout, purged_holdout

    labels = [0, 1] * 500
    assert choose_holdout(range(1000), labels, 28) == purged_holdout(range(1000), 28)


def test_the_policy_moves_a_split_whose_slice_holds_one_class() -> None:
    """Stopping and calibrating on one class rewards certainty (`usable_split`)."""
    from prosper.predict.calibration import choose_holdout

    labels = [0, 1] * 400 + [1] * 200
    fit_end, cal_start = choose_holdout(range(1000), labels, 28)

    assert len(set(labels[cal_start:])) == 2
    assert len(set(labels[:fit_end])) == 2
    assert cal_start - fit_end == 28


def test_the_policy_holds_nothing_out_when_no_split_works() -> None:
    from prosper.predict.calibration import choose_holdout

    assert choose_holdout(range(1000), [0] * 990 + [1] * 10, 28) == (1000, 1000)


def test_a_moved_pooled_split_still_starts_with_a_whole_instant() -> None:
    from prosper.predict.calibration import choose_holdout

    days = np.datetime64("2021-01-01", "D") + np.arange(400)
    times = np.repeat(days, 3).astype("datetime64[us]")
    labels = [0, 1] * 450 + [1] * 300
    fit_end, cal_start = choose_holdout(times, labels, np.timedelta64(28, "D"))

    assert cal_start < len(times) and times[cal_start - 1] < times[cal_start]
    assert len(set(labels[cal_start:])) == 2


# ── a window that holds nothing out ──────────────────────────────────────────

def test_a_window_without_a_slice_takes_the_horizons_latest_temperature() -> None:
    """The raw models are near-certain, so an uncalibrated window was the most
    confident of its run for no better reason than missing data. The latest
    temperature of the same horizon was chosen on a slice realised before it;
    before the first, a forecast is given no confidence beyond chance."""
    from prosper.predict.calibration import (
        TEMPERATURE_GRID,
        TemperatureCalibrator,
        TemperatureMemory,
    )

    memory = TemperatureMemory()
    first = memory.settle("year", TemperatureCalibrator())
    assert (first.temperature, first.source) == (TEMPERATURE_GRID[-1], "prior")

    fitted = TemperatureCalibrator(50.0, fitted_on=120, source="fitted")
    assert memory.settle("year", fitted) is fitted
    later = memory.settle("year", TemperatureCalibrator())
    assert (later.temperature, later.source, later.fitted_on) == (50.0, "carried", 0)
    other = memory.settle("week", TemperatureCalibrator())
    assert other.source == "prior", "each horizon keeps its own"


def test_a_run_never_leaves_a_trained_window_uncalibrated(tmp_path) -> None:
    import json

    from prosper.predict.xgboost_model import predict_xgboost

    settings = Settings(data_root=tmp_path)
    _write_trend_then_cycle(settings)
    build_features(SYMBOL, "1d", "2018-01-01", "2022-05-01", settings=settings)
    predict_xgboost(
        symbol=SYMBOL, start="2022-01-01", end="2022-01-31", settings=settings, interval="1d"
    )

    run_dir = next((settings.reports_predictions_dir / SYMBOL).glob("xgboost_1d_*"))
    rows = [json.loads(line) for line in
            (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    quarter = [row["quarter"] for row in rows if row["quarter"]["trained"]]
    assert quarter and all(p["calibrated_on"] == 0 for p in quarter), "the case under test"
    assert all(p["temperature_source"] == "prior" for p in quarter)
    assert all(max(p["P_long"], p["P_short"]) < 0.51 for p in quarter)
