"""One label definition, used by every model and by the label file.

These tests pin the shared function to a per-bar reference loop and hold the
label builder to the models' horizons, so the label file and the models cannot
disagree - a builder shifting by rows would make "one day" one hour on hourly
data.
"""

from __future__ import annotations

import inspect
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from prosper.config import Settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    DEFAULT_TRADED_HORIZON,
    DIR_TO_IDX,
    HORIZON_NAMES,
    LEGACY_DEPTH_BINS_STR,
    assign_depth_bin,
    direction_from_return,
    parse_depth_bins,
)
from prosper.inventory import DataManager
from prosper.labels.build import build_labels, label_columns
from prosper.labels.forward import UNKNOWN, forward_labels
from prosper.storage.layout import get_labels_parquet_path, get_parquet_file_path
from prosper.storage.parquet import save_parquet

SYMBOL = "TESTUSDT"


def _reference(closes, steps, bins):
    """A per-bar reference loop: the arithmetic the vectorised function must match."""
    n = len(closes)
    direction, depth = {}, {}
    for name, forward in steps.items():
        dir_arr = np.full(n, -1, dtype=np.int64)
        depth_arr = np.full(n, -1, dtype=np.int64)
        for i in range(n - forward):
            j = i + forward
            if closes[i] and closes[j]:
                r = closes[j] / closes[i] - 1.0
                side = direction_from_return(r)
                if side is not None:
                    dir_arr[i] = DIR_TO_IDX[side]
                    bin_idx = assign_depth_bin(abs(r) * 100.0, bins)
                    if bin_idx is not None:
                        depth_arr[i] = bin_idx
        direction[name], depth[name] = dir_arr, depth_arr
    return direction, depth


@pytest.mark.parametrize("scheme", [DEFAULT_DEPTH_BINS_STR, LEGACY_DEPTH_BINS_STR])
def test_matches_the_loop_it_replaced(scheme) -> None:
    rng = np.random.default_rng(0)
    closes = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, 600))))
    # Every awkward case the loop had an opinion on: a missing price, a zero
    # price, an unchanged price (a return with no side), and a NaN.
    closes[50] = None
    closes[80] = 0.0
    closes[121] = closes[120]
    closes[200] = math.nan
    bins, _ = parse_depth_bins(scheme)
    steps = {"a": 1, "b": 7, "c": 28, "d": 364, "longer_than_series": 700}

    expected_dir, expected_depth = _reference(closes, steps, bins)
    labels = forward_labels(closes, steps, bins)

    for name in steps:
        assert labels.direction[name].dtype == np.int64
        assert np.array_equal(labels.direction[name], expected_dir[name]), name
        assert np.array_equal(labels.depth[name], expected_depth[name]), name
    assert (labels.direction["longer_than_series"] == UNKNOWN).all()


def test_nan_has_no_side() -> None:
    """NaN fails both `== 0` and `> 0`, so unchecked it falls through to `short`."""
    assert direction_from_return(math.nan) is None
    assert direction_from_return(0.0) is None
    assert direction_from_return(0.01) == "long"
    assert direction_from_return(-0.01) == "short"


def test_no_predictor_carries_its_own_copy() -> None:
    from prosper.predict import baseline, gru, pool, tft

    for module in (baseline, gru, pool, tft):
        source = inspect.getsource(module)
        assert "forward_labels(" in source, module.__name__
        assert "direction_from_return(" not in source, module.__name__


# ── the label file ───────────────────────────────────────────────────────────

def _write_klines(settings: Settings, interval: str, bars: int, step: timedelta) -> None:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(1)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, bars)))
    frame = pl.DataFrame(
        {"open_time": [start + step * i for i in range(bars)], "close": close,
         "open": close, "high": close, "low": close, "volume": np.ones(bars)}
    ).with_columns(pl.col("open_time").dt.year().alias("_y"), pl.col("open_time").dt.month().alias("_m"))
    for (year, month), group in frame.group_by(["_y", "_m"]):
        save_parquet(
            group.drop("_y", "_m").sort("open_time"),
            get_parquet_file_path(SYMBOL, interval, int(year), month=int(month), settings=settings),
        )


def test_the_file_holds_every_horizon_the_models_train(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    _write_klines(settings, "1d", 800, timedelta(days=1))

    stats = build_labels(SYMBOL, "1d", settings=settings)
    frame = pl.read_parquet(get_labels_parquet_path(SYMBOL, "1d", settings=settings))

    assert set(stats["horizons"]) == set(HORIZON_NAMES)
    for horizon in HORIZON_NAMES:
        return_col, direction_col, depth_col = label_columns(horizon)
        assert {return_col, direction_col, depth_col} <= set(frame.columns)
        # Depth is stored by name, so the file says which scheme it used.
        assert set(frame[depth_col].drop_nulls()) <= set(parse_depth_bins(DEFAULT_DEPTH_BINS_STR)[1])


def test_a_horizon_is_the_same_span_at_every_interval(tmp_path) -> None:
    """At 1h the week label looks 168 bars ahead, not 7."""
    settings = Settings(data_root=tmp_path)
    _write_klines(settings, "1h", 24 * 40, timedelta(hours=1))

    stats = build_labels(SYMBOL, "1h", settings=settings)
    frame = pl.read_parquet(get_labels_parquet_path(SYMBOL, "1h", settings=settings)).sort("open_time")

    week = next(h for h in DEFAULT_HORIZONS if h.name == "week")
    assert stats["horizons"]["week"]["forward_bars"] == week.steps("1h") == 168
    close = frame["close"].to_numpy()
    returns = frame[label_columns("week")[0]].to_numpy()
    assert returns[0] == pytest.approx(close[168] / close[0] - 1.0)
    assert np.isnan(returns[-1]) or returns[-1] is None


# ── what the dashboard reads ─────────────────────────────────────────────────

def test_the_chart_shows_the_traded_horizon(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    _write_klines(settings, "1d", 400, timedelta(days=1))
    build_labels(SYMBOL, "1d", settings=settings)
    stored = pl.read_parquet(get_labels_parquet_path(SYMBOL, "1d", settings=settings))

    manager = DataManager(settings=settings)
    shown = manager._collect_labels(
        SYMBOL, "1d", stored["open_time"].min(), stored["open_time"].max()
    )

    direction_col = label_columns(DEFAULT_TRADED_HORIZON)[1]
    assert shown.columns == ["open_time", "return_fwd", "direction", "depth_bin"]
    assert shown["direction"].to_list() == stored.sort("open_time")[direction_col].to_list()


def test_a_one_bar_label_file_counts_as_outdated(tmp_path) -> None:
    """A one-bar label file is real parquet with the wrong content."""
    settings = Settings(data_root=tmp_path)
    manager = DataManager(settings=settings)
    path = get_labels_parquet_path(SYMBOL, "1d", settings=settings)
    path.parent.mkdir(parents=True)
    pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "return_fwd": [0.01],
                  "direction": ["long"], "depth_bin": [0]}).write_parquet(path)

    assert not manager._has_current_labels(path.parent)

    _write_klines(settings, "1d", 400, timedelta(days=1))
    build_labels(SYMBOL, "1d", settings=settings)
    assert manager._has_current_labels(path.parent)
