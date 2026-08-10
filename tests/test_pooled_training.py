"""Pooling symbols must add observations without adding leakage.

Two failures would be invisible in the output and fatal to the result, so they
are pinned here rather than argued about.

The first is **look-ahead across symbols**. Every predictor used to slice its
training set by row index, which is exact inside one sorted frame and meaningless
across several: row 400 of BTCUSDT is September 2018 and row 400 of SOLUSDT is
September 2021. An index-based pooled window would have trained a 2021 forecast
on bars that had not happened yet, and nothing downstream could tell — accuracy
would simply have improved.

The second is **silent divergence**. The pooled path is the only path now, taken
even for a single symbol, precisely so there is one implementation of "which rows
may this model see". This file pins that it agrees with the index arithmetic it
replaced; `test_one_grader.py` exists because three copies of a rule drifted once
already.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    effective_symbols,
    parse_depth_bins,
)
from prosper.predict.pool import (
    TrainingPool,
    build_frame,
    describe_pool,
    parse_symbols,
)
from prosper.predict.window import resolve_train_windows, training_bounds

INTERVAL = "1d"
DEPTH_BINS, _ = parse_depth_bins(DEFAULT_DEPTH_BINS_STR)
FEATURES = ["a", "b"]


def _frame(symbol: str, start: datetime, bars: int, horizons: dict[str, int], seed: int = 0):
    """A synthetic symbol with its own launch date and its own price path."""
    rng = np.random.default_rng(seed)
    times = [start + timedelta(days=i) for i in range(bars)]
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, size=bars))
    df = pl.DataFrame(
        {
            "open_time": times,
            "close": close,
            "a": rng.normal(size=bars),
            "b": rng.normal(size=bars),
        }
    )
    return build_frame(symbol, df, FEATURES, horizons, DEPTH_BINS)


def _steps() -> dict[str, int]:
    return {h.name: h.steps(INTERVAL) for h in DEFAULT_HORIZONS}


def _windows(history_bars: int):
    return resolve_train_windows(None, INTERVAL, DEFAULT_HORIZONS, history_bars=history_bars)


# ── The identity that keeps one code path honest ─────────────────────────────

def test_one_symbol_pooled_matches_the_index_arithmetic_it_replaced() -> None:
    """The whole reason the single-symbol case goes through the pool as well.

    On a contiguous series the two must select exactly the same rows. Where they
    could differ — a gap in the data — the timestamp version is the correct one,
    because it asks "had this outcome happened yet" rather than "how many rows
    back is it".
    """
    steps = _steps()
    frame = _frame("BTCUSDT", datetime(2018, 1, 1, tzinfo=UTC), 1600, steps, seed=1)
    pool = TrainingPool([frame], INTERVAL)
    windows = _windows(len(frame))

    for horizon in ("week", "month", "quarter"):
        for row in (900, 1200, 1599):
            start, end = training_bounds(
                row, windows.window_steps[horizon], windows.forward_steps[horizon]
            )
            pooled = pool.training_slice(frame.timestamps[row], horizon, windows)

            assert len(pooled) == end - start, (horizon, row)
            assert np.array_equal(pooled.features, frame.features[start:end])
            assert np.array_equal(pooled.direction, frame.direction[horizon][start:end])


# ── Leakage ──────────────────────────────────────────────────────────────────

def test_no_pooled_row_reaches_the_cutoff() -> None:
    """The bound is an instant, so this holds for every symbol at once."""
    steps = _steps()
    pool = TrainingPool(
        [
            _frame("BTCUSDT", datetime(2017, 8, 17, tzinfo=UTC), 3000, steps, seed=2),
            _frame("SOLUSDT", datetime(2020, 8, 11, tzinfo=UTC), 1900, steps, seed=3),
        ],
        INTERVAL,
    )
    windows = _windows(3000)
    cutoff = datetime(2024, 6, 1, tzinfo=UTC)

    for horizon in windows.forward_steps:
        pooled = pool.training_slice(cutoff, horizon, windows)
        if not len(pooled):
            continue
        forward = timedelta(days=windows.forward_steps[horizon])
        latest = pooled.open_time.max().astype("datetime64[us]").astype(datetime)
        assert latest + forward < cutoff.replace(tzinfo=None), horizon


def test_a_symbol_that_had_not_launched_contributes_nothing() -> None:
    """The trap an index-based window walks into: SOLUSDT's row 400 is 2021, so
    an index window centred on 2018 would have pulled future bars into the past.
    """
    steps = _steps()
    btc = _frame("BTCUSDT", datetime(2017, 8, 17, tzinfo=UTC), 3000, steps, seed=4)
    sol = _frame("SOLUSDT", datetime(2020, 8, 11, tzinfo=UTC), 1900, steps, seed=5)
    pool = TrainingPool([btc, sol], INTERVAL)
    windows = _windows(3000)

    pooled = pool.training_slice(datetime(2019, 1, 1, tzinfo=UTC), "week", windows)

    assert len(pooled) > 0, "BTCUSDT existed and must have contributed"
    assert set(pooled.symbol_counts) == {"BTCUSDT"}, pooled.symbol_counts


def test_labels_stay_inside_their_own_symbol() -> None:
    """A BTCUSDT label is the sign of a BTCUSDT return. Pooling shares features
    and outcomes, never price paths: one symbol's close deciding another's label
    would be a leak no metric could see."""
    steps = {"week": 7}
    frame = _frame("BTCUSDT", datetime(2020, 1, 1, tzinfo=UTC), 200, steps, seed=6)

    for i in range(200 - 7):
        expected = frame.close[i + 7] / frame.close[i] - 1.0
        if expected == 0:
            continue
        assert frame.direction["week"][i] == (1 if expected > 0 else 0)


# ── Ordering, because the calibrator depends on it ───────────────────────────

def test_the_pooled_slice_is_chronological_not_stacked_by_symbol() -> None:
    """`calibration_split` takes the newest rows of the training region. Stacking
    symbols end to end would hand it one symbol's recent history and call that a
    held-out slice."""
    steps = _steps()
    pool = TrainingPool(
        [
            _frame("BTCUSDT", datetime(2019, 1, 1, tzinfo=UTC), 1500, steps, seed=7),
            _frame("ETHUSDT", datetime(2019, 1, 1, tzinfo=UTC), 1500, steps, seed=8),
        ],
        INTERVAL,
    )
    windows = _windows(1500)
    pooled = pool.training_slice(datetime(2022, 6, 1, tzinfo=UTC), "month", windows)

    times = pooled.open_time
    assert np.all(times[:-1] <= times[1:]), "the slice must be sorted by instant"
    # Both symbols must appear throughout, not one after the other.
    newest_tenth = pooled.symbol[-len(pooled) // 10 :]
    assert len(set(newest_tenth)) == 2, "the held-out tail must span the pool"


def test_the_slice_is_reproducible_when_symbols_share_a_timestamp() -> None:
    """Ties are broken stably, or two runs of one configuration could split the
    calibration set differently and agree only by luck."""
    steps = _steps()
    frames = [
        _frame("BTCUSDT", datetime(2019, 1, 1, tzinfo=UTC), 1200, steps, seed=9),
        _frame("ETHUSDT", datetime(2019, 1, 1, tzinfo=UTC), 1200, steps, seed=10),
    ]
    windows = _windows(1200)
    cutoff = datetime(2021, 9, 1, tzinfo=UTC)

    first = TrainingPool(frames, INTERVAL).training_slice(cutoff, "week", windows)
    second = TrainingPool(frames, INTERVAL).training_slice(cutoff, "week", windows)

    assert np.array_equal(first.features, second.features)
    assert list(first.symbol) == list(second.symbol)


# ── Timestamps are the prediction key ───────────────────────────────────────

def test_the_frame_keeps_the_zone_the_parquet_had() -> None:
    """Found by a byte comparison, not by reasoning: routing timestamps through
    `datetime64` dropped `+00:00` from every `open_time`, and predictions are
    keyed by `open_time` (invariant 4). The run still looked perfectly healthy —
    it simply no longer joined against anything written before it.
    """
    frame = _frame("BTCUSDT", datetime(2020, 1, 1, tzinfo=UTC), 50, {"week": 7})

    assert frame.timestamps[0].tzinfo is not None
    assert frame.timestamps[0].isoformat().endswith("+00:00")
    # The slicing copy is deliberately naive; it must never be written out.
    assert frame.open_time.dtype == np.dtype("datetime64[us]")


# ── What pooling is actually worth ──────────────────────────────────────────

def test_three_correlated_symbols_are_not_three_observations() -> None:
    """The measurement that decides whether pooling helps at all. Mean pairwise
    correlation of forward returns on this project's data is 0.60 at the weekly
    horizon and 0.76 at the annual one."""
    assert effective_symbols(3, 0.0) == pytest.approx(3.0)
    assert effective_symbols(3, 0.604) == pytest.approx(1.36, abs=0.01)
    assert effective_symbols(3, 0.761) == pytest.approx(1.19, abs=0.01)
    assert effective_symbols(1, 0.9) == pytest.approx(1.0)


def test_pooling_more_symbols_cannot_beat_one_over_the_correlation() -> None:
    """Why the annual horizon is not a pooling problem.

    `n / (1 + (n - 1) * rho)` tends to `1 / rho`, so at the measured 0.76 even a
    hundred crypto symbols are worth about 1.3 — they share one dominant market
    factor and pooling averages the part that differs, not the part that
    dominates. Predicting *relative* moves is the lever that removes it.
    """
    rho = 0.761
    ceiling = 1.0 / rho

    assert effective_symbols(100, rho) < ceiling
    assert effective_symbols(1000, rho) == pytest.approx(ceiling, abs=0.01)
    assert effective_symbols(20, rho) < 1.35


def test_the_correlation_is_measured_from_the_pool_itself() -> None:
    """Two symbols driven by the same series must read as correlated; two
    independent ones must not."""
    steps = {"week": 7}
    base = _frame("BTCUSDT", datetime(2019, 1, 1, tzinfo=UTC), 900, steps, seed=11)
    other = _frame("ETHUSDT", datetime(2019, 1, 1, tzinfo=UTC), 900, steps, seed=12)

    independent = TrainingPool([base, other], INTERVAL).return_correlation(7)
    assert abs(independent) < 0.2, f"unrelated paths correlated at {independent}"

    twin = TrainingPool([base, base], INTERVAL).return_correlation(7)
    assert twin == pytest.approx(1.0, abs=1e-6), "a symbol against itself is 1.0"

    assert TrainingPool([base], INTERVAL).return_correlation(7) == 0.0


def test_a_summary_reports_effective_symbols_beside_the_row_count() -> None:
    """Rows flatter pooling threefold. A summary carrying only rows would support
    a claim the data does not."""
    steps = _steps()
    pool = TrainingPool(
        [
            _frame("BTCUSDT", datetime(2019, 1, 1, tzinfo=UTC), 1400, steps, seed=13),
            _frame("ETHUSDT", datetime(2019, 1, 1, tzinfo=UTC), 1400, steps, seed=14),
        ],
        INTERVAL,
    )
    windows = _windows(1400)
    slices = {
        name: pool.training_slice(datetime(2022, 1, 1, tzinfo=UTC), name, windows)
        for name in ("week", "month")
    }

    described = describe_pool(slices, pool, windows.forward_steps)

    for name in ("week", "month"):
        assert described[name]["rows"] > 0
        assert "return_correlation" in described[name]
        assert 1.0 <= described[name]["effective_symbols"] <= 2.0
    # Without the pool it still reports rows, and claims nothing more.
    assert "effective_symbols" not in describe_pool(slices)["week"]


# ── The flag ────────────────────────────────────────────────────────────────

def test_the_predicted_symbol_is_always_in_the_pool_and_first() -> None:
    """A run that trained on everything except the symbol it forecasts is a
    different experiment, and not one anybody asks for by omitting a flag."""
    assert parse_symbols(None, "ETHUSDT") == ["ETHUSDT"]
    assert parse_symbols("BTCUSDT,SOLUSDT", "ETHUSDT") == ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
    # Naming the target again must not duplicate its rows.
    assert parse_symbols("ETHUSDT,BTCUSDT", "ETHUSDT") == ["ETHUSDT", "BTCUSDT"]
    assert parse_symbols("btcusdt", "ethusdt") == ["ETHUSDT", "BTCUSDT"]
    assert parse_symbols("  ", "ETHUSDT") == ["ETHUSDT"]


def test_an_empty_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        TrainingPool([], INTERVAL)
