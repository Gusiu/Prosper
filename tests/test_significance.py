"""An error bar that ignores label overlap is worse than none at all.

Every accuracy in this project is a mean over rows whose labels overlap: the
label at bar k covers [k, k+forward], so the label at k+1 shares forward-1 days
with it. At the annual horizon consecutive rows agree on 363 of 364 days. A
naive interval treats those as independent and comes out several times too
narrow, which would make 0.565 look distinguishable from 0.500 when it is not.

The moving-block bootstrap resamples runs of consecutive rows instead, so each
resample keeps that correlation. These tests pin the property that matters —
that it is *wider* than the naive version on correlated data and agrees with it
when the data really is independent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from prosper.eval.significance import (
    MIN_BLOCKS,
    accuracy_interval,
    block_bootstrap,
    calibration_error_interval,
    effective_samples,
    mean_interval,
)


def _persistent_series(n: int, block: int, seed: int = 0) -> np.ndarray:
    """0/1 values in runs of `block`, mimicking what overlap produces: whole
    stretches sharing an outcome."""
    rng = np.random.default_rng(seed)
    runs = rng.integers(0, 2, size=int(np.ceil(n / block)))
    return np.repeat(runs, block)[:n].astype(float)


def test_effective_samples_divides_by_the_overlap_length() -> None:
    assert effective_samples(2040, 28) == pytest.approx(72.9, abs=0.1)
    assert effective_samples(1249, 364) == pytest.approx(3.4, abs=0.1)
    # No overlap means every row counts.
    assert effective_samples(500, 1) == 500.0


def test_the_interval_is_wider_on_correlated_data_than_on_independent_data() -> None:
    """The whole reason this module exists."""
    n, block = 1000, 50
    correlated = _persistent_series(n, block, seed=1)

    honest = block_bootstrap(correlated, block_size=block, seed=0)
    naive = block_bootstrap(correlated, block_size=1, seed=0)

    honest_width = honest.high - honest.low
    naive_width = naive.high - naive.low
    assert honest_width > 3 * naive_width, (
        f"block bootstrap {honest_width:.3f} vs naive {naive_width:.3f}; "
        "ignoring overlap must produce a visibly narrower interval"
    )


def test_on_independent_data_it_matches_the_analytic_interval() -> None:
    """With block_size=1 there is nothing to correct for, so it must not invent
    extra width either."""
    rng = np.random.default_rng(3)
    values = rng.integers(0, 2, size=4000).astype(float)

    interval = block_bootstrap(values, block_size=1, seed=0)
    analytic = 1.96 * math.sqrt(values.mean() * (1 - values.mean()) / len(values))

    width = (interval.high - interval.low) / 2
    assert width == pytest.approx(analytic, rel=0.2)



def test_the_point_estimate_is_the_statistic_itself() -> None:
    values = np.array([1.0, 0.0, 1.0, 1.0] * 100)
    assert block_bootstrap(values, block_size=4, seed=0).point == pytest.approx(0.75)


def test_the_interval_brackets_the_point_estimate() -> None:
    values = _persistent_series(600, 20, seed=5)
    interval = block_bootstrap(values, block_size=20, seed=0)
    assert interval.low <= interval.point <= interval.high


def test_too_few_blocks_is_a_refusal_not_a_narrow_interval() -> None:
    """The annual horizon is close to this: 1249 rows over a 364-bar overlap is
    about three blocks. Resampling one observation would report false precision.
    """
    values = _persistent_series(400, 364, seed=0)
    interval = block_bootstrap(values, block_size=364, seed=0)

    assert not interval.estimable
    assert interval.low is None and interval.high is None
    assert "independent blocks" in interval.reason
    # The point estimate is still reported; only the precision claim is withheld.
    assert not np.isnan(interval.point)
    assert interval.blocks < MIN_BLOCKS


def test_an_unestimable_interval_never_claims_a_difference() -> None:
    """`excludes` must return None, not False: "cannot say" is not "no effect"."""
    interval = block_bootstrap(_persistent_series(400, 364), block_size=364, seed=0)
    assert interval.excludes(0.5) is None


def test_excludes_detects_a_real_separation() -> None:
    always_right = np.ones(1000)
    interval = block_bootstrap(always_right, block_size=10, seed=0)

    assert interval.excludes(0.5) is True
    assert interval.excludes(1.0) is False


def test_a_coin_flip_is_not_separated_from_chance() -> None:
    rng = np.random.default_rng(11)
    interval = accuracy_interval(rng.integers(0, 2, size=2000).astype(bool), block_size=28, seed=0)

    assert interval.excludes(0.5) is False, "a coin flip must not read as skill"


def test_no_scored_rows_is_reported_as_such() -> None:
    interval = block_bootstrap(np.array([]), block_size=28, seed=0)
    assert not interval.estimable
    assert interval.samples == 0
    assert interval.reason == "no scored rows"


def test_the_result_is_reproducible_from_the_seed() -> None:
    values = _persistent_series(800, 25, seed=2)
    first = block_bootstrap(values, block_size=25, seed=7)
    second = block_bootstrap(values, block_size=25, seed=7)

    assert (first.low, first.high) == (second.low, second.high)


def test_a_different_seed_moves_the_interval_only_a_little() -> None:
    values = _persistent_series(800, 25, seed=2)
    a = block_bootstrap(values, block_size=25, seed=1)
    b = block_bootstrap(values, block_size=25, seed=2)

    width = a.high - a.low
    assert abs(a.low - b.low) < 0.25 * width
    assert abs(a.high - b.high) < 0.25 * width


def test_every_resample_has_the_original_length() -> None:
    """A short final block must be truncated, not dropped: dropping it would
    shift the statistic toward whatever the earlier blocks happen to hold."""
    seen: list[int] = []

    def record(block: np.ndarray) -> float:
        seen.append(len(block))
        return float(np.mean(block))

    values = _persistent_series(97, 10, seed=0)
    block_bootstrap(values, block_size=10, statistic=record, resamples=20, seed=0)

    assert set(seen) == {97}


def test_mean_interval_handles_an_unbounded_score() -> None:
    """NLL is not a proportion; nothing may assume [0, 1]."""
    rng = np.random.default_rng(4)
    nll = rng.exponential(scale=0.7, size=1500)
    interval = mean_interval(nll, block_size=28, seed=0)

    assert interval.low < interval.point < interval.high
    assert interval.high > 0.0


def test_calibration_error_is_recomputed_inside_each_resample() -> None:
    """ECE is a weighted sum over bins, not a row-wise mean, so averaging
    per-row values would give the wrong quantity entirely."""
    rng = np.random.default_rng(9)
    n = 2000
    confidence = rng.uniform(0.5, 1.0, size=n)
    # Perfectly calibrated: right exactly as often as it claims.
    correct = rng.random(n) < confidence

    interval = calibration_error_interval(confidence, correct, block_size=28, seed=0)

    assert interval.estimable
    assert 0.0 <= interval.low <= interval.high
    assert interval.point < 0.1, "a calibrated forecast has a small ECE"


def test_a_miscalibrated_forecast_has_an_interval_away_from_zero() -> None:
    rng = np.random.default_rng(10)
    n = 2000
    confidence = np.full(n, 0.95)
    correct = rng.random(n) < 0.55

    interval = calibration_error_interval(confidence, correct, block_size=28, seed=0)

    assert interval.low > 0.2, "claiming 0.95 while right 0.55 of the time is a large gap"
    assert interval.excludes(0.0) is True
