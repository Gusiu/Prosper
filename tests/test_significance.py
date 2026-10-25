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

from prosper.domain import DEFAULT_HORIZONS
from prosper.eval.significance import (
    MIN_BLOCKS,
    accuracy_interval,
    benjamini_hochberg,
    block_bootstrap,
    bootstrap_p_value,
    calibration_error_interval,
    effective_samples,
    mean_interval,
)

SHORTEST = DEFAULT_HORIZONS[0].name
SHORTEST_STEPS = DEFAULT_HORIZONS[0].steps("1d")


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


# ── A p-value read off the same resampling ───────────────────────────────────

def _biased(n: int, rate: float, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).random(n) < rate).astype(float)


def test_the_p_value_agrees_with_the_interval_about_chance() -> None:
    """Small where the interval leaves 0.5 out, on either side, and not where it
    holds it."""
    for rate, separated in ((0.56, True), (0.44, True), (0.5, False)):
        values = _biased(3000, rate, seed=6)
        interval = accuracy_interval(values.astype(bool), block_size=7, seed=0)
        p = bootstrap_p_value(values, 7, 0.5, seed=0)
        assert interval.excludes(0.5) is separated
        assert (p < 0.05) is separated, (rate, p)


def test_at_either_end_of_the_interval_the_p_value_is_alpha() -> None:
    """The p-value is the 95% interval inverted, so a reference placed on one of
    its ends comes out at about 0.05."""
    values = _persistent_series(3000, 20, seed=3)
    interval = block_bootstrap(values, block_size=20, seed=0)
    for bound in (interval.low, interval.high):
        assert bootstrap_p_value(values, 20, bound, seed=0) == pytest.approx(0.05, abs=0.005)


def test_the_p_value_is_read_off_the_intervals_own_resampling() -> None:
    """Same draws as the interval it is reported next to, and a longer resampling
    under the same seed only adds draws after them."""
    seen: dict[str, list[float]] = {"interval": [], "p": [], "longer": []}

    def recorder(name: str):
        def statistic(block: np.ndarray) -> float:
            seen[name].append(float(block.sum()))
            return float(np.mean(block))
        return statistic

    values = _persistent_series(500, 10, seed=4)
    block_bootstrap(values, 10, recorder("interval"), resamples=200, seed=3)
    bootstrap_p_value(values, 10, 0.5, recorder("p"), resamples=200, seed=3)
    bootstrap_p_value(values, 10, 0.5, recorder("longer"), resamples=400, seed=3)

    assert seen["p"] == seen["interval"]
    assert seen["longer"][: len(seen["interval"])] == seen["interval"]


def test_a_resampling_that_never_crosses_the_reference_is_not_p_zero() -> None:
    """It shows only that p lies below the resolution of the resampling."""
    always_right = np.ones(1000)
    assert bootstrap_p_value(always_right, 10, 0.5, resamples=999, seed=0) == pytest.approx(0.002)


def test_the_estimate_itself_is_no_departure() -> None:
    values = _persistent_series(3000, 20, seed=8)
    point = float(values.mean())
    assert bootstrap_p_value(values, 20, point, seed=0) > 0.8


def test_q_values_scale_each_p_by_its_rank_and_stay_monotone() -> None:
    """Worked by hand: sorted 0.005, 0.01, 0.03, 0.04 scale to 0.02, 0.02, 0.04,
    0.04, and come back in the order given."""
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert q == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_a_smaller_p_never_gets_a_larger_q() -> None:
    """Without the pass from the largest down, 0.011 at rank 2 would scale to
    0.022 and outrank 0.012 at rank 3, which scales to 0.016."""
    p = [0.011, 0.012, 0.009, 0.5]
    q = benjamini_hochberg(p)
    by_p = [q[i] for i in sorted(range(len(p)), key=lambda i: p[i])]
    assert by_p == sorted(by_p)
    assert all(qi >= pi for pi, qi in zip(p, q))
    assert max(q) <= 1.0


def test_with_one_test_the_q_value_is_the_p_value() -> None:
    assert benjamini_hochberg([0.03]) == [0.03]
    assert benjamini_hochberg([]) == []


def test_a_refused_interval_has_no_p_value() -> None:
    """Too few blocks, and a resampling that cannot bracket its own estimate."""
    assert bootstrap_p_value(_persistent_series(400, 364), 364, 0.5, seed=0) is None
    lopsided = np.concatenate([np.ones(300), np.zeros(60)])
    assert bootstrap_p_value(lopsided, 120, 0.5, seed=0) is None
    assert bootstrap_p_value(np.array([]), 28, 0.5, seed=0) is None


# ── Wired into the evaluation ────────────────────────────────────────────────

def test_the_evaluation_reports_an_interval_for_every_horizon() -> None:
    """A bare accuracy is not a result, so it must not be possible to get one."""
    from prosper.eval.predictions import EvaluationSettings, _aggregate_quality

    rng = np.random.default_rng(0)
    rows = []
    spans = [(h.name, h.steps("1d")) for h in DEFAULT_HORIZONS]
    for horizon, forward in spans:
        for i in range(1200):
            confidence = float(rng.uniform(0.5, 0.9))
            rows.append(
                {
                    "status": "scored",
                    "horizon": horizon,
                    "forward_steps": forward,
                    "prediction": {"confidence": confidence, "probabilities": {"a": 0.5, "b": 0.5}},
                    "metrics": {
                        "correct": bool(rng.random() < 0.5),
                        "brier": 0.5,
                        "nll": 0.7,
                        "entropy": 0.69,
                        "depth_kl": None,
                        "depth_js": None,
                        "depth_abs_error_pct": None,
                    },
                }
            )

    metrics, _ = _aggregate_quality(rows, EvaluationSettings())
    by_horizon = metrics["by_horizon"]

    for horizon, forward in spans:
        stats = by_horizon[horizon]
        interval = stats["accuracy_ci"]
        assert interval is not None, horizon
        assert interval["block_size"] == forward, "the block must be the overlap length"
        assert stats["ece_ci"] is not None
        # 1200 rows over a 364-bar overlap is 3.3 blocks: estimable, but barely.
        assert interval["effective_samples"] == pytest.approx(1200 / forward, abs=0.1)


def test_a_coin_flip_evaluation_never_claims_to_beat_chance() -> None:
    from prosper.eval.predictions import EvaluationSettings, _aggregate_quality

    rng = np.random.default_rng(4)
    rows = [
        {
            "status": "scored",
            "horizon": SHORTEST,
            "forward_steps": SHORTEST_STEPS,
            "prediction": {"confidence": 0.6, "probabilities": {"a": 0.6, "b": 0.4}},
            "metrics": {
                "correct": bool(rng.random() < 0.5),
                "brier": 0.5, "nll": 0.7, "entropy": 0.69,
                "depth_kl": None, "depth_js": None, "depth_abs_error_pct": None,
            },
        }
        for _ in range(2000)
    ]

    metrics, _ = _aggregate_quality(rows, EvaluationSettings())
    by_horizon = metrics["by_horizon"]
    assert by_horizon[SHORTEST]["accuracy_vs_chance"] == "indistinguishable"


def test_the_cli_prints_the_interval_and_the_verdict() -> None:
    """The table is the point: a number quoted without it is the failure this
    module exists to prevent."""
    import inspect

    from prosper.cli import _run_predictions_evaluation

    source = inspect.getsource(_run_predictions_evaluation)
    assert "accuracy_ci" in source
    assert "low > 0.5" in source and "high < 0.5" in source
    assert "effective_samples" in source
    for verdict in ("> chance", "< chance", "= chance", "unknown"):
        assert verdict in source, verdict


def test_an_interval_that_cannot_bracket_its_estimate_is_refused() -> None:
    """Found in real output, not imagined: at 3.1 blocks two of fifty-five
    intervals came out excluding their own point estimate — 0.339 against
    [0.066, 0.335] — because four blocks drawn and truncated cannot represent a
    series holding three. The block count alone did not catch it, so the
    inconsistency is detected directly.
    """
    # A series with a single long run, so resampling cannot reproduce the mean.
    values = np.concatenate([np.ones(300), np.zeros(60)])
    interval = block_bootstrap(values, block_size=120, seed=0)

    assert not interval.estimable
    assert interval.reason is not None


def test_the_floor_reflects_the_measured_failure() -> None:
    """Three blocks was the training-window floor borrowed for a different
    question. It is not enough for a stable interval."""
    assert MIN_BLOCKS >= 10


def test_a_well_sampled_interval_always_brackets_its_estimate() -> None:
    """The guard must not fire on data that is fine, or it would silently turn
    every result into "unknown"."""
    rng = np.random.default_rng(2)
    for seed in range(5):
        values = _persistent_series(3000, 20, seed=seed)
        interval = block_bootstrap(values, block_size=20, seed=seed)
        assert interval.estimable, interval.reason
        assert interval.low <= interval.point <= interval.high
    assert rng is not None


def test_the_verdict_says_which_side_of_chance() -> None:
    """`excludes` is two-sided: it is true for a run whose whole interval sits
    below 0.5, which no count of runs "above chance" may read as skill."""
    from prosper.eval.significance import Interval

    def interval(low, high):
        return Interval(point=(low + high) / 2, low=low, high=high, block_size=7,
                        blocks=40.0, samples=280, resamples=2000)

    assert interval(0.253, 0.415).versus(0.5) == "below"
    assert interval(0.253, 0.415).excludes(0.5) is True  # which is why a name matters
    assert interval(0.512, 0.590).versus(0.5) == "above"
    assert interval(0.460, 0.553).versus(0.5) == "indistinguishable"
    refused = Interval(point=0.5, low=None, high=None, block_size=364, blocks=4.7,
                       samples=1700, resamples=2000, reason="too few blocks")
    assert refused.versus(0.5) is None

