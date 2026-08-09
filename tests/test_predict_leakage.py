"""Tests for the walk-forward training-window guard used by every predictor.

These exercise `prosper.predict.window` directly. An earlier version of this
module re-implemented the boundary arithmetic inside the test, so it kept
passing while the real predictors shipped a window that could not train the
medium and long horizons at all.
"""

import pytest
from prosper.domain import DEFAULT_HORIZONS, DIRECTION_CLASSES, HorizonSpec, days_to_steps
from prosper.predict.window import (
    MIN_TRAIN_SAMPLES,
    training_bounds,
    untrained_horizon_payload,
    usable_sample_count,
    validate_train_window,
)


def test_training_bounds_exclude_labels_that_need_future_closes() -> None:
    start, end = training_bounds(row_index=100, train_window_steps=730, forward_steps=28)

    assert start == 0  # window is wider than the available history
    assert end == 72
    # The last included label sits at 71 and needs close[71 + 28] = close[99],
    # which is known when predicting bar 100.
    assert end - 1 + 28 < 100


def test_training_bounds_respects_window_floor() -> None:
    start, end = training_bounds(row_index=1000, train_window_steps=200, forward_steps=28)

    assert start == 800
    assert end == 972
    assert usable_sample_count(1000, 200, 28) == 172


def test_usable_sample_count_is_zero_when_window_matches_horizon() -> None:
    # This is precisely the collapse the guard exists to prevent.
    assert usable_sample_count(row_index=1000, train_window_steps=150, forward_steps=182) == 0


def test_validate_train_window_accepts_a_sufficient_window() -> None:
    steps = validate_train_window(730, "1d", DEFAULT_HORIZONS)

    # Read from the specs, not repeated: the horizons are whole weeks (4/26/52)
    # and a literal here would silently disagree the next time they move.
    assert steps == {h.name: h.forward_days for h in DEFAULT_HORIZONS}
    assert all(h.forward_days % 7 == 0 for h in DEFAULT_HORIZONS), "whole weeks"


def test_validate_train_window_rejects_a_window_narrower_than_the_horizon() -> None:
    with pytest.raises(ValueError) as exc:
        validate_train_window(150, "1d", DEFAULT_HORIZONS)

    message = str(exc.value)
    longest = max(DEFAULT_HORIZONS, key=lambda h: h.forward_days)
    assert f"{longest.name} ({longest.forward_days}d)" in message
    assert "--train-window-days" in message


def test_validate_train_window_needs_headroom_beyond_the_horizon() -> None:
    # Exactly one horizon-length of window leaves zero training samples.
    with pytest.raises(ValueError):
        validate_train_window(365, "1d", [HorizonSpec("long", 365)])

    # Horizon plus the minimum sample count is enough.
    validate_train_window(365 + MIN_TRAIN_SAMPLES, "1d", [HorizonSpec("long", 365)])


def test_horizons_span_the_same_calendar_time_across_intervals() -> None:
    long_horizon = HorizonSpec("long", 364)

    assert long_horizon.steps("1d") == 364
    assert long_horizon.steps("1h") == 364 * 24
    assert long_horizon.steps("1w") == 52


def test_the_shipped_horizons_convert_exactly_to_whole_weeks() -> None:
    """The reason `long` is 364 rather than 365.

    `days_to_steps` rounds, so a 365-day horizon became 52 bars at the 1w
    interval — 364 days — while staying 365 at 1d. Invariant 2 requires both to
    ask the same question, and they did not: a 1w run's `long` was a day shorter
    than a 1d run's. Whole weeks remove the rounding entirely.
    """
    for horizon in DEFAULT_HORIZONS:
        assert horizon.forward_days % 7 == 0, f"{horizon.name} is not whole weeks"
        weeks = horizon.forward_days // 7
        assert horizon.steps("1w") == weeks
        # Round-trip: the bar count means exactly the calendar span asked for.
        assert horizon.steps("1w") * 7 == horizon.forward_days
        assert horizon.steps("1d") == horizon.forward_days
        assert horizon.steps("1h") == horizon.forward_days * 24


def test_the_longest_horizon_is_a_whole_number_of_the_others() -> None:
    """Commensurate spans, so a horizon is always a clean multiple of a shorter
    one. 365 could not do this: 365/2 is 182.5 and 365/7 is not an integer."""
    spans = {h.name: h.forward_days for h in DEFAULT_HORIZONS}

    assert spans["year"] == 4 * spans["quarter"], "a year is four quarters"
    assert spans["month"] == 4 * spans["week"]
    assert spans["year"] == 52 * spans["week"]
    assert all(span % spans["week"] == 0 for span in spans.values())


def test_train_window_scales_with_interval() -> None:
    assert days_to_steps(730, "1d") == 730
    assert days_to_steps(730, "1h") == 730 * 24


def test_validate_train_window_scales_the_requirement_by_interval() -> None:
    # 730 calendar days is plenty at 1d and, expressed in bars, also at 1h.
    validate_train_window(730, "1h", DEFAULT_HORIZONS)

    # But a window shorter in calendar terms than the horizon fails regardless
    # of how many bars it happens to contain.
    with pytest.raises(ValueError):
        validate_train_window(200, "1h", DEFAULT_HORIZONS)


def test_untrained_payload_is_flagged_and_normalised() -> None:
    payload = untrained_horizon_payload()

    assert payload["trained"] is False
    direction_mass = sum(v for k, v in payload.items() if k.startswith("P_"))
    assert direction_mass == pytest.approx(1.0)
    assert set(DIRECTION_CLASSES) == {k[2:] for k in payload if k.startswith("P_")}
    assert sum(payload["depth_long_bins"].values()) == pytest.approx(1.0)


# ── Statistical power ────────────────────────────────────────────────────────

def test_the_window_reports_how_many_observations_each_horizon_has() -> None:
    """`validate_train_window` refuses a window that cannot train; this is its
    sibling for a window that can train but cannot measure.

    A 730-day window holds 730 - 364 = 366 labelled bars at the annual horizon,
    and those overlap by 363 of 364 days — about one independent observation. No
    model, optimiser or epoch budget changes that; it is arithmetic about how
    much non-overlapping history two years contains.
    """
    from prosper.predict.window import training_window_power

    steps = {"month": 28, "quarter": 91, "year": 364}
    power = training_window_power(days_to_steps(730, "1d"), steps)

    assert power["month"] == pytest.approx(25.1, abs=0.1)
    assert power["quarter"] == pytest.approx(7.0, abs=0.1)
    assert power["year"] == pytest.approx(1.0, abs=0.1)


def test_a_quarter_has_more_than_twice_the_power_of_a_half_year() -> None:
    """The reason the horizon set moved to 91 days, and not an aesthetic one."""
    from prosper.predict.window import training_window_power

    window = days_to_steps(730, "1d")
    half_year = training_window_power(window, {"h": 182})["h"]
    quarter = training_window_power(window, {"h": 91})["h"]

    assert quarter > 2 * half_year


def test_an_underpowered_horizon_is_warned_about() -> None:
    from prosper.predict.window import warn_low_power

    messages = warn_low_power(days_to_steps(730, "1d"), {"month": 28, "year": 364})

    assert len(messages) == 1
    assert "'year'" in messages[0]
    assert "1.0 independent observations" in messages[0]


def test_a_well_powered_configuration_warns_about_nothing() -> None:
    from prosper.predict.window import warn_low_power

    assert warn_low_power(days_to_steps(730, "1d"), {"week": 7, "month": 28}) == []


def test_validate_train_window_warns_without_refusing() -> None:
    """The annual horizon must still run — that it cannot be validated is a
    finding worth reporting, not a reason to hide it."""
    steps = validate_train_window(730, "1d", DEFAULT_HORIZONS)
    assert steps, "the configuration is accepted"


def test_the_floor_is_the_one_the_bootstrap_uses() -> None:
    """Two thresholds for one idea would drift; the evaluation refusing an
    interval and the trainer warning must agree on what "too few" means."""
    from prosper.domain import MIN_INDEPENDENT_OBSERVATIONS
    from prosper.eval.significance import MIN_BLOCKS

    assert MIN_BLOCKS == MIN_INDEPENDENT_OBSERVATIONS
