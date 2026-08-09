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
    assert "medium (182d)" in message
    longest = max(DEFAULT_HORIZONS, key=lambda h: h.forward_days)
    assert f"long ({longest.forward_days}d)" in message
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


def test_medium_is_exactly_half_of_long() -> None:
    """Not decoration: 365/2 is 182.5, so the two horizons were not
    commensurate and no window arithmetic could make them so."""
    spans = {h.name: h.forward_days for h in DEFAULT_HORIZONS}
    assert spans["medium"] * 2 == spans["long"]
    assert spans["long"] == spans["short"] * 13


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
