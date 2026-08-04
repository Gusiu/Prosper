"""Tests for walk-forward evaluation utility functions."""

import pytest
from prosper.domain import direction_from_return
from prosper.eval.walkforward import multiclass_logloss_brier


def test_direction_is_the_sign_of_the_return() -> None:
    assert direction_from_return(0.02) == "long"
    assert direction_from_return(-0.03) == "short"
    # No threshold and no flat class: a 0.2% move is a real move. Whether it
    # is worth trading is a cost question, answered by the planner.
    assert direction_from_return(0.002) == "long"
    assert direction_from_return(-0.002) == "short"


def test_an_exactly_zero_return_has_no_side() -> None:
    """It must not be forced into a class — the row is excluded instead."""
    assert direction_from_return(0.0) is None
    assert direction_from_return(None) is None


def test_multiclass_logloss_brier_are_finite_positive() -> None:
    logloss, brier = multiclass_logloss_brier({"long": 0.7, "short": 0.3}, "long")

    assert logloss >= 0.0
    assert 0.0 <= brier <= 2.0
    assert brier == pytest.approx(0.09 + 0.09)


def test_scoring_follows_the_run_s_own_class_set() -> None:
    """A legacy three-class run stays scorable by the same function."""
    logloss, brier = multiclass_logloss_brier({"long": 0.7, "flat": 0.2, "short": 0.1}, "long")

    assert logloss == pytest.approx(-__import__("math").log(0.7))
    assert brier == pytest.approx(0.09 + 0.04 + 0.01)


def test_a_class_the_run_never_emitted_scores_as_a_miss() -> None:
    logloss, _ = multiclass_logloss_brier({"long": 0.7, "short": 0.3}, "flat")
    assert logloss > 0.0
