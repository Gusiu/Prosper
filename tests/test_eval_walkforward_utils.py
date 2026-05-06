"""Tests for walk-forward evaluation utility functions."""

from prosper.eval.walkforward import multiclass_logloss_brier
from prosper.labels.depth import direction_from_return


def test_label_from_return_thresholding() -> None:
    thr = 0.01
    assert direction_from_return(0.02, thr) == "long"
    assert direction_from_return(-0.03, thr) == "short"
    assert direction_from_return(0.0, thr) == "flat"
    assert direction_from_return(0.01, thr) == "flat"
    assert direction_from_return(-0.01, thr) == "flat"


def test_multiclass_logloss_brier_are_finite_positive() -> None:
    logloss, brier = multiclass_logloss_brier(
        p_long=0.7,
        p_flat=0.2,
        p_short=0.1,
        y_true="long",
    )
    assert logloss >= 0.0
    assert 0.0 <= brier <= 2.0

