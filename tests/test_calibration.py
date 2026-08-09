"""Calibration is fitted inside the training window, never after it.

The direction heads were confident far past their accuracy: median |edge| of
0.996 against roughly coin-flip accuracy, ECE 0.40-0.46. That makes every
threshold downstream meaningless — the planner's seven-point scale collapsed to
Strong Buy / Strong Sell on 90% of bars. Temperature scaling pulls the mass
back without reordering anything.
"""

from __future__ import annotations

import numpy as np
import pytest
from prosper.domain import DIRECTION_CLASSES
from prosper.predict.calibration import (
    CALIBRATION_FRACTION,
    MIN_CALIBRATION_SAMPLES,
    TemperatureCalibrator,
    calibrated_nll,
    calibration_split,
    fit_calibrator_from_model,
    fit_temperature,
)
from prosper.predict.window import training_bounds


def test_the_holdout_is_carved_out_of_the_training_region() -> None:
    split, cal_start = calibration_split(0, 1000)

    assert split == cal_start
    assert split == 1000 - int(1000 * CALIBRATION_FRACTION)
    assert 0 < split < 1000


def test_a_region_too_small_to_hold_out_is_left_whole() -> None:
    """Below the sample floor the calibrator would fit noise, so it is skipped."""
    span = int(MIN_CALIBRATION_SAMPLES / CALIBRATION_FRACTION) - 1
    split, _ = calibration_split(0, span)

    assert split == span, "no rows held out"


def test_calibration_never_reaches_past_the_training_bound() -> None:
    """The whole point: the held-out slice must still be realised history.

    `training_bounds` already pulls the end back by `forward_steps` from the
    prediction bar, so both halves of the split sit strictly before it.
    """
    row_index, window, forward_steps = 900, 730, 365
    train_start, train_end = training_bounds(row_index, window, forward_steps)
    split, cal_start = calibration_split(train_start, train_end)

    assert train_start <= split <= train_end
    assert cal_start <= train_end
    # Every calibration label needs a close before the bar being predicted.
    assert train_end + forward_steps <= row_index


def test_temperature_above_one_softens_an_overconfident_forecast() -> None:
    calibrator = TemperatureCalibrator(3.0)
    softened = calibrator.apply([0.98, 0.02])

    assert sum(softened) == pytest.approx(1.0)
    assert softened[0] < 0.98
    assert softened[0] > softened[1], "the ranking must not flip"


def test_temperature_below_one_sharpens() -> None:
    sharpened = TemperatureCalibrator(0.5).apply([0.6, 0.4])
    assert sharpened[0] > 0.6


def test_the_identity_temperature_changes_nothing() -> None:
    probs = [0.7, 0.3]
    assert TemperatureCalibrator(1.0).apply(probs) == pytest.approx(probs)


def test_calibration_preserves_the_ranking_for_every_temperature() -> None:
    probs = [0.55, 0.45]
    for t in (0.25, 0.5, 1.0, 2.0, 10.0):
        out = TemperatureCalibrator(t).apply(probs)
        assert out[0] >= out[1]
        assert sum(out) == pytest.approx(1.0)


def test_a_zeroed_class_does_not_produce_a_nan() -> None:
    out = TemperatureCalibrator(2.0).apply([1.0, 0.0])
    assert all(np.isfinite(v) for v in out)
    assert sum(out) == pytest.approx(1.0)


def test_a_degenerate_vector_falls_back_to_uniform() -> None:
    out = TemperatureCalibrator(1.5).apply([0.0, 0.0])
    assert out == pytest.approx([0.5, 0.5])


def test_fitting_recovers_a_softening_temperature_from_overconfidence() -> None:
    """Predictions of 0.95 that are right only ~half the time need T > 1."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, len(DIRECTION_CLASSES), size=n)
    probs = np.full((n, len(DIRECTION_CLASSES)), 0.05)
    # Confidently assert a class that is correct only half the time.
    asserted = np.where(rng.random(n) < 0.5, y, 1 - y)
    probs[np.arange(n), asserted] = 0.95
    probs /= probs.sum(axis=1, keepdims=True)

    calibrator = fit_temperature(probs, y)

    assert calibrator.temperature > 1.0


def test_fitting_leaves_an_already_calibrated_forecast_alone() -> None:
    rng = np.random.default_rng(7)
    n = 600
    p_long = rng.uniform(0.2, 0.8, size=n)
    y = (rng.random(n) < p_long).astype(int)
    probs = np.column_stack([1.0 - p_long, p_long])

    calibrator = fit_temperature(probs, y)

    assert 0.7 <= calibrator.temperature <= 1.5


def test_a_single_class_holdout_never_sharpens() -> None:
    """Minimising NLL on one class drives the temperature to its floor.

    The model is "right" on every held-out row, so the loss keeps falling as
    the temperature drops and the fit turns an overconfident forecast into a
    certain one. On BTCUSDT the 365-day horizon hits this often — long
    stretches where every forward return shares a sign — and it left the long
    horizon at |edge| = 1.000 for every single bar.
    """
    n = 200
    probs = np.tile([0.99, 0.01], (n, 1))
    y = np.zeros(n, dtype=int)  # the model is always "correct"

    assert fit_temperature(probs, y).is_identity


def test_too_few_samples_yields_the_identity() -> None:
    probs = np.full((5, len(DIRECTION_CLASSES)), 1.0 / len(DIRECTION_CLASSES))
    assert fit_temperature(probs, np.zeros(5, dtype=int)).is_identity


class _Overconfident:
    def predict_proba(self, X):
        return np.tile([0.98, 0.02], (len(X), 1))


class _Broken:
    def predict_proba(self, X):
        raise RuntimeError("no model")


def test_a_model_is_calibrated_on_its_own_holdout_predictions() -> None:
    X = np.zeros((200, 3))
    # It claims class 0 at 0.98 but is right only half the time.
    y = [0, 1] * 100

    calibrator = fit_calibrator_from_model(_Overconfident(), X, y, present_classes=[0, 1])

    assert calibrator.temperature > 1.0


def test_a_model_that_cannot_score_the_holdout_is_not_fatal() -> None:
    """Calibration is an improvement, never a prerequisite for a prediction."""
    calibrator = fit_calibrator_from_model(_Broken(), np.zeros((100, 3)), [0, 1] * 50)
    assert calibrator.is_identity


def test_a_missing_model_yields_the_identity() -> None:
    assert fit_calibrator_from_model(None, np.zeros((100, 2)), [0] * 100).is_identity


def test_a_single_class_column_is_placed_at_its_own_index() -> None:
    """A window where only `long` occurred still maps to the right column."""

    class OneColumn:
        def predict_proba(self, X):
            return np.ones((len(X), 1))

    calibrator = fit_calibrator_from_model(
        OneColumn(), np.zeros((100, 2)), [1] * 100, present_classes=[1]
    )
    assert isinstance(calibrator, TemperatureCalibrator)


def test_apply_to_mapping_keeps_the_class_names() -> None:
    out = TemperatureCalibrator(2.0).apply_to_mapping({"short": 0.9, "long": 0.1})

    assert set(out) == {"short", "long"}
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["short"] > out["long"]


def _raw_nll(probs: np.ndarray, y: np.ndarray) -> float:
    picked = np.asarray(probs)[np.arange(len(y)), y]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def test_sharpening_raises_the_raw_loss_but_not_the_calibrated_one() -> None:
    """This is the whole reason DL training stopped after one epoch.

    Raw cross-entropy charges for confidence. A network that keeps the same
    ranking but states it more strongly scores worse on it, even though the
    temperature fit that follows training undoes exactly that. Measured on
    BTCUSDT the raw sequence ran 0.7244, 0.9109, 0.9282, 1.0216 — rising from
    the first epoch — while an epoch sweep showed accuracy still climbing.
    """
    rng = np.random.default_rng(11)
    n = 500
    p_long = rng.uniform(0.15, 0.85, size=n)
    y = (rng.random(n) < p_long).astype(int)
    probs = np.column_stack([1.0 - p_long, p_long])

    sharpened = np.array([TemperatureCalibrator(0.5).apply(row) for row in probs])

    assert _raw_nll(sharpened, y) > _raw_nll(probs, y), "raw loss punishes the sharpening"
    assert calibrated_nll(sharpened, y) == pytest.approx(calibrated_nll(probs, y), rel=1e-6)


def test_the_calibrated_loss_never_exceeds_the_raw_one() -> None:
    """T = 1 is in the grid, so the minimum can only be lower or equal."""
    rng = np.random.default_rng(3)
    n = 300
    p_long = rng.uniform(0.05, 0.95, size=n)
    y = (rng.random(n) < 0.5).astype(int)
    probs = np.column_stack([1.0 - p_long, p_long])

    assert calibrated_nll(probs, y) <= _raw_nll(probs, y) + 1e-9


def test_better_separation_lowers_the_calibrated_loss() -> None:
    """It still measures something: what survives is discrimination."""
    rng = np.random.default_rng(5)
    n = 400
    y = rng.integers(0, 2, size=n)

    # Informative: the stated probability tracks the outcome.
    good_long = np.where(y == 1, 0.7, 0.3)
    good = np.column_stack([1.0 - good_long, good_long])
    # Uninformative: same confidence, unrelated to the outcome.
    noise_long = rng.choice([0.3, 0.7], size=n)
    noise = np.column_stack([1.0 - noise_long, noise_long])

    assert calibrated_nll(good, y) < calibrated_nll(noise, y)


def test_a_single_class_holdout_falls_back_to_the_raw_loss() -> None:
    """Minimising over T on one class drives the loss to zero for free."""
    probs = np.tile([0.6, 0.4], (50, 1))
    y = np.zeros(50, dtype=int)

    assert calibrated_nll(probs, y) == pytest.approx(_raw_nll(probs, y))


def test_an_empty_holdout_is_infinitely_bad_not_zero() -> None:
    """Returning 0.0 would make an unscoreable epoch look like the best one."""
    assert calibrated_nll(np.zeros((0, 2)), []) == float("inf")


def test_max_entropy_follows_the_class_count() -> None:
    """Scoring normalised NLL against ln(3) after direction became two classes.

    A coin-flip forecast over two classes has NLL ln 2 = 0.693. Divided by
    ln 3 = 1.099 it scored 0.37 of the "skill" it was meant to earn zero of,
    so every run in the table was credited for guessing.
    """
    import math

    from prosper.eval.predictions import max_entropy

    assert max_entropy(2) == pytest.approx(math.log(2))
    assert max_entropy(3) == pytest.approx(math.log(3))
    # A degenerate count must not make the divisor zero or negative.
    assert max_entropy(1) == pytest.approx(math.log(2))
    assert max_entropy(0) == pytest.approx(math.log(2))


def test_a_coin_flip_earns_no_nll_points() -> None:
    """The component exists to reward beating ignorance, not reaching it."""
    import math

    from prosper.eval.predictions import _score_prediction

    guess = _score_prediction(
        correct=False, confidence=0.5, brier=0.5,
        nll=math.log(2), depth_abs_error=None, n_classes=2,
    )
    skilled = _score_prediction(
        correct=False, confidence=0.5, brier=0.5,
        nll=0.2, depth_abs_error=None, n_classes=2,
    )
    assert skilled > guess
    # 45 direction + 25*(1-0.25) + 20*0 + 10*0.5 + 10 depth
    assert guess == pytest.approx(0.0 + 18.75 + 0.0 + 5.0 + 10.0)


def test_a_legacy_three_class_run_keeps_its_own_reference() -> None:
    import math

    from prosper.eval.predictions import _score_prediction

    nll = math.log(3)
    assert _score_prediction(
        correct=True, confidence=0.6, brier=0.4,
        nll=nll, depth_abs_error=None, n_classes=3,
    ) == pytest.approx(
        _score_prediction(
            correct=True, confidence=0.6, brier=0.4,
            nll=math.log(2), depth_abs_error=None, n_classes=2,
        )
    )
