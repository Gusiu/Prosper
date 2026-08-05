"""Probability calibration for walk-forward direction forecasts.

The models are confident far beyond their accuracy. On the recomputed runs the
median |edge| was 0.996 while directional accuracy sat near 0.5, and expected
calibration error ran 0.40-0.46. Raw probabilities that extreme make every
downstream threshold meaningless: the planner's seven-point scale collapses to
`Strong Buy`/`Strong Sell` because |edge| clears 0.3 on almost every bar.

Calibration is fitted the same way everything else in this package is: on data
that was already realised at the moment of the forecast. Splitting the
*training* window rather than holding out later bars is what keeps invariant 6
intact — the calibrator never sees a label whose forward close lies at or after
the bar being predicted.

    |<------------- trainable region (training_bounds) ------------->|
    |<------- fit the model ------->|<---- fit the calibrator ------>|
    train_start                   split                        train_end   ... i

Both halves end before ``train_end``, which `training_bounds` has already
pulled back by ``forward_steps`` from the prediction bar.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from prosper.domain import DIRECTION_CLASSES
from prosper.predict.defaults import MIN_EARLY_STOPPING_SAMPLES

# Below this many held-out rows a calibrator fits noise, so the raw
# probabilities are passed through unchanged instead.
MIN_CALIBRATION_SAMPLES = 40

# Share of the trainable region reserved for fitting the calibrator. The model
# keeps the older, larger part; the calibrator gets the most recent rows, which
# is also the regime closest to the bar being predicted.
CALIBRATION_FRACTION = 0.25


class SupportsPredictProba(Protocol):
    def predict_proba(self, X: Any) -> Any: ...


def calibration_split(train_start: int, train_end: int) -> tuple[int, int]:
    """Split a trainable region into (model_end, calibration_start).

    Returns ``(train_end, train_end)`` when the region is too small to give the
    calibrator a usable slice, which the caller reads as "do not calibrate".
    """
    span = train_end - train_start
    if span <= 0:
        return train_end, train_end

    held_out = int(span * CALIBRATION_FRACTION)
    if held_out < MIN_CALIBRATION_SAMPLES:
        return train_end, train_end

    split = train_end - held_out
    if split <= train_start:
        return train_end, train_end
    return split, split


class TemperatureCalibrator:
    """Single-parameter (temperature) scaling of a probability vector.

    Temperature scaling is the conservative choice here. It has one degree of
    freedom, so it cannot overfit a few hundred held-out rows the way isotonic
    regression does, and it is monotone: it never reorders the classes, so a
    calibrated run still ranks bars the same way. All it does is move mass
    toward or away from the uniform prior, which is exactly the defect being
    corrected.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    @property
    def is_identity(self) -> bool:
        return abs(self.temperature - 1.0) < 1e-9

    def apply(self, probs: Sequence[float]) -> list[float]:
        """Rescale one probability vector. Order and support are preserved."""
        vector = np.asarray(list(probs), dtype=float)
        total = vector.sum()
        if not np.isfinite(total) or total <= 0:
            return [1.0 / len(vector)] * len(vector)
        vector = vector / total
        if self.is_identity:
            return [float(v) for v in vector]

        # Work in log space so the temperature acts on the logits that produced
        # these probabilities. Clipping keeps log() finite for a zeroed class.
        logits = np.log(np.clip(vector, 1e-12, 1.0)) / self.temperature
        logits -= logits.max()
        exp = np.exp(logits)
        scaled = exp / exp.sum()
        return [float(v) for v in scaled]

    def apply_to_mapping(self, probs: dict[str, float]) -> dict[str, float]:
        names = list(probs)
        values = self.apply([probs[name] for name in names])
        return dict(zip(names, values, strict=True))


def _nll(probs: np.ndarray, y_true: np.ndarray, temperature: float) -> float:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    scaled = exp / exp.sum(axis=1, keepdims=True)
    picked = scaled[np.arange(len(y_true)), y_true]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid: Sequence[float] | None = None,
) -> TemperatureCalibrator:
    """Pick the temperature minimising held-out negative log-likelihood.

    A grid search rather than an optimiser: the objective is one-dimensional
    and smooth, the grid is cheap at this size, and it cannot fail to converge
    part-way through a walk-forward run.
    """
    if len(y_true) < MIN_CALIBRATION_SAMPLES:
        return TemperatureCalibrator(1.0)

    # A held-out slice containing one class cannot say anything about
    # confidence, and minimising NLL on it is actively harmful: the model is
    # "right" on every row, so the loss falls monotonically as the temperature
    # drops and the fit sharpens an already overconfident forecast to certainty.
    # This is not hypothetical — on BTCUSDT the 365-day horizon has long
    # stretches where every forward return shares a sign, and the first version
    # of this function left the long horizon at |edge| = 1.000 for every bar.
    if len(np.unique(np.asarray(y_true))) < 2:
        return TemperatureCalibrator(1.0)

    candidates = list(grid) if grid is not None else [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75,
        2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0,
    ]
    best_t, best_loss = 1.0, math.inf
    for t in candidates:
        loss = _nll(probs, y_true, t)
        if loss < best_loss:
            best_t, best_loss = t, loss
    return TemperatureCalibrator(best_t)


def fit_calibrator_from_model(
    model: SupportsPredictProba | None,
    X_holdout: np.ndarray,
    y_holdout: Sequence[int],
    present_classes: Sequence[int] | None = None,
    n_classes: int = len(DIRECTION_CLASSES),
) -> TemperatureCalibrator:
    """Fit a calibrator on a model's own held-out predictions.

    *present_classes* names the class each `predict_proba` column belongs to,
    for models trained on a window where one direction never occurred.
    """
    if model is None or len(y_holdout) < MIN_CALIBRATION_SAMPLES:
        return TemperatureCalibrator(1.0)

    try:
        raw = np.asarray(model.predict_proba(X_holdout), dtype=float)
    except Exception:
        # A calibrator is an improvement, never a prerequisite: if the model
        # cannot score the held-out slice, the run continues uncalibrated
        # rather than losing the horizon.
        return TemperatureCalibrator(1.0)

    if raw.ndim != 2 or len(raw) != len(y_holdout):
        return TemperatureCalibrator(1.0)

    dense = np.full((len(raw), n_classes), 1e-12, dtype=float)
    columns = list(present_classes) if present_classes is not None else list(range(raw.shape[1]))
    for column, class_idx in enumerate(columns):
        if 0 <= int(class_idx) < n_classes and column < raw.shape[1]:
            dense[:, int(class_idx)] = raw[:, column]
    dense /= dense.sum(axis=1, keepdims=True)

    return fit_temperature(dense, np.asarray(list(y_holdout), dtype=int))


def usable_split(
    labels: Sequence[int],
    default_split: int,
    min_holdout: int = MIN_EARLY_STOPPING_SAMPLES,
) -> int | None:
    """A split point where both halves carry more than one class.

    `calibration_split` divides by position alone, which is right for the
    look-ahead rule but blind to what the labels contain. When the older part
    turns out to be single-class — common at the long horizon, where a whole
    year of start dates can share one sign — the model cannot be fitted on it,
    and the caller's only recourse was to train on everything and drop the
    holdout, losing early stopping and calibration for that window.

    Often a nearby split works: on BTCUSDT, 5 of the 13 affected windows have a
    valid one, some with more than fifty candidates. This returns the candidate
    closest to *default_split*, keeping the held-out fraction near what was
    asked for, or ``None`` when no split satisfies both halves — in which case
    dropping the holdout really is the only option.

    A single-class *holdout* is deliberately rejected too. Cross-entropy over
    one class is minimised by predicting it with certainty, so stopping on it
    would reward exactly the overconfidence the calibrator exists to undo.
    """
    total = len(labels)
    if total <= 0 or min_holdout <= 0:
        return None

    candidates = [
        split
        for split in range(1, total - min_holdout + 1)
        if len(set(labels[:split])) > 1 and len(set(labels[split:])) > 1
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda split: (abs(split - default_split), split))
