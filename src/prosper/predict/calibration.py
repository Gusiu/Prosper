"""Probability calibration for walk-forward direction forecasts.

The models are confident far beyond their accuracy. On the recomputed runs the
median |edge| was 0.996 while directional accuracy sat near 0.5, and expected
calibration error ran 0.40-0.46. Raw probabilities that extreme make every
downstream threshold meaningless: the planner's seven-point scale collapses to
`Strong Buy`/`Strong Sell` because |edge| clears 0.3 on almost every bar.

Calibration is fitted the same way everything else in this package is: on data
that was already realised at the moment of the forecast. Splitting the
*training* window rather than holding out later bars is what keeps invariant 6
intact - the calibrator never sees a label whose forward close lies at or after
the bar being predicted.

    |<--------------- trainable region (training_bounds) --------------->|
    |<----- fit the model ----->|<-- F -->|<---- fit the calibrator ---->|
    train_start              fit_end   calibration_start            train_end   ... i

Both parts end before ``train_end``, which `training_bounds` has already
pulled back by ``forward_steps`` from the prediction bar. The ``F`` bars
between them, one horizon, belong to neither: a label at bar ``k`` is the
return to ``close[k + F]``, so the model's newest labels would otherwise be
computed from closes inside the slice that is meant to measure it. See
`purged_holdout`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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

# Temperatures searched when fitting. One grid, shared by the fit and by the
# stopping criterion, so the two cannot disagree about what "calibrated" means.
# It reaches far above 10 because the raw models are near-certain: a slice that
# shows no skill asks for a temperature without bound, and one ending at 10
# left a 0.99 forecast at 0.61. At 1000 the same forecast is 0.501, and the
# direction - which a temperature never changes - is still defined.
TEMPERATURE_GRID: tuple[float, ...] = (
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75,
    2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0,
    15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0,
)


class SupportsPredictProba(Protocol):
    def predict_proba(self, X: Any) -> Any: ...


def calibration_split(train_start: int, train_end: int) -> tuple[int, int]:
    """Where the held-out slice of a trainable region begins, by position.

    Returns ``(calibration_start, calibration_start)``, or ``(train_end,
    train_end)`` when the region is too small to give the calibrator a usable
    slice, which the caller reads as "do not calibrate". It knows nothing of
    the horizon: `purged_holdout` builds on it and moves the end of the fitted
    rows back by one.
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


def purged_holdout(times: Sequence[Any] | np.ndarray, purge: Any) -> tuple[int, int]:
    """Positions ``(fit_end, calibration_start)`` dividing rows ordered by *times*.

    *times* is each row's place in time, ascending: a bar index for one series,
    an ``open_time`` for a pool, where several rows share an instant. The newest
    `CALIBRATION_FRACTION` of the rows is held out, from the first row of its
    instant so that no instant lies on both sides. The model fits only the rows
    more than *purge* - the horizon, in the units of *times* - older than the
    first held-out one.

    That gap is what makes the slice out of sample. A label at bar ``k`` is the
    return to ``close[k + F]``, so without it the model's newest ``F`` labels are
    computed from closes inside the slice, and the calibrator - and the stopping
    rule of the GRU and TFT - scores the model on outcomes it was fitted on. It
    then looks more skilful there than on any later bar and is calibrated, and
    stopped, accordingly. At the yearly horizon the slice is shorter than the
    horizon, so that was every held-out label. The gap is the separation
    `training_bounds` keeps from the bar being predicted.

    Returns ``(len(times), len(times))`` - fit everything, hold out nothing -
    when the region is too small for a slice or the gap would leave nothing to fit.
    """
    total = len(times)
    _, calibration_start = calibration_split(0, total)
    if calibration_start >= total:
        return total, total
    order = np.asarray(times)
    calibration_start = int(np.searchsorted(order, order[calibration_start], side="left"))
    fit_end = purged_end(order, calibration_start, purge)
    if fit_end <= 0:
        return total, total
    return fit_end, calibration_start


def purged_end(times: Sequence[Any] | np.ndarray, calibration_start: int, purge: Any) -> int:
    """Leading rows the model may fit when the held-out slice starts at *calibration_start*.

    Those whose time lies more than *purge* before the slice's first row; see
    `purged_holdout`.
    """
    order = np.asarray(times)
    return int(np.searchsorted(order, order[calibration_start] - purge, side="left"))


def choose_holdout(
    times: Sequence[Any] | np.ndarray, labels: Sequence[int], purge: Any
) -> tuple[int, int]:
    """``(fit_end, calibration_start)`` for one window, the same policy for every model.

    `purged_holdout` decides by time alone. When the rows it leaves to fit, or
    the slice it holds out, carry a single class, the nearest split where both
    carry two is taken instead (`usable_split`); when there is none, the window
    fits every row and holds nothing out. Every model goes through here, so no
    architecture keeps a calibrator that another loses on the same window: the
    tree models used to train on everything and drop it where the networks
    moved the split.
    """
    total = len(labels)
    fit_end, calibration_start = purged_holdout(times, purge)
    if calibration_start >= total:
        return total, total
    if len(set(labels[:fit_end])) > 1 and len(set(labels[calibration_start:])) > 1:
        return fit_end, calibration_start
    moved = usable_split(labels, calibration_start, MIN_CALIBRATION_SAMPLES, times=times,
                         purge=purge)
    if moved is None:
        return total, total
    return purged_end(times, moved, purge), moved


class TemperatureCalibrator:
    """Single-parameter (temperature) scaling of a probability vector.

    Temperature scaling is the conservative choice here. It has one degree of
    freedom, so it cannot overfit a few hundred held-out rows the way isotonic
    regression does, and it is monotone: it never reorders the classes, so a
    calibrated run still ranks bars the same way. All it does is move mass
    toward or away from the uniform prior, which is exactly the defect being
    corrected.
    """

    def __init__(
        self, temperature: float = 1.0, fitted_on: int = 0, source: str = "identity"
    ) -> None:
        self.temperature = float(temperature)
        # Held-out rows the temperature was chosen on; 0 when none was fitted
        # here, which a fitted 1.0 must not be mistaken for.
        self.fitted_on = int(fitted_on)
        # "fitted" on this window's slice, "carried" from an earlier window
        # of the horizon, "prior" before any was fitted - see TemperatureMemory.
        self.source = source

    @property
    def is_identity(self) -> bool:
        return abs(self.temperature - 1.0) < 1e-9

    def apply(self, probs: Sequence[float] | np.ndarray) -> list[float]:
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
    # This is not hypothetical - on BTCUSDT the 365-day horizon has long
    # stretches where every forward return shares a sign, and the first version
    # of this function left the long horizon at |edge| = 1.000 for every bar.
    if len(np.unique(np.asarray(y_true))) < 2:
        return TemperatureCalibrator(1.0)

    candidates = list(grid) if grid is not None else list(TEMPERATURE_GRID)
    best_t, best_loss = 1.0, math.inf
    for t in candidates:
        loss = _nll(probs, y_true, t)
        if loss < best_loss:
            best_t, best_loss = t, loss
    return TemperatureCalibrator(best_t, fitted_on=len(y_true), source="fitted")


class TemperatureMemory:
    """The newest fitted temperature of each horizon, for a window that could not fit one.

    A window holds nothing out when its slice would be too small or no split
    leaves both parts two classes (`choose_holdout`). The raw output of these
    models is near-certain, so such a window was the most confident of its run
    for no better reason than missing data. The temperature last fitted at the
    same horizon of the run, chosen on a slice realised before this window, is
    the best measurement of the model's reliability there is; before the first
    one a forecast is given no confidence beyond chance, the top of
    `TEMPERATURE_GRID`. The direction is the model's either way.
    """

    def __init__(self) -> None:
        self.latest: dict[str, float] = {}

    def settle(self, horizon: str, calibrator: TemperatureCalibrator) -> TemperatureCalibrator:
        """*calibrator* if it was fitted, remembered; otherwise what stands in for it."""
        if calibrator.fitted_on:
            self.latest[horizon] = calibrator.temperature
            return calibrator
        if horizon in self.latest:
            return TemperatureCalibrator(self.latest[horizon], source="carried")
        return TemperatureCalibrator(TEMPERATURE_GRID[-1], source="prior")


def calibration_fields(
    uncalibrated: Mapping[str, float], calibrator: TemperatureCalibrator
) -> dict[str, Any]:
    """What the calibrator did to one forecast, kept beside it in the payload.

    The probabilities before it was applied, the temperature and how many
    held-out rows chose it, so that how much calibration helps, and whether a
    temperature ran into the end of `TEMPERATURE_GRID`, can be measured from a
    run rather than argued. Nested and without the ``P_`` prefix, which
    readers of a payload take to mean a direction class.
    """
    total = sum(max(0.0, float(uncalibrated.get(name, 0.0))) for name in DIRECTION_CLASSES)
    return {
        "temperature": calibrator.temperature,
        "temperature_source": calibrator.source,
        "calibrated_on": calibrator.fitted_on,
        "uncalibrated": {
            name: (max(0.0, float(uncalibrated.get(name, 0.0))) / total if total > 0 else
                   1.0 / len(DIRECTION_CLASSES))
            for name in DIRECTION_CLASSES
        },
    }


def calibrated_nll(probs: np.ndarray, y_true: Sequence[int] | np.ndarray) -> float:
    """Held-out loss the forecast carries *once its temperature is fitted*.

    This is the quantity DL training stops on, and raw cross-entropy is the
    wrong one. A proper score decomposes into a calibration term - how far the
    stated confidence is from the observed frequency - and a refinement term,
    how well the forecast separates the classes at all. Training moves both:
    the network sharpens long before it stops learning to discriminate, so its
    raw NLL on the holdout climbs from the first epoch while accuracy is still
    rising. Measured on BTCUSDT the sequence ran 0.7244, 0.9109, 0.9282,
    1.0216, which stopped training after one epoch, while an epoch sweep showed
    the short horizon still gaining accuracy at 20 (0.510 -> 0.584).

    Minimising over the temperature grid removes the calibration term and
    leaves refinement. That is not an optimistic proxy for what ships: the
    model *is* temperature-scaled on this exact slice immediately afterwards,
    so this is the loss the stored artifact will actually carry. The fit is
    over one parameter on the same rows at every epoch, so the comparison
    between epochs is like for like.

    A single-class holdout cannot say anything about confidence and its NLL is
    driven to zero by an arbitrarily small temperature, so it falls back to the
    raw value; `usable_split` already tries to avoid handing one over.
    """
    y = np.asarray(list(y_true), dtype=int)
    if y.size == 0:
        return math.inf
    p = np.asarray(probs, dtype=float)
    if len(np.unique(y)) < 2:
        return _nll(p, y, 1.0)
    return min(_nll(p, y, t) for t in TEMPERATURE_GRID)


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
    *,
    times: Sequence[Any] | np.ndarray | None = None,
    purge: Any = 0,
) -> int | None:
    """A held-out slice start where both the fitted rows and the slice carry more than one class.

    `purged_holdout` divides by time alone, which is right for the look-ahead
    rule but blind to what the labels contain. When the older part turns out
    to be single-class - common at the long horizon, where a whole year of start
    dates can share one sign - the model cannot be fitted on it, and the
    caller's only recourse was to train on everything and drop the holdout,
    losing early stopping and calibration for that window.

    Often a nearby split works: on BTCUSDT, 5 of the 13 affected windows have a
    valid one, some with more than fifty candidates. This returns the candidate
    closest to *default_split*, keeping the held-out fraction near what was
    asked for, or ``None`` when no split satisfies both parts - in which case
    dropping the holdout really is the only option. Given *times* and *purge*,
    a candidate is judged on the rows the model would actually fit, those
    `purged_end` leaves before it; without them every row before it is fitted.

    A single-class *holdout* is deliberately rejected too. Cross-entropy over
    one class is minimised by predicting it with certainty, so stopping on it
    would reward exactly the overconfidence the calibrator exists to undo.
    """
    total = len(labels)
    if total <= 0 or min_holdout <= 0:
        return None
    order = np.arange(total) if times is None else np.asarray(times)

    candidates = [
        split
        for split in range(1, total - min_holdout + 1)
        if order[split - 1] < order[split]  # a pooled instant stays on one side
        and len(set(labels[: purged_end(order, split, purge)])) > 1
        and len(set(labels[split:])) > 1
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda split: (abs(split - default_split), split))
