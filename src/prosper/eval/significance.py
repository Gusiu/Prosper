"""Confidence intervals for metrics computed on overlapping labels.

Every accuracy figure this project reports is a mean over rows whose labels
overlap. A label at bar `k` is the sign of the return over `[k, k + forward]`,
so the label at `k + 1` shares `forward - 1` days of that interval with it: at
the annual horizon, consecutive rows agree on 363 of 364 days. They are not two
observations, they are one observation seen twice.

That makes the usual error bar meaningless. A run scoring 0.565 over 2040 rows
looks like it has 2040 observations behind it; at the 28-day horizon it has
closer to 73, and at the annual horizon a 1249-row figure rests on about three.
Without saying so, "0.565" and "0.500" cannot be compared at all — which is
what every table in this project did until now.

The moving-block bootstrap resamples *blocks* of consecutive rows rather than
individual rows, so each resample preserves the local correlation the overlap
creates. The block length is the overlap length, `forward_steps`, because that
is exactly the distance at which two labels stop sharing anything.

The intervals this produces are wide. That is the finding, not a defect of the
method.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from prosper.domain import effective_samples

# Enough for a stable 95% percentile interval; the cost is linear and the inputs
# here are a few thousand rows at most.
DEFAULT_RESAMPLES = 2000

# Below this many non-overlapping blocks the resampling cannot reproduce the
# statistic it is describing. This started as the training-window floor of 3 and
# was raised on evidence: at 3.1 blocks, two of fifty-five intervals came out
# *excluding their own point estimate* — 0.339 with an interval of
# [0.066, 0.335] — because four blocks drawn and truncated to the original length
# cannot represent a series that holds only three.
#
# It is deliberately not the same number as `MIN_INDEPENDENT_OBSERVATIONS`. That
# one asks "can a model be fitted here at all", which needs very little; this
# asks "is an interval around the answer stable", which needs an order more.
# Sharing one constant was tidiness, and the measurement says they differ.
MIN_BLOCKS = 10


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap interval, or an honest refusal."""

    point: float
    low: float | None
    high: float | None
    block_size: int
    blocks: float
    samples: int
    resamples: int
    reason: str | None = None

    @property
    def estimable(self) -> bool:
        return self.low is not None and self.high is not None

    def excludes(self, reference: float) -> bool | None:
        """Whether *reference* lies outside the interval.

        `None` when the interval could not be estimated — which must not be read
        as "no difference", only as "this data cannot say".
        """
        if not self.estimable:
            return None
        return reference < self.low or reference > self.high

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "block_size": self.block_size,
            "effective_samples": round(self.blocks, 1),
            "samples": self.samples,
            "resamples": self.resamples,
            "reason": self.reason,
        }


def block_bootstrap(
    rows: np.ndarray | Sequence[float],
    block_size: int,
    statistic: Callable[[np.ndarray], float] | None = None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile interval for *statistic* under a moving-block bootstrap.

    *rows* is one row per scored bar, in chronological order — either a 1-D array
    of values or a 2-D array whose columns the statistic knows how to read (the
    calibration error needs confidence and correctness together).

    Blocks start at any position, not on a fixed grid: a fixed grid would let the
    interval depend on where the series happens to begin.
    """
    data = np.asarray(rows)
    if data.ndim == 0:
        data = data.reshape(1)
    n = len(data)
    compute = statistic if statistic is not None else (lambda block: float(np.mean(block)))

    length = max(1, int(block_size))
    blocks = effective_samples(n, length)
    point = compute(data) if n else math.nan

    if n == 0:
        return Interval(math.nan, None, None, length, 0.0, 0, resamples, "no scored rows")
    if blocks < MIN_BLOCKS:
        return Interval(
            point, None, None, length, blocks, n, resamples,
            f"only {blocks:.1f} independent blocks; an interval here would be "
            f"an artefact of resampling one observation",
        )

    rng = np.random.default_rng(seed)
    # Enough whole blocks to cover n, then truncate, so every resample has the
    # same length as the original series.
    per_resample = int(math.ceil(n / length))
    starts_high = n - length + 1
    estimates = np.empty(resamples, dtype=float)
    for i in range(resamples):
        starts = rng.integers(0, starts_high, size=per_resample)
        index = (starts[:, None] + np.arange(length)[None, :]).reshape(-1)[:n]
        estimates[i] = compute(data[index])

    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    # The resampling is supposed to describe the statistic it was built from. If
    # it cannot even bracket it, the block structure is too coarse for this
    # series and the interval would be an artefact, not a measurement. This
    # catches the failure directly rather than trusting the block count alone.
    if not low <= point <= high:
        return Interval(
            point, None, None, length, blocks, n, resamples,
            f"the resampled interval [{low:.3f}, {high:.3f}] does not contain the "
            f"estimate {point:.3f}; {blocks:.1f} blocks is too coarse to describe it",
        )
    return Interval(point, float(low), float(high), length, blocks, n, resamples)


def accuracy_interval(correct: Sequence[bool], block_size: int, **kwargs: Any) -> Interval:
    """Directional accuracy with an interval, to compare against a coin flip."""
    return block_bootstrap(np.asarray(correct, dtype=float), block_size, **kwargs)


def mean_interval(values: Sequence[float], block_size: int, **kwargs: Any) -> Interval:
    """Any per-row score averaged over bars — Brier, NLL, the composite."""
    return block_bootstrap(np.asarray(values, dtype=float), block_size, **kwargs)


def calibration_error_interval(
    confidence: Sequence[float],
    correct: Sequence[bool],
    block_size: int,
    bins: int = 10,
    **kwargs: Any,
) -> Interval:
    """Expected calibration error with an interval.

    ECE is not a mean of per-row values — it is a weighted sum over confidence
    bins — so it has to be recomputed inside every resample rather than averaged.
    """
    paired = np.column_stack(
        [np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)]
    )

    def ece(block: np.ndarray) -> float:
        conf, hit = block[:, 0], block[:, 1]
        edges = np.clip((conf * bins).astype(int), 0, bins - 1)
        total = len(block)
        error = 0.0
        for index in range(bins):
            mask = edges == index
            count = int(mask.sum())
            if not count:
                continue
            error += (count / total) * abs(hit[mask].mean() - conf[mask].mean())
        return error

    return block_bootstrap(paired, block_size, ece, **kwargs)
