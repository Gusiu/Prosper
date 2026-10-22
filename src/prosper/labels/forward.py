"""The label definition, in one place.

A label says what the market did over a horizon: the side of the forward return
and the depth bin of its magnitude. Every model trains on it, evaluation scores
against it, and the dashboard draws it.

Every predictor and `prosper labels build` call it, so the models and the
label file cannot disagree about what a label is: separate copies of the loop
could agree today, and nothing would make them keep agreeing.

It is vectorised so the label builder can afford it on 1m data, and it is exact
rather than approximately equal to the per-bar loop: the arithmetic is the same
IEEE operations applied element-wise, and the predictors are pinned byte for
byte against the loop's output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prosper.domain import DIR_TO_IDX

UNKNOWN = -1


@dataclass(frozen=True)
class ForwardLabels:
    """Per-horizon arrays aligned to the input bars.

    ``direction`` holds a `DIR_TO_IDX` index and ``depth`` a bin index, both
    `UNKNOWN` (-1) where the label cannot be known: the horizon runs past the
    last bar, a price is missing or zero, or the return is exactly zero and so
    has no side. ``returns`` is NaN in exactly the same places, plus zero
    returns, which are real returns without a side.
    """

    direction: dict[str, np.ndarray]
    depth: dict[str, np.ndarray]
    returns: dict[str, np.ndarray]


def forward_returns(close: Sequence[float | None] | np.ndarray, forward: int) -> np.ndarray:
    """``close[i + forward] / close[i] - 1`` for every bar, NaN where unknowable."""
    prices = np.asarray(close, dtype=np.float64)
    n = len(prices)
    out = np.full(n, np.nan)
    if forward < 1 or forward >= n:
        return out
    base, ahead = prices[: n - forward], prices[forward:]
    usable = np.isfinite(base) & np.isfinite(ahead) & (base != 0.0) & (ahead != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        change = ahead / base - 1.0
    out[: n - forward] = np.where(usable, change, np.nan)
    return out


def depth_bin_indices(
    values_pct: np.ndarray, bins: Sequence[tuple[float, float | None]]
) -> np.ndarray:
    """Element-wise `domain.assign_depth_bin`: first bin with ``lo <= v < hi``.

    Bins must be ordered by their lower edge, which every scheme the project has
    used is. NaN matches no bin, as in the scalar version.
    """
    lows = np.array([lo for lo, _ in bins], dtype=np.float64)
    highs = np.array([np.inf if hi is None else hi for _, hi in bins], dtype=np.float64)
    if np.any(np.diff(lows) < 0):
        raise ValueError("depth bins must be ordered by their lower edge")

    index = np.searchsorted(lows, values_pct, side="right") - 1
    safe = np.clip(index, 0, len(bins) - 1)
    inside = (index >= 0) & (values_pct >= lows[safe]) & (values_pct < highs[safe])
    # An open-ended top bin has no upper edge to fail, and `inf < inf` is False.
    inside |= (index == len(bins) - 1) & (values_pct >= lows[-1]) & np.isinf(highs[-1])
    return np.where(inside, safe, UNKNOWN).astype(np.int64)


def forward_labels(
    close: Sequence[float | None] | np.ndarray,
    horizon_steps: Mapping[str, int],
    depth_bins: Sequence[tuple[float, float | None]],
) -> ForwardLabels:
    """Direction and depth labels for every horizon, in bars of the given series.

    *horizon_steps* maps a horizon name to its span in bars of this series -
    `HorizonSpec.steps(interval)` - so the same calendar horizon means the same
    thing at every interval. Labels are computed inside one series: pooling
    symbols never lets one symbol's price path decide another's outcome.
    """
    direction: dict[str, np.ndarray] = {}
    depth: dict[str, np.ndarray] = {}
    returns: dict[str, np.ndarray] = {}
    long_idx, short_idx = DIR_TO_IDX["long"], DIR_TO_IDX["short"]

    for name, forward in horizon_steps.items():
        change = forward_returns(close, forward)
        side = np.where(change > 0.0, long_idx, np.where(change < 0.0, short_idx, UNKNOWN))
        side = side.astype(np.int64)
        with np.errstate(invalid="ignore"):
            magnitude = np.abs(change) * 100.0
        bins = np.where(side != UNKNOWN, depth_bin_indices(magnitude, depth_bins), UNKNOWN)
        direction[name] = side
        depth[name] = bins.astype(np.int64)
        returns[name] = change

    return ForwardLabels(direction=direction, depth=depth, returns=returns)
