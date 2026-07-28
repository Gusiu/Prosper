"""Walk-forward training-window arithmetic shared by every predictor.

Every model in :mod:`prosper.predict` follows the same protocol: at bar ``i``
it may only train on labels that were already realised at that point in time.
A label at bar ``k`` for a horizon spanning ``forward_steps`` bars needs the
close of bar ``k + forward_steps``, so it is only usable once
``k + forward_steps < i``.

That constraint interacts with the training window: a window of ``W`` bars
yields ``W - forward_steps`` usable labels. When the window is not clearly
wider than the horizon the training set silently collapses to nothing, which
is why :func:`validate_train_window` refuses such a configuration up front
instead of letting a model emit an untrained placeholder for every row.
"""

from __future__ import annotations

from collections.abc import Iterable

from prosper.domain import DEPTH_BIN_LABELS, HorizonSpec, days_to_steps

# Below this many leakage-free labels a fitted classifier is noise, not a model.
MIN_TRAIN_SAMPLES = 30


def training_bounds(
    row_index: int,
    train_window_steps: int,
    forward_steps: int,
) -> tuple[int, int]:
    """Return ``(start, end)`` of the leakage-free training slice for *row_index*.

    ``end`` is exclusive. The last included label sits at ``end - 1`` and needs
    the close of bar ``row_index - 1``, which is known when predicting bar
    ``row_index``. An empty slice is signalled by ``end <= start``.
    """
    start = max(0, row_index - train_window_steps)
    end = row_index - forward_steps
    return start, end


def usable_sample_count(
    row_index: int,
    train_window_steps: int,
    forward_steps: int,
) -> int:
    """Number of leakage-free labels available when predicting *row_index*."""
    start, end = training_bounds(row_index, train_window_steps, forward_steps)
    return max(0, end - start)


def validate_train_window(
    train_window_days: int,
    interval: str,
    horizons: Iterable[HorizonSpec],
    min_samples: int = MIN_TRAIN_SAMPLES,
) -> dict[str, int]:
    """Check the window can train every horizon; return per-horizon step counts.

    Raises ``ValueError`` naming the horizons that cannot be trained and the
    smallest window that would work. Failing here is deliberate: the
    alternative is a run that completes successfully while emitting a constant
    placeholder distribution for the affected horizons.
    """
    horizons = list(horizons)
    window_steps = days_to_steps(train_window_days, interval)

    steps_by_horizon = {h.name: h.steps(interval) for h in horizons}
    too_short = [
        h for h in horizons if window_steps - steps_by_horizon[h.name] < min_samples
    ]
    if too_short:
        widest = max(steps_by_horizon[h.name] for h in too_short)
        required_days = max(h.forward_days for h in too_short) + _steps_to_days(
            min_samples, interval
        )
        names = ", ".join(f"{h.name} ({h.forward_days}d)" for h in too_short)
        raise ValueError(
            f"train_window_days={train_window_days} is too small at interval {interval}: "
            f"it spans {window_steps} bars, but horizon(s) {names} need "
            f"{widest} bars of lead time plus at least {min_samples} training samples. "
            f"Use --train-window-days {int(required_days) + 1} or higher, "
            f"or restrict the run to shorter horizons."
        )

    return steps_by_horizon


def _steps_to_days(steps: int, interval: str) -> float:
    from prosper.domain import MINUTES_PER_DAY, interval_minutes

    return steps * interval_minutes(interval) / MINUTES_PER_DAY


def untrained_horizon_payload(
    depth_labels: Iterable[str] = DEPTH_BIN_LABELS,
) -> dict[str, object]:
    """Placeholder emitted when a horizon had no trainable data at this bar.

    Carries ``trained: False`` so evaluation can exclude it instead of scoring
    a uniform distribution as if it were a genuine forecast.
    """
    labels = list(depth_labels)
    uniform_depth = {label: 1.0 / len(labels) for label in labels}
    return {
        "P_long": 1.0 / 3.0,
        "P_flat": 1.0 / 3.0,
        "P_short": 1.0 / 3.0,
        "depth_long_bins": dict(uniform_depth),
        "depth_short_bins": dict(uniform_depth),
        "trained": False,
    }
