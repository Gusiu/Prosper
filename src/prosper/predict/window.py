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

from prosper.domain import (
    DEPTH_BIN_LABELS,
    DIRECTION_CLASSES,
    MIN_INDEPENDENT_OBSERVATIONS,
    HorizonSpec,
    days_to_steps,
    effective_samples,
)

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

    warn_low_power(window_steps, steps_by_horizon)
    return steps_by_horizon


def training_window_power(
    window_steps: int, steps_by_horizon: dict[str, int]
) -> dict[str, float]:
    """Independent observations each horizon has inside the training window.

    The window holds `window_steps` bars, of which the last `forward_steps` have
    no realised label yet, so `window_steps - forward_steps` are labelled — and
    those overlap, which is what `effective_samples` divides out.

    At the shipped defaults this is 25.1 for a 28-day horizon, 7.0 for 91 days
    and **1.0** for 364. One observation cannot fit or calibrate anything, and no
    model, optimiser setting or epoch budget changes that: it is arithmetic about
    how much non-overlapping history a two-year window contains.
    """
    return {
        name: effective_samples(max(0, window_steps - forward), forward)
        for name, forward in steps_by_horizon.items()
    }


def warn_low_power(window_steps: int, steps_by_horizon: dict[str, int]) -> list[str]:
    """Warn about horizons the window cannot statistically support.

    The sibling of `validate_train_window`'s exception. That one refuses a
    configuration that cannot train at all; this one runs, but says so when the
    result will not be a measurement. `validate_train_window` shipped for months
    without it, and the annual horizon was reported to three decimal places on
    the strength of one observation.

    Returns the messages as well as printing them, so a caller can surface them
    somewhere other than a console.
    """
    power = training_window_power(window_steps, steps_by_horizon)
    messages = [
        f"horizon {name!r} has ~{observations:.1f} independent observations in the "
        f"training window (needs {MIN_INDEPENDENT_OBSERVATIONS:.0f}); its metrics "
        f"will not be statistically meaningful"
        for name, observations in sorted(power.items(), key=lambda item: item[1])
        if observations < MIN_INDEPENDENT_OBSERVATIONS
    ]
    for message in messages:
        print(f"[power] {message}")
    return messages


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
    uniform_direction = 1.0 / len(DIRECTION_CLASSES)
    payload: dict[str, object] = {
        f"P_{direction}": uniform_direction for direction in DIRECTION_CLASSES
    }
    payload.update(
        {
            "depth_long_bins": dict(uniform_depth),
            "depth_short_bins": dict(uniform_depth),
            "trained": False,
        }
    )
    return payload
