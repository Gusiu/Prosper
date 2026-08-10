"""Walk-forward training-window arithmetic shared by every predictor.

Every model in :mod:`prosper.predict` follows the same protocol: at bar ``i``
it may only train on labels that were already realised at that point in time.
A label at bar ``k`` for a horizon spanning ``forward_steps`` bars needs the
close of bar ``k + forward_steps``, so it is only usable once
``k + forward_steps < i``.

That constraint interacts with the training window: a window of ``W`` bars
yields ``W - forward_steps`` usable labels. When the window is not clearly
wider than the horizon the training set silently collapses to nothing, which
is why :func:`resolve_train_windows` refuses such a configuration up front
instead of letting a model emit an untrained placeholder for every row.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

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

# How many labelled rows a window aims to hold, over and above the horizon's own
# lead time. Unlike everything else here this is a judgement, not arithmetic:
# 33 features and a GRU reading 30-bar sequences want rows in the hundreds, and
# 730 is what the project trained on for its whole history, so a rule that keeps
# the short horizons where they were is one fewer thing changing at once.
MIN_TRAINING_ROWS = 730


@dataclass(frozen=True)
class TrainingWindows:
    """How far back each horizon trains and how far forward it looks.

    The two used to be one number and one dict, because every horizon shared a
    window. They cannot share one any more: a window wide enough to hold a few
    independent annual observations costs the weekly horizon most of the bars it
    would otherwise be scored on, and the weekly horizon has no such problem to
    fix. Keeping both spans on one object means a caller cannot pair the window
    of one horizon with the forward span of another — which is the mistake that
    would silently train on labels the model could not have known.
    """

    forward_steps: Mapping[str, int]
    window_steps: Mapping[str, int]
    window_days: Mapping[str, int]

    def bounds(self, row_index: int, horizon: str) -> tuple[int, int]:
        """The leakage-free training slice for *horizon* when predicting *row_index*."""
        return training_bounds(
            row_index, self.window_steps[horizon], self.forward_steps[horizon]
        )

    def describe(
        self,
        scored_bars: Mapping[str, int] | None = None,
        effective_symbols: Mapping[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """The windows this run chose, and what they are worth, for the summary.

        A derived setting that is not written down is a setting nobody can check.
        The width now follows from the horizon and the history, so two runs on
        different symbols legitimately differ — and without this, the only way to
        find out what a run actually trained on would be to re-derive it and hope
        the rule had not changed in between.
        """
        power = training_window_power(self, effective_symbols)
        scored = scored_power(self, scored_bars) if scored_bars is not None else {}
        return {
            name: {
                "window_days": self.window_days[name],
                "window_bars": self.window_steps[name],
                "forward_bars": forward,
                "train_observations": round(power[name], 2),
                **(
                    {"scored_observations": round(scored[name], 2)}
                    if name in scored
                    else {}
                ),
            }
            for name, forward in self.forward_steps.items()
        }

    @property
    def widest_window_steps(self) -> int:
        """The longest window any horizon uses.

        Feature normalisation is fitted once per bar rather than once per
        horizon, so it needs a span covering every horizon's training slice.
        Still causal: everything it reads sits strictly before the predicted bar.
        """
        return max(self.window_steps.values(), default=0)


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


def dynamic_window_steps(forward_steps: int, history_steps: int | None = None) -> int:
    """How wide a window a horizon spanning *forward_steps* bars should get.

    Two requirements collide here, and which one binds depends on the horizon.

    A model needs **rows**: a few hundred labelled bars, or fitting 33 features is
    curve-drawing. A window holds ``W - forward_steps`` of them, so at a 7-bar
    horizon almost the whole window counts and rows are cheap.

    Statistics needs **independent observations**: consecutive labels share all
    but one bar of their forward return, so ``W`` bars are worth
    ``(W - forward_steps) / forward_steps`` of them. At a 364-bar horizon a
    two-year window is worth *one*, and one observation cannot support a number
    quoted to three decimal places.

    Hence ``forward + max(MIN_TRAINING_ROWS, MIN_INDEPENDENT_OBSERVATIONS x
    forward)``: the row floor decides the short horizons, the observation floor
    decides the long ones, and neither is ever the only thing checked. On daily
    bars that is 737 for a week, 758 for a month, 821 for a quarter and 1456 for
    a year.

    *history_steps* then applies the constraint the arithmetic above ignores:
    the history is finite, and every bar spent training is a bar that cannot be
    scored. When the ideal window would not leave enough scored bars, the window
    falls back to half the history — the split that maximises whichever side is
    weaker — and the caller is warned that neither side reaches the floor. That
    is not a fixable configuration; it is a horizon this much history cannot
    answer, which is a finding in itself.
    """
    ideal = forward_steps + max(MIN_TRAINING_ROWS, MIN_INDEPENDENT_OBSERVATIONS * forward_steps)
    if history_steps is None:
        return int(ideal)

    # The widest window still leaving enough scored bars for an interval.
    affordable = history_steps - forward_steps * (1 + MIN_INDEPENDENT_OBSERVATIONS)
    if ideal <= affordable:
        return int(ideal)

    floor = forward_steps + MIN_TRAIN_SAMPLES
    return int(max(floor, min(ideal, history_steps // 2)))


def resolve_train_windows(
    train_window_days: int | None,
    interval: str,
    horizons: Iterable[HorizonSpec],
    overrides: Mapping[str, int] | None = None,
    min_samples: int = MIN_TRAIN_SAMPLES,
    history_bars: int | None = None,
) -> TrainingWindows:
    """Work out each horizon's training window and check it can train at all.

    Each horizon is sized by :func:`dynamic_window_steps` unless *overrides*
    names it, or *train_window_days* fixes one width for every horizon. An
    explicit setting always wins, including when it makes the window narrower —
    a caller reproducing an older run must be able to ask for what that run used.

    Raises ``ValueError`` naming the horizons that cannot be trained and the
    smallest window that would work. Failing here is deliberate: the alternative
    is a run that completes successfully while emitting a constant placeholder
    distribution for the affected horizons.

    It does not warn about statistical power, because the useful warning needs
    the scored range too and that is only known once the data is loaded. Every
    predictor calls :func:`warn_low_power` there instead, and
    ``test_predict_leakage`` fails if one stops.
    """
    horizons = list(horizons)
    overrides = dict(overrides or {})

    forward_steps = {h.name: h.steps(interval) for h in horizons}
    window_steps: dict[str, int] = {}
    for h in horizons:
        if h.name in overrides:
            window_steps[h.name] = days_to_steps(int(overrides[h.name]), interval)
        elif train_window_days is not None:
            window_steps[h.name] = days_to_steps(int(train_window_days), interval)
        else:
            window_steps[h.name] = dynamic_window_steps(forward_steps[h.name], history_bars)
    window_days = {
        name: int(round(_steps_to_days(steps, interval))) for name, steps in window_steps.items()
    }

    too_short = [
        h for h in horizons if window_steps[h.name] - forward_steps[h.name] < min_samples
    ]
    if too_short:
        lines = []
        for h in too_short:
            required = h.forward_days + _steps_to_days(min_samples, interval)
            lines.append(
                f"{h.name} ({h.forward_days}d) has a {window_days[h.name]}d window "
                f"= {window_steps[h.name]} bars, needing {forward_steps[h.name]} of lead "
                f"time plus {min_samples} samples; use at least {int(required) + 1}d"
            )
        raise ValueError(
            f"The training window is too small at interval {interval}. "
            + "; ".join(lines)
            + ". Widen it with --train-window-days, set it for the horizon alone with "
            "--train-window-per-horizon, or restrict the run to shorter horizons."
        )

    return TrainingWindows(
        forward_steps=forward_steps, window_steps=window_steps, window_days=window_days
    )


def training_window_power(
    windows: TrainingWindows, effective_symbols: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Independent observations each horizon has inside its training window.

    A window holds `window_steps` bars, of which the last `forward_steps` have
    no realised label yet, so `window_steps - forward_steps` are labelled — and
    those overlap, which is what `effective_samples` divides out.

    At 730 days this is 103 observations for a 7-day horizon, 25.1 for 28 days
    and **1.0** for 364. One observation cannot fit or calibrate anything, and no
    model, optimiser setting or epoch budget changes that: it is arithmetic about
    how much non-overlapping history a two-year window contains.

    *effective_symbols* scales the count when several symbols are pooled — by
    `domain.effective_symbols`, not by the number of symbols. Three correlated
    crypto symbols multiply the annual count by 1.19, not by 3, and using the
    row count instead would report a horizon as measurable when it is not.
    """
    scale = dict(effective_symbols or {})
    return {
        name: effective_samples(max(0, windows.window_steps[name] - forward), forward)
        * max(1.0, scale.get(name, 1.0))
        for name, forward in windows.forward_steps.items()
    }


def count_scored_bars(
    dates: Sequence[date],
    start: date,
    end: date,
    windows: TrainingWindows,
) -> dict[str, int]:
    """Bars each horizon will actually be scored on.

    A bar counts when it falls in the requested range *and* its label resolves
    inside the data — the last `forward_steps` bars have no realised outcome, so
    they are predicted but never graded. `dates` must be sorted, as every
    predictor sorts it on load.
    """
    total = len(dates)
    in_range = [i for i, day in enumerate(dates) if start <= day <= end]
    return {
        name: sum(1 for i in in_range if i + forward < total)
        for name, forward in windows.forward_steps.items()
    }


def scored_power(windows: TrainingWindows, scored_bars: Mapping[str, int]) -> dict[str, float]:
    """Independent observations each horizon will be *scored* on.

    The other half of the same fixed supply. Widening a window to buy training
    observations takes them from here, one for one, because the history does not
    grow to meet the configuration — and nothing in a run reports that, which is
    exactly how a well-trained horizon ends up with an unmeasurable result.
    """
    return {
        name: effective_samples(scored_bars.get(name, 0), forward)
        for name, forward in windows.forward_steps.items()
    }


def warn_low_power(
    windows: TrainingWindows,
    scored_bars: Mapping[str, int] | None = None,
    effective_symbols: Mapping[str, float] | None = None,
) -> list[str]:
    """Warn about horizons the run cannot statistically support.

    The sibling of `resolve_train_windows`'s exception. That one refuses a
    configuration that cannot train at all; this one runs, but says so when the
    result will not be a measurement. It shipped for months without either, and
    the annual horizon was reported to three decimal places on the strength of
    one observation.

    Both sides are checked once *scored_bars* is known, because the window trades
    one against the other: widening it until the training side clears the floor
    pushes the scored side under it, and a warning about only the first would
    read as an invitation to do exactly that.

    Returns the messages as well as printing them, so a caller can surface them
    somewhere other than a console.
    """
    sides: list[tuple[str, dict[str, float]]] = [
        ("training window", training_window_power(windows, effective_symbols))
    ]
    if scored_bars is not None:
        sides.append(("scored range", scored_power(windows, scored_bars)))

    messages: list[str] = []
    for side, power in sides:
        messages.extend(
            f"horizon {name!r} has ~{observations:.1f} independent observations in the "
            f"{side} (needs {MIN_INDEPENDENT_OBSERVATIONS:.0f}); its metrics will not "
            f"be statistically meaningful"
            for name, observations in sorted(power.items(), key=lambda item: item[1])
            if observations < MIN_INDEPENDENT_OBSERVATIONS
        )
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
