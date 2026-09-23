"""The walk-forward protocol, written once.

Every predictor does the same thing around its model: decide each horizon's
training window from the history available, walk the requested range bar by
bar, refit at the first bar of every calendar month on data that ends before
that bar, write one row per bar with an explicitly untrained payload wherever a
horizon could not be fitted, and record what the run did. That skeleton used to
be written out in full five times, so a rule added to one copy - the power
warning, the untrained counter, the empty-range error - reached the others one
at a time or not at all. The baseline, for one, wrote an empty run where the
other four refused to.

A model now supplies two things: what to fit at the start of a month, and what
to say about one bar. `walk_forward` owns the loop and `finish_run` the output.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from prosper.config import Settings
from prosper.domain import DEFAULT_HORIZONS, IDX_TO_DIR, HorizonSpec
from prosper.predict.defaults import parse_horizons, parse_window_overrides
from prosper.predict.window import (
    TrainingWindows,
    count_scored_bars,
    resolve_train_windows,
    untrained_horizon_payload,
    warn_low_power,
)
from prosper.storage.predictions import write_run_summary, write_versioned_predictions
from prosper.utils.time import parse_date

Month = tuple[int, int]


@dataclass
class RunPlan:
    """What a run covers and how each horizon is trained, fixed before the loop."""

    symbol: str
    interval: str
    start: str
    end: str
    horizons: list[HorizonSpec]
    selected: set[str]
    open_times: list[datetime]
    dates: list[date]
    windows: TrainingWindows
    scored_bars: dict[str, int]
    depth_labels: list[str]
    effective_symbols: Mapping[str, float] | None = None
    months: dict[Month, list[int]] = field(default_factory=dict)

    @property
    def steps(self) -> Mapping[str, int]:
        return self.windows.forward_steps


def plan_run(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    *,
    open_times: list[datetime],
    dates: list[date],
    depth_labels: list[str],
    train_window_days: int | None,
    train_window_by_horizon: str | None,
    horizons_selected: str | None,
    effective_symbols: Mapping[str, float] | None = None,
) -> RunPlan:
    """Resolve the windows for this history and warn if the run cannot be powered.

    After the data is loaded, never before: the window a horizon should get
    depends on how much history there is to divide between training it and
    scoring it.
    """
    horizons = list(DEFAULT_HORIZONS)
    windows = resolve_train_windows(
        train_window_days,
        interval,
        horizons,
        parse_window_overrides(train_window_by_horizon),
        history_bars=len(dates),
    )
    first, last = parse_date(start).date(), parse_date(end).date()
    scored_bars = count_scored_bars(dates, first, last, windows)
    warn_low_power(windows, scored_bars, effective_symbols)

    # Bars to predict, grouped by the month whose model serves them. Dates are
    # in order, so each month's bars are contiguous and months come in sequence.
    months: dict[Month, list[int]] = {}
    for index, day in enumerate(dates):
        if first <= day <= last:
            months.setdefault((day.year, day.month), []).append(index)

    return RunPlan(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        horizons=horizons,
        selected=set(parse_horizons(horizons_selected)),
        open_times=open_times,
        dates=dates,
        windows=windows,
        scored_bars=scored_bars,
        depth_labels=depth_labels,
        effective_symbols=effective_symbols,
        months=months,
    )


class Forecaster(Protocol):
    """A model, as the walk-forward loop sees it."""

    name: str

    def fit_month(self, plan: RunPlan, month: Month, bars: list[int]) -> None:
        """Refit for *month*, whose first bar is ``bars[0]``, on data before it.

        Only horizons in ``plan.selected`` are fitted; the rest are written as
        untrained without being asked about.
        """

    def predict(self, bar: int, horizon: str) -> dict[str, Any] | None:
        """One horizon's payload for *bar*, or ``None`` when it is untrained."""

    def summary(self) -> dict[str, Any]:
        """Anything the run should record about itself beyond the common keys."""

    def after_write(self, run_dir: Path) -> None:
        """Write any artifact of the model's own beside the predictions."""


def walk_forward(model: Forecaster, plan: RunPlan) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Every bar in the plan's range, in order, with its payload per horizon."""
    predictions: list[dict[str, Any]] = []
    untrained = {h.name: 0 for h in plan.horizons}

    for month, bars in plan.months.items():
        model.fit_month(plan, month, bars)
        for bar in bars:
            row: dict[str, Any] = {
                "open_time": plan.open_times[bar].isoformat(),
                "date": plan.dates[bar].isoformat(),
                "symbol": plan.symbol,
            }
            for horizon in plan.horizons:
                payload = (
                    model.predict(bar, horizon.name) if horizon.name in plan.selected else None
                )
                if payload is None:
                    row[horizon.name] = untrained_horizon_payload(plan.depth_labels)
                    untrained[horizon.name] += 1
                else:
                    row[horizon.name] = payload
            predictions.append(row)

    return predictions, untrained


def finish_run(
    model: Forecaster, plan: RunPlan, settings: Settings,
    predictions: list[dict[str, Any]], untrained: dict[str, int],
) -> dict[str, Any]:
    """Write the predictions and the run summary; the result the CLI prints."""
    if not predictions:
        return {
            "error": "No predictions generated for the requested date range",
            "symbol": plan.symbol,
        }

    run_dir = write_versioned_predictions(
        predictions, plan.symbol, model.name, settings, interval=plan.interval
    )
    model.after_write(run_dir)

    result = {
        "symbol": plan.symbol,
        "start": plan.start,
        "end": plan.end,
        "interval": plan.interval,
        "predictions": len(predictions),
        "untrained_horizons": untrained,
        "training_windows": plan.windows.describe(plan.scored_bars, plan.effective_symbols),
        **model.summary(),
        "run_dir": str(run_dir),
    }
    write_run_summary(run_dir, result, settings)
    return result


def run(model: Forecaster, plan: RunPlan, settings: Settings) -> dict[str, Any]:
    predictions, untrained = walk_forward(model, plan)
    return finish_run(model, plan, settings, predictions, untrained)


# ── pieces every model uses ─────────────────────────────────────────────────


def seed_run(settings: Settings, *, torch_seed: bool = False) -> int:
    """Seed Python's and numpy's generators, and torch's if asked; return the seed.

    Only when the caller asked for reproducibility (`--deterministic` or
    `--seed`). The returned seed - 42 when none was given - is what the models
    pass as their own `random_state`, and what `window_seed` derives each
    retraining window's stream from.
    """
    seed = settings.seed if settings.seed is not None else 42
    if not (settings.deterministic or settings.seed is not None):
        return seed

    random.seed(seed)
    np.random.seed(seed)
    if torch_seed:
        import torch

        torch.manual_seed(seed)
        if settings.deterministic:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
    mode = "Deterministic mode ON" if settings.deterministic else "Seed set"
    print(f"[research] {mode} (seed={seed})")
    return seed


def uniform_depth(labels: list[str]) -> dict[str, float]:
    return {label: 1.0 / len(labels) for label in labels}


def empirical_depth(
    directions: np.ndarray, depths: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float]]:
    """Realised depth-bin distribution per direction over a training window.

    Every model but the ML one forecasts depth this way - the share of past
    moves in each bin, conditioned on the side they went. Four copies of it
    existed, two in list-counting form and two as this function.
    """
    counts = {"long": [0] * len(labels), "short": [0] * len(labels)}
    for direction_idx, depth_idx in zip(directions, depths, strict=True):
        if depth_idx < 0:
            continue
        side = IDX_TO_DIR.get(int(direction_idx))
        if side in counts:
            counts[side][int(depth_idx)] += 1

    out: dict[str, dict[str, float]] = {}
    for side, bucket in counts.items():
        total = sum(bucket)
        out[side] = (
            uniform_depth(labels)
            if total == 0
            else {label: count / total for label, count in zip(labels, bucket, strict=True)}
        )
    return out
