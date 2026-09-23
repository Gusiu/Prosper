"""Retraining windows that raised, counted instead of only printed.

A predictor that catches an exception while fitting or predicting one window
does the honest thing for the rows - they become untrained, and evaluation
excludes them (invariant 3). What it did not do was say so anywhere but the
console. A run where half the months hit an exception then looked exactly like
a run with a long horizon and a short history: more untrained rows, metrics on
fewer samples, and nothing in the artifact to tell the two apart.

That is the failure mode behind this project's worst defect, where two horizons
emitted a constant for months and the evaluation scored it. So the count now
goes into `run_summary.json`, next to everything else the run says about itself.
"""

from __future__ import annotations

import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Enough to see what went wrong without turning the summary into a log.
KEPT_MESSAGES = 5


@dataclass
class WindowFailures:
    events: list[dict[str, str]] = field(default_factory=list)

    def record(
        self, stage: str, horizon: str, window: str, error: BaseException,
        *, with_traceback: bool = False,
    ) -> None:
        message = f"{type(error).__name__}: {error}"
        self.events.append(
            {"stage": stage, "horizon": horizon, "window": window, "error": message}
        )
        print(f"[{stage}] {horizon} {window} failed, rows become untrained: {message}")
        if with_traceback:
            print(traceback.format_exc())

    def __len__(self) -> int:
        return len(self.events)

    def summary(self) -> dict[str, Any]:
        return {
            "count": len(self.events),
            "by_horizon": dict(Counter(event["horizon"] for event in self.events)),
            "by_stage": dict(Counter(event["stage"] for event in self.events)),
            "first": self.events[:KEPT_MESSAGES],
        }

    def warn_if_any(self) -> None:
        if self.events:
            print(
                f"WARNING: {len(self.events)} retraining window(s) failed and were written as "
                f"untrained: {self.summary()['by_horizon']}. See failed_windows in "
                "run_summary.json."
            )
