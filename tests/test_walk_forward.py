"""The walk-forward loop, tested once instead of five times.

With a copy per predictor, each rule of the protocol would be only as good as
the copy it had reached. These tests drive the shared loop with a stub model
that records what it was asked.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prosper.config import Settings
from prosper.domain import DEPTH_BIN_LABELS, HORIZON_NAMES
from prosper.predict.walk_forward import (
    empirical_depth,
    finish_run,
    plan_run,
    seed_run,
    uniform_depth,
    walk_forward,
)

START = datetime(2020, 1, 1, tzinfo=UTC)


class Stub:
    """Answers for every bar, except for horizons it is told to refuse."""

    name = "stub"

    def __init__(self, refuse: set[str] | None = None) -> None:
        self.refuse = refuse or set()
        self.months: list[tuple[tuple[int, int], int, int]] = []
        self.asked: list[tuple[int, str]] = []
        self.written_to: Path | None = None

    def fit_month(self, plan, month, bars):
        self.months.append((month, bars[0], len(bars)))

    def predict(self, bar, horizon):
        self.asked.append((bar, horizon))
        if horizon in self.refuse:
            return None
        return {"P_short": 0.4, "P_long": 0.6, "trained": True, "bar": bar}

    def summary(self) -> dict[str, Any]:
        return {"stub_note": "kept"}

    def after_write(self, run_dir: Path) -> None:
        self.written_to = run_dir


def _plan(days: int = 1600, start: str = "2023-11-15", end: str = "2024-02-10",
          horizons: str | None = None):
    times = [START + timedelta(days=i) for i in range(days)]
    return plan_run(
        "TESTUSDT", "1d", start, end,
        open_times=times,
        dates=[t.date() for t in times],
        depth_labels=list(DEPTH_BIN_LABELS),
        train_window_days=None,
        train_window_by_horizon=None,
        horizons_selected=horizons,
    )


def test_one_row_per_bar_in_range_in_order() -> None:
    plan = _plan()
    rows, _ = walk_forward(Stub(), plan)

    dates = [row["date"] for row in rows]
    assert dates[0] == "2023-11-15" and dates[-1] == "2024-02-10"
    assert dates == sorted(dates) and len(dates) == len(set(dates)) == 88
    assert list(rows[0])[:3] == ["open_time", "date", "symbol"]
    assert list(rows[0])[3:] == list(HORIZON_NAMES)


def test_the_model_refits_once_per_month_at_its_first_bar() -> None:
    plan = _plan()
    stub = Stub()
    walk_forward(stub, plan)

    assert [month for month, _, _ in stub.months] == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
    # The first refit happens at the first bar in range, not the first of the month.
    assert plan.dates[stub.months[0][1]].isoformat() == "2023-11-15"
    assert [count for _, _, count in stub.months] == [16, 31, 31, 10]


def test_a_refused_horizon_is_written_untrained_and_counted() -> None:
    plan = _plan()
    rows, untrained = walk_forward(Stub(refuse={"year"}), plan)

    assert all(row["year"]["trained"] is False for row in rows)
    assert all(row["week"]["trained"] is True for row in rows)
    assert untrained == {"week": 0, "month": 0, "quarter": 0, "year": len(rows)}


def test_an_unselected_horizon_is_never_asked() -> None:
    plan = _plan(horizons="week,month")
    stub = Stub()
    rows, untrained = walk_forward(stub, plan)

    assert {horizon for _, horizon in stub.asked} == {"week", "month"}
    assert untrained["quarter"] == untrained["year"] == len(rows)


def test_selecting_horizons_does_not_change_the_windows() -> None:
    """--horizons chooses what is fitted, never what the others are fitted on."""
    full, partial = _plan(), _plan(horizons="month")
    assert full.windows == partial.windows


def test_an_empty_range_is_refused_rather_than_written(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    plan = _plan(start="2030-01-01", end="2030-02-01")
    rows, untrained = walk_forward(Stub(), plan)

    result = finish_run(Stub(), plan, settings, rows, untrained)
    assert "error" in result
    assert not (tmp_path / "reports").exists()


def test_a_run_is_written_with_its_summary(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    plan = _plan()
    stub = Stub(refuse={"year"})
    rows, untrained = walk_forward(stub, plan)

    result = finish_run(stub, plan, settings, rows, untrained)
    run_dir = Path(result["run_dir"])
    assert stub.written_to == run_dir
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(rows)

    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["stub_note"] == "kept"
    assert summary["untrained_horizons"]["year"] == len(rows)
    assert summary["model_type"] == "stub" and summary["interval"] == "1d"


def test_empirical_depth_counts_each_side_and_falls_back_to_uniform() -> None:
    labels = ["0-1", "1-2", "2-3"]
    directions = np.array([1, 1, 1, 0, -1], dtype=np.int64)  # long, long, long, short, unknown
    depths = np.array([0, 0, 2, 1, -1], dtype=np.int64)

    out = empirical_depth(directions, depths, labels)
    assert out["long"] == pytest.approx({"0-1": 2 / 3, "1-2": 0.0, "2-3": 1 / 3})
    assert out["short"] == {"0-1": 0.0, "1-2": 1.0, "2-3": 0.0}

    none_short = empirical_depth(np.array([1]), np.array([0]), labels)
    assert none_short["short"] == uniform_depth(labels)


def test_seeding_only_happens_when_asked(tmp_path) -> None:
    import random

    random.seed(123)
    expected = random.random()
    random.seed(123)
    assert seed_run(Settings(data_root=tmp_path)) == 42
    assert random.random() == expected, "an unseeded run must not reseed the generators"

    seed_run(Settings(data_root=tmp_path, seed=7))
    first = random.random()
    seed_run(Settings(data_root=tmp_path, seed=7))
    assert random.random() == first
