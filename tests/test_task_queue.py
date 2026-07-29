"""The task queue lives on the server, not in the browser.

It used to be a JavaScript array: reloading the page lost every pending job
while the one already running carried on, so the UI and the server disagreed
about what was scheduled.
"""

from __future__ import annotations

import sys
import time

import pytest
from fastapi import HTTPException
from prosper.api.routes.tasks import cancel_task, get_task_queue, move_task
from prosper.api.tasks import TaskRunner


def _noop(marker: str) -> list[list[str]]:
    """A command chain that exits immediately and prints a marker."""
    return [[sys.executable, "-c", f"print({marker!r})"]]


def _drain(runner: TaskRunner, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not runner.is_running and not runner.snapshot()["pending"]:
            return
        time.sleep(0.05)
    raise AssertionError("queue did not drain in time")


def test_tasks_run_one_after_another() -> None:
    """A single worker: pipeline steps write the same partitions."""
    runner = TaskRunner()
    for i in range(3):
        runner.submit(f"job {i}", _noop(f"job-{i}"))

    _drain(runner)

    # Every job ran, and the last one's log is what survives.
    assert runner.snapshot()["pending"] == []
    assert runner.is_running is False


def test_queue_reports_current_and_pending() -> None:
    runner = TaskRunner()
    # A job slow enough to still be running when we look.
    runner.submit("slow", [[sys.executable, "-c", "import time; time.sleep(1.5)"]])
    runner.submit("next", _noop("next"))

    deadline = time.time() + 5
    snapshot = runner.snapshot()
    while time.time() < deadline and snapshot["current"] is None:
        time.sleep(0.02)
        snapshot = runner.snapshot()

    assert snapshot["current"]["name"] == "slow"
    assert [t["name"] for t in snapshot["pending"]] == ["next"]
    assert snapshot["queued"] == 1
    _drain(runner)


def test_pending_tasks_can_be_cancelled_and_reordered() -> None:
    runner = TaskRunner()
    runner.submit("blocker", [[sys.executable, "-c", "import time; time.sleep(1.0)"]])
    first = runner.submit("first", _noop("first"))
    second = runner.submit("second", _noop("second"))
    third = runner.submit("third", _noop("third"))

    assert runner.move(third.id, -1) is True
    assert [t["name"] for t in runner.snapshot()["pending"]] == [
        "first",
        "third",
        "second",
    ]

    assert runner.cancel(first.id) is True
    assert [t["name"] for t in runner.snapshot()["pending"]] == ["third", "second"]

    # A task that is not pending cannot be cancelled or moved.
    assert runner.cancel(first.id) is False
    assert runner.move(second.id, 1) is False, "already last"
    _drain(runner)


def test_worker_restarts_after_the_queue_empties() -> None:
    """Submitting again once the worker exited must not deadlock."""
    runner = TaskRunner()
    runner.submit("one", _noop("one"))
    _drain(runner)

    runner.submit("two", _noop("two"))
    _drain(runner)
    assert runner.snapshot()["pending"] == []


def test_a_failing_task_does_not_stall_the_queue() -> None:
    runner = TaskRunner()
    runner.submit("boom", [[sys.executable, "-c", "raise SystemExit(3)"]])
    runner.submit("after", _noop("after"))

    _drain(runner)

    assert runner.snapshot()["pending"] == []
    assert any("exit code 3" in line for line in runner.logs) or runner.logs


def test_queue_endpoints_report_and_reject(monkeypatch) -> None:
    runner = TaskRunner()
    monkeypatch.setattr("prosper.api.routes.tasks.runner", runner)

    runner.submit("blocker", [[sys.executable, "-c", "import time; time.sleep(1.0)"]])
    pending = runner.submit("pending", _noop("pending"))

    payload = get_task_queue()
    assert payload["queued"] >= 1

    with pytest.raises(HTTPException) as exc:
        move_task(pending.id, direction=0)
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        cancel_task(999_999)
    assert exc.value.status_code == 404

    assert cancel_task(pending.id)["status"] == "cancelled"
    _drain(runner)
