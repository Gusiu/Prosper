"""The background task queue and its single worker.

One worker by design: the pipeline steps mutate the same data lake, so running
two at once would interleave writes to the same parquet partitions.

The queue itself lives here rather than in the browser. It used to be a
JavaScript array, so reloading the page lost every pending job while the one in
flight kept running — the UI and the server disagreed about what was scheduled.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# A long training run emits hundreds of thousands of lines. The UI only ever
# renders the tail, so keep a bounded window in memory instead of the lot.
MAX_TASK_LOG_LINES = 5000

_ids = itertools.count(1)


@dataclass
class QueuedTask:
    """A unit of work: a display name and the argv chain to run for it."""

    name: str
    commands: list[list[str]]
    id: int = field(default_factory=lambda: next(_ids))
    queued_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def command_display(self) -> str:
        return " && ".join(" ".join(cmd) for cmd in self.commands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "queued_at": self.queued_at,
            "steps": len(self.commands),
            "command": self.command_display,
        }


class TaskRunner:
    """Serialises queued work onto a single worker thread."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.logs: list[str] = []
        self.dropped_log_lines: int = 0
        self.is_running: bool = False
        self.current: QueuedTask | None = None
        self.progress: dict[str, Any] = {
            "visible": False,
            "percent": 0.0,
            "label": "Idle",
            "stage": "idle",
        }
        self._queue: deque[QueuedTask] = deque()
        # Guards the queue and the current-task handoff. Log appends stay
        # unlocked: the worker is the only writer and readers tolerate a race.
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    # ── queue API ────────────────────────────────────────────────────────────

    def submit(self, name: str, commands: list[list[str]]) -> QueuedTask:
        """Add a task and make sure the worker is running."""
        task = QueuedTask(name=name, commands=commands)
        with self._lock:
            self._queue.append(task)
            self._ensure_worker()
        return task

    def cancel(self, task_id: int) -> bool:
        """Drop a *pending* task. The running one is not interrupted."""
        with self._lock:
            for task in self._queue:
                if task.id == task_id:
                    self._queue.remove(task)
                    return True
        return False

    def move(self, task_id: int, direction: int) -> bool:
        """Shift a pending task one slot earlier (-1) or later (+1)."""
        with self._lock:
            items = list(self._queue)
            index = next((i for i, t in enumerate(items) if t.id == task_id), None)
            target = None if index is None else index + direction
            if index is None or target is None or not 0 <= target < len(items):
                return False
            items[index], items[target] = items[target], items[index]
            self._queue = deque(items)
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pending = [task.to_dict() for task in self._queue]
        return {
            "is_running": self.is_running,
            "current": self.current.to_dict() if self.current else None,
            "pending": pending,
            "queued": len(pending),
            "progress": self.progress,
        }

    # ── worker ───────────────────────────────────────────────────────────────

    def _ensure_worker(self) -> None:
        """Start the worker if it is not already draining the queue.

        Caller must hold the lock.
        """
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._drain, daemon=True)
        self._worker.start()

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self.current = None
                    self.is_running = False
                    return
                task = self._queue.popleft()
                self.current = task
                self.is_running = True

            self.logs = [f"$ {task.command_display}"]
            self.dropped_log_lines = 0
            self.progress = {
                "visible": True,
                "percent": 0.0,
                "label": "Starting",
                "stage": "queued",
            }
            self._run_task(task)

    def _run_task(self, task: QueuedTask) -> None:
        try:
            total_commands = max(1, len(task.commands))
            for command_index, original_args in enumerate(task.commands):
                self._append_log(f"$ {' '.join(original_args)}")
                self._set_command_progress(command_index, total_commands, original_args)

                # Ensure we use the current virtualenv's Python
                run_args = list(original_args)
                if run_args[0] == "python":
                    run_args[0] = sys.executable

                self.process = subprocess.Popen(
                    run_args,
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                if self.process.stdout:
                    for line in iter(self.process.stdout.readline, ""):
                        if line:
                            clean_line = line.rstrip("\n")
                            self._append_log(clean_line)
                            self._update_progress_from_log(
                                clean_line, command_index, total_commands
                            )
                self.process.wait()
                self._append_log(f"Task finished with exit code {self.process.returncode}")
                self._set_command_progress(
                    command_index, total_commands, original_args, finished=True
                )
                if self.process.returncode != 0:
                    self.progress.update({"label": "Failed", "stage": "failed"})
                    break
        except Exception as e:
            self._append_log(f"Error executing task: {e}")
            self.progress.update({"label": "Failed", "stage": "failed"})
        finally:
            self.process = None
            if self.progress.get("stage") != "failed":
                self.progress.update({"percent": 100.0, "label": "Complete", "stage": "complete"})

    # ── logs ─────────────────────────────────────────────────────────────────

    def _append_log(self, line: str) -> None:
        """Append a line, discarding the oldest once the window is full.

        `dropped_log_lines` keeps client cursors absolute: readers page by a
        line number that survives trimming rather than by list position.
        """
        self.logs.append(line)
        overflow = len(self.logs) - MAX_TASK_LOG_LINES
        if overflow > 0:
            del self.logs[:overflow]
            self.dropped_log_lines += overflow

    def read_logs(self, start_idx: int) -> tuple[list[str], int]:
        """Return log lines from absolute *start_idx* and the next cursor."""
        offset = self.dropped_log_lines
        local_start = max(0, start_idx - offset)
        lines = self.logs[local_start:]
        return lines, offset + len(self.logs)

    # ── progress ─────────────────────────────────────────────────────────────

    def _set_command_progress(
        self, command_index: int, total_commands: int, args: list[str], finished: bool = False
    ) -> None:
        # Extract a human-readable command name + symbol
        cli_cmd = args[3] if len(args) > 3 else "task"

        symbol = ""
        for i, arg in enumerate(args):
            if arg == "--symbol" and i + 1 < len(args):
                symbol = args[i + 1]
                break

        label = f"{cli_cmd.title()} {symbol}" if symbol else cli_cmd.title()
        percent = ((command_index + (1 if finished else 0)) / total_commands) * 100.0
        self.progress.update(
            {
                "visible": True,
                "percent": round(percent, 1),
                "label": label,
                "stage": cli_cmd,
            }
        )

    def _update_progress_from_log(
        self, line: str, command_index: int, total_commands: int
    ) -> None:
        match = re.search(r"\[\s*(\d+(?:\.\d+)?)%\s*\]", line)
        if not match:
            return
        local_percent = min(100.0, max(0.0, float(match.group(1))))
        self._set_percent(((command_index + local_percent / 100.0) / total_commands) * 100.0)

    def _set_percent(self, percent: float) -> None:
        self.progress["visible"] = True
        self.progress["percent"] = round(min(100.0, max(0.0, percent)), 1)


runner = TaskRunner()
