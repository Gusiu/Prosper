"""Structured progress reporting for CLI tasks (JSON-based)."""
import json
from pathlib import Path
from typing import Any


class ProgressReporter:
    def __init__(self, path: Path):
        self.path = path
        self.state = {}

    def update(self, pct: float, stage: str = "", details: dict[str, Any] = None):
        self.state = {
            "progress": pct,
            "stage": stage,
            "details": details or {},
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f)

    def read(self):
        if not self.path.exists():
            return None
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)
