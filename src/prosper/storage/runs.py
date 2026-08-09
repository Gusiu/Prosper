"""Discovery and loading of versioned prediction runs.

A prediction run is identified by ``(symbol, model_type, interval, timestamp)``
and lives in its own folder under ``reports/predictions/``. Everything derived
from predictions — action windows, stability reports, backtests — is keyed
by that identity, so a result can always be traced back to the exact model that
produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prosper.config import Settings, get_settings
from prosper.storage.layout import (
    parse_versioned_prediction_folder,
    resolve_versioned_prediction_dir,
    versioned_prediction_folder_name,
)


@dataclass(frozen=True)
class RunRef:
    """A single versioned prediction run."""

    symbol: str
    model_type: str
    interval: str
    timestamp: str
    directory: Path

    @property
    def slug(self) -> str:
        """Stable folder-safe identity, e.g. ``ml_1d_20240501123000``."""
        return versioned_prediction_folder_name(self.model_type, self.timestamp, self.interval)

    @property
    def predictions_path(self) -> Path:
        return self.directory / "predictions.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "interval": self.interval,
            "timestamp": self.timestamp,
            "slug": self.slug,
            "path": str(self.directory),
        }


def list_runs(
    symbol: str | None = None,
    model_type: str | None = None,
    interval: str | None = None,
    settings: Settings | None = None,
) -> list[RunRef]:
    """Return every prediction run matching the filters, newest first."""
    settings = settings or get_settings()
    base = settings.reports_predictions_dir
    if not base.exists():
        return []

    symbol_filter = symbol.upper() if symbol else None
    model_filter = model_type.lower() if model_type else None

    runs: list[RunRef] = []
    for symbol_dir in sorted(base.iterdir()):
        if not symbol_dir.is_dir() or symbol_dir.name.startswith("."):
            continue
        if symbol_filter and symbol_dir.name.upper() != symbol_filter:
            continue
        for run_dir in sorted(symbol_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            parsed = parse_versioned_prediction_folder(run_dir.name)
            if not parsed:
                continue
            parsed_model, parsed_interval, parsed_timestamp = parsed
            if model_filter and parsed_model.lower() != model_filter:
                continue
            if interval and parsed_interval != interval:
                continue
            if not (run_dir / "predictions.jsonl").exists():
                continue
            runs.append(
                RunRef(
                    symbol=symbol_dir.name,
                    model_type=parsed_model,
                    interval=parsed_interval,
                    timestamp=parsed_timestamp,
                    directory=run_dir,
                )
            )

    runs.sort(key=lambda run: (run.timestamp, run.symbol, run.model_type), reverse=True)
    return runs


def resolve_run(
    symbol: str,
    model_type: str | None = None,
    timestamp: str | None = None,
    interval: str | None = None,
    settings: Settings | None = None,
) -> RunRef:
    """Resolve one run, defaulting to the most recent match.

    Raises ``FileNotFoundError`` with the available candidates listed, because
    picking an arbitrary run silently is how results stop being attributable.
    """
    settings = settings or get_settings()

    if model_type and timestamp:
        directory = resolve_versioned_prediction_dir(
            symbol.upper(), model_type.lower(), timestamp, settings=settings, interval=interval
        )
        if not (directory / "predictions.jsonl").exists():
            raise FileNotFoundError(
                f"No predictions.jsonl in run {directory}. "
                f"Available: {_describe_available(symbol, settings)}"
            )
        parsed = parse_versioned_prediction_folder(directory.name)
        parsed_interval = parsed[1] if parsed else (interval or "1d")
        return RunRef(
            symbol=symbol.upper(),
            model_type=model_type.lower(),
            interval=interval or parsed_interval,
            timestamp=timestamp,
            directory=directory,
        )

    candidates = list_runs(
        symbol=symbol, model_type=model_type, interval=interval, settings=settings
    )
    if not candidates:
        raise FileNotFoundError(
            f"No prediction runs found for {symbol.upper()}"
            + (f" model={model_type}" if model_type else "")
            + (f" interval={interval}" if interval else "")
            + f". Available: {_describe_available(symbol, settings)}"
        )
    return candidates[0]


def _describe_available(symbol: str, settings: Settings) -> str:
    runs = list_runs(symbol=symbol, settings=settings)
    if not runs:
        return "none (run `prosper predict ...` first)"
    return ", ".join(run.slug for run in runs[:10])


def load_run_predictions(run: RunRef) -> list[dict[str, Any]]:
    """Read a run's predictions.jsonl, skipping malformed lines."""
    rows: list[dict[str, Any]] = []
    with open(run.predictions_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda row: str(row.get("open_time") or row.get("date") or ""))
    return rows
