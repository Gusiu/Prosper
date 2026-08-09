"""Shared JSONL prediction writing utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_versioned_prediction_dir


def _json_converter(o: object) -> object:
    """Convert numpy and other non-serializable types to built-in types.

    This makes JSON dumping of predictions robust when values are numpy types
    (e.g., float32) coming from model outputs.
    """
    # numpy scalar
    if isinstance(o, np.floating | np.float32 | np.float64):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    # fallback for objects exposing item()
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if isinstance(o, set):
        return list(o)
    raise TypeError(f"Type {type(o)} not serializable")


def write_versioned_predictions(
    predictions: list[dict[str, Any]],
    symbol: str,
    model_type: str,
    settings: Settings | None = None,
    interval: str = "1d",
) -> Path:
    """Write a consolidated `predictions.jsonl` into a versioned folder.

    The folder structure is:
    `reports/predictions/{symbol}/{model_type}_{interval}_{timestamp}/predictions.jsonl`.
    """
    if settings is None:
        settings = get_settings()

    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    out_dir = get_versioned_prediction_dir(
        symbol, model_type, timestamp, settings=settings, interval=interval
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in predictions:
            f.write(json.dumps(r, default=_json_converter, sort_keys=True) + "\n")
    return out_dir


def write_run_summary(run_dir: Path, summary: dict[str, Any]) -> Path:
    """Persist a run's own description beside its predictions.

    Everything a predictor learns about its own execution used to die with the
    process: the result dict went to the caller, the CLI printed two lines of it
    and the rest was gone. `epochs_used` in particular is a *result* — the epoch
    count is decided by early stopping, not by the flag — and it was invisible
    the moment the command exited, so checking whether stopping fired at all
    needed a throwaway instrumented probe every time.

    Invariant 5 says every derived artifact names its source run; the same logic
    says a run should name its own configuration.
    """
    path = run_dir / "run_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_converter),
        encoding="utf-8",
    )
    return path
