"""Shared JSONL prediction writing utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_prediction_report_path, get_versioned_prediction_dir


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


def write_predictions_jsonl(
    predictions: list[dict[str, Any]],
    symbol: str,
    settings: Settings | None = None,
) -> None:
    """Write prediction rows as per-month JSONL files.

    Groups *predictions* by the ``YYYY-MM`` prefix of their ``date`` field and
    writes one ``.jsonl`` file per month (idempotent overwrite).
    """
    if settings is None:
        settings = get_settings()

    by_month: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        mk = row["date"][:7]
        by_month.setdefault(mk, []).append(row)

    for mk, rows in by_month.items():
        year = int(mk[:4])
        month = int(mk[5:7])
        path = get_prediction_report_path(symbol, year, month, settings=settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=_json_converter, sort_keys=True) + "\n")


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
