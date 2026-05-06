"""Shared JSONL prediction writing utilities."""

from __future__ import annotations

import json
from typing import Any

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_prediction_report_path


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
                f.write(json.dumps(r, sort_keys=True) + "\n")
