"""Checks for every value a request can turn into a filesystem path.

A run is addressed by symbol, model, interval and timestamp, and all four end up
inside a path - one of which the delete endpoint hands to `shutil.rmtree` - so
all four are checked, not only the symbol. Unchecked,
`?interval=1d/../../../victim` walks out of the data lake through the query
string, and on Windows a `%5C` in the timestamp segment does the same through
the URL path, because a backslash is a separator there.

Each helper returns the value normalised the way the storage layer expects it,
or raises 400. Routes call them first, before anything touches the disk.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException

from prosper.domain import INTERVAL_MINUTES, MODEL_TYPES

SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,30}$")
# Run folders are stamped `%Y%m%d%H%M%S`; nothing else has ever been written.
RUN_TIMESTAMP_RE = re.compile(r"^\d{14}$")


def is_valid_symbol(symbol: str) -> bool:
    return bool(SYMBOL_RE.fullmatch(symbol))


def is_valid_interval(interval: str) -> bool:
    return interval in INTERVAL_MINUTES


def is_valid_model_type(model_type: str) -> bool:
    return model_type in MODEL_TYPES


def is_valid_run_timestamp(timestamp: str) -> bool:
    return bool(RUN_TIMESTAMP_RE.fullmatch(timestamp))


def require_symbol(symbol: str) -> str:
    normalised = symbol.upper()
    if not is_valid_symbol(normalised):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    return normalised


def require_model_type(model_type: str) -> str:
    normalised = model_type.lower()
    if not is_valid_model_type(normalised):
        raise HTTPException(status_code=400, detail="Invalid model type")
    return normalised


def require_interval(interval: str) -> str:
    if not is_valid_interval(interval):
        raise HTTPException(status_code=400, detail="Invalid interval")
    return interval


def require_run_timestamp(timestamp: str) -> str:
    if not is_valid_run_timestamp(timestamp):
        raise HTTPException(status_code=400, detail="Invalid run timestamp")
    return timestamp


def require_inside(path: Path, root: Path) -> Path:
    """Refuse *path* unless it resolves to somewhere under *root*.

    The checks above make an escape impossible to express; this one makes it
    impossible to act on, for any path built by code that has not been written
    yet. Deletion must never depend on every caller having validated its input.
    """
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise HTTPException(status_code=400, detail="Path outside the data lake")
    return resolved
