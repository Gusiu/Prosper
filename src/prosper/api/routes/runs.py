"""Versioned prediction runs: listing, predictions, importances, deletion."""

import json
import shutil
from typing import Any

from fastapi import APIRouter, HTTPException

from prosper.api.commands import _is_valid_symbol
from prosper.config import get_settings
from prosper.storage.layout import (
    get_versioned_prediction_file,
    parse_versioned_prediction_folder,
    resolve_versioned_prediction_dir,
)
from prosper.storage.runs import list_runs

# A 1d run is ~2000 bars; a 1h run over the same span is 24x that. The cap
# is a ceiling on the response, not a paging scheme — narrow with
# year/month to walk a run.
MAX_PREDICTION_ROWS = 5000

router = APIRouter()


@router.get("/api/ai/models")
def get_ai_models() -> dict[str, list[dict[str, Any]]]:
    settings = get_settings()
    base = settings.reports_predictions_dir
    if not base.exists():
        return {"models": []}

    models: list[dict[str, Any]] = []
    for symbol_dir in sorted(base.iterdir() if base.exists() else []):
        if not symbol_dir.is_dir():
            continue
        for entry in sorted(symbol_dir.iterdir()):
            if not entry.is_dir() or "_" not in entry.name:
                continue
            parsed = parse_versioned_prediction_folder(entry.name)
            if not parsed:
                continue
            model_type, interval, timestamp = parsed
            has_fi = (entry / "feature_importances.json").exists()
            jsonl_count = sum(1 for _ in entry.rglob("*.jsonl"))
            models.append(
                {
                    "symbol": symbol_dir.name,
                    "model_type": model_type,
                    "interval": interval,
                    "timestamp": timestamp,
                    "path": str(entry),
                    "has_feature_importances": has_fi,
                    "predictions_files": jsonl_count,
                }
            )
    return {"models": models}


@router.get("/api/runs")
def list_prediction_runs(
    symbol: str | None = None,
    model_type: str | None = None,
    interval: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """List versioned prediction runs, newest first.

    Every downstream artifact (windows, stability, backtest) is keyed by one
    of these runs, so this is the canonical picker for the UI.
    """
    if symbol and not _is_valid_symbol(symbol.upper()):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    runs = list_runs(
        symbol=symbol, model_type=model_type, interval=interval, settings=get_settings()
    )
    return {"runs": [run.to_dict() for run in runs]}


@router.get("/api/ai/predictions")
def get_ai_predictions(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str = "1d",
    year: int | None = None,
    month: int | None = None,
    limit: int = MAX_PREDICTION_ROWS,
):
    """Return a run's predictions, optionally narrowed to one calendar month.

    The response is capped. Asking for a whole run used to serialise every bar
    — 5 MB for a 1d run and far more for a sub-daily one — and the docstring
    telling callers to pass ``year``/``month`` was the only thing standing
    between the browser and that payload. `truncated` says when the cap bit,
    and the month narrowing is still the right way to page through a run.
    """
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    path = get_versioned_prediction_file(
        symbol, model_type, timestamp, settings=settings, interval=interval
    )
    if not path.exists():
        raise HTTPException(status_code=404, detail="Versioned prediction file not found")

    month_prefix: str | None = None
    if year is not None and month is not None:
        month_prefix = f"{year:04d}-{month:02d}"

    capped = max(1, min(int(limit), MAX_PREDICTION_ROWS))
    try:
        rows = []
        matched = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if month_prefix and str(row.get("date", ""))[:7] != month_prefix:
                    continue
                matched += 1
                if len(rows) < capped:
                    rows.append(row)
        return {
            "predictions": rows,
            "month": month_prefix,
            "count": len(rows),
            "matched": matched,
            "truncated": matched > len(rows),
            "limit": capped,
        }
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/ai/predictions/months")
def get_ai_prediction_months(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str = "1d",
) -> dict[str, Any]:
    """Return the sorted list of ``YYYY-MM`` months present in a run."""
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    path = get_versioned_prediction_file(
        symbol, model_type, timestamp, settings=settings, interval=interval
    )
    if not path.exists():
        raise HTTPException(status_code=404, detail="Versioned prediction file not found")

    months: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                date_value = str(row.get("date", ""))
                if len(date_value) >= 7:
                    months.add(date_value[:7])
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"months": sorted(months)}


@router.get("/api/ai/feature_importances")
def get_feature_importances(
    symbol: str, model_type: str, timestamp: str, interval: str = "1d"
):
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    p = (
        resolve_versioned_prediction_dir(
            symbol, model_type, timestamp, settings=settings, interval=interval
        )
        / "feature_importances.json"
    )
    if not p.exists():
        raise HTTPException(status_code=404, detail="Feature importances not found for this run")
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/api/ai/models/{symbol}/{model_type}/{timestamp}")
def delete_ai_model(
    symbol: str, model_type: str, timestamp: str, interval: str = "1d"
) -> dict[str, Any]:
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    run_dir = resolve_versioned_prediction_dir(
        symbol, model_type.lower(), timestamp, settings=settings, interval=interval
    )
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Model run not found")

    # Every derived artifact names its source run (invariant 5), so once the
    # run is gone they are unattributable — and worse, still readable. A
    # deleted run left its backtest and recommendations behind, and a reader
    # globbing those directories happily reported its numbers alongside
    # current ones.
    slug = run_dir.name
    removed = [run_dir]
    for base in (
        settings.reports_dir / "backtests",
        settings.reports_dir / "evaluations",
        settings.reports_recommendations_dir,
        settings.reports_eval_dir,
    ):
        derived = base / symbol / slug
        if derived.is_dir():
            removed.append(derived)

    try:
        for path in removed:
            shutil.rmtree(path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "status": "deleted",
        "path": str(run_dir),
        "removed": [str(path) for path in removed],
    }
