"""Evaluation listings and per-run detail."""

import json
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException

from prosper.api.commands import _is_valid_symbol
from prosper.config import get_settings
from prosper.eval.predictions import flatten_quality_row
from prosper.storage.layout import parse_evaluation_folder, resolve_evaluation_run_dir

router = APIRouter()


def _read_json_file(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl_file(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


def _evaluation_summary_from_metrics(metrics: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    return {
        "symbol": metrics.get("symbol"),
        "model_type": metrics.get("model_type"),
        "timestamp": metrics.get("timestamp"),
        "interval": metrics.get("interval", "1d"),
        "created_at": metrics.get("created_at"),
        "path": str(run_dir),
        "samples": overall.get("samples", metrics.get("scored_rows", 0)),
        "missing_targets": overall.get("missing_targets"),
        "untrained_rows": overall.get("untrained_rows", 0),
        "model_score": overall.get("model_score"),
        "accuracy": overall.get("accuracy"),
        "brier": overall.get("brier"),
        "nll": overall.get("nll"),
        "ece": overall.get("ece"),
        "artifacts": metrics.get("artifacts", {}),
    }


@router.get("/api/ai/evaluations")
def list_ai_evaluations(
    symbol: str | None = None,
    model_type: str | None = None,
    sort_by: str = "model_score",
) -> dict[str, list[dict[str, Any]]]:
    settings = get_settings()
    base = settings.reports_dir / "evaluations"
    if not base.exists():
        return {"evaluations": []}

    symbol_filter = symbol.upper() if symbol else None
    model_filter = model_type.lower() if model_type else None
    evaluations: list[dict[str, Any]] = []
    for symbol_dir in sorted(base.iterdir()):
        if not symbol_dir.is_dir() or symbol_dir.name.startswith("."):
            continue
        if symbol_filter and symbol_dir.name.upper() != symbol_filter:
            continue
        for run_dir in sorted(symbol_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            parsed = parse_evaluation_folder(run_dir.name)
            if not parsed:
                continue
            parsed_model_type, parsed_interval, parsed_timestamp = parsed
            if model_filter and parsed_model_type.lower() != model_filter:
                continue
            metrics = _read_json_file(run_dir / "metrics.json", {})
            if not metrics:
                metrics = {
                    "symbol": symbol_dir.name,
                    "model_type": parsed_model_type,
                    "interval": parsed_interval,
                    "timestamp": parsed_timestamp,
                }
            evaluations.append(_evaluation_summary_from_metrics(metrics, run_dir))

    sort_key = sort_by if sort_by in {"model_score", "accuracy", "brier", "ece", "nll"} else "model_score"
    reverse = sort_key != "brier" and sort_key != "ece" and sort_key != "nll"
    evaluations.sort(
        key=lambda item: (
            item.get(sort_key) is not None,
            item.get(sort_key) if reverse else -(item.get(sort_key) or 0),
            item.get("created_at") or "",
        ),
        reverse=True,
    )
    return {"evaluations": evaluations}


@router.get("/api/ai/evaluations/status")
def get_evaluation_batch_status() -> dict[str, Any]:
    settings = get_settings()
    status_path = settings.reports_dir / "evaluations" / "batch_status.json"
    payload = _read_json_file(status_path, {})
    if not payload:
        return {"status": "idle", "evaluated": 0, "failed": 0, "results": [], "alerts": []}

    alerts: list[str] = []
    failed = int(payload.get("failed", 0) or 0)
    evaluated = int(payload.get("evaluated", 0) or 0)
    if failed > 0:
        alerts.append(f"{failed} evaluation run(s) failed in the latest batch.")
    for item in payload.get("results", []):
        if item.get("status") != "ok":
            continue
        score = item.get("model_score")
        if isinstance(score, int | float) and score < 50:
            alerts.append(
                f"Low model score ({score:.1f}) for {item.get('symbol')} "
                f"{item.get('model_type')} {item.get('timestamp')}."
            )
    payload["alerts"] = alerts
    payload["status"] = "failed" if failed and not evaluated else ("partial" if failed else "ok")
    payload["status_path"] = str(status_path)
    return payload


def _load_worst_predictions(
    run_dir: Path, limit: int, flag: str | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Return the worst-scoring rows of a run, plus the total scored count.

    Reads `predictions_quality.parquet` lazily so a run with hundreds of
    thousands of rows costs a bounded amount of memory. Falls back to the
    legacy JSONL for evaluations produced before the Parquet switch.
    """
    parquet_path = run_dir / "predictions_quality.parquet"
    if parquet_path.exists():
        lazy = pl.scan_parquet(str(parquet_path)).filter(pl.col("status") == "scored")
        if flag:
            lazy = lazy.filter(pl.col("flags").str.contains(flag, literal=True))
        total = int(lazy.select(pl.len()).collect().item())
        frame = lazy.sort("error_score", descending=True).head(limit).collect()
        return frame.to_dicts(), total

    legacy = run_dir / "predictions_quality.jsonl"
    rows = [
        row
        for row in _read_jsonl_file(legacy, limit=20000)
        if row.get("status") == "scored"
        and (not flag or flag in (row.get("flags") or []))
    ]
    rows.sort(key=lambda row: row.get("error_score", 0.0), reverse=True)
    return [flatten_quality_row(row) for row in rows[:limit]], len(rows)


@router.get("/api/ai/evaluations/{symbol}/{model_type}/{timestamp}")
def get_ai_evaluation_detail(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str = "1d",
    limit: int = 100,
    flag: str | None = None,
) -> dict[str, Any]:
    """Summary, calibration and the worst rows of one evaluation run.

    Deliberately bounded: an unfiltered response for a 1h run used to be ~71 MB
    because every recommendation was serialised so the UI could render fifty.
    """
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 5000")

    run_dir = resolve_evaluation_run_dir(
        symbol, model_type.lower(), timestamp, settings=settings, interval=interval
    )
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Evaluation not found")

    metrics = _read_json_file(run_dir / "metrics.json", {})
    calibration = _read_json_file(run_dir / "calibration.json", {})
    worst_rows, scored_total = _load_worst_predictions(run_dir, limit=limit, flag=flag)
    recommendations = _read_json_file(run_dir / "recommendations.json", {"items": []})
    return {
        "metrics": metrics,
        "summary": _evaluation_summary_from_metrics(metrics, run_dir),
        "calibration": calibration,
        "worst_predictions": worst_rows,
        "worst_predictions_total": scored_total,
        "recommendations": recommendations.get("items", [])[:limit],
        "recommendations_truncated": bool(recommendations.get("truncated")),
        "artifacts": metrics.get("artifacts", {}),
    }


@router.get("/api/ai/evaluations/{symbol}/{model_type}/{timestamp}/flags")
def get_ai_evaluation_flags(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str = "1d",
) -> dict[str, Any]:
    """Distinct quality flags in a run, with counts, for the UI filter."""
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    run_dir = resolve_evaluation_run_dir(
        symbol, model_type.lower(), timestamp, settings=settings, interval=interval
    )
    parquet_path = run_dir / "predictions_quality.parquet"
    if not parquet_path.exists():
        return {"flags": []}

    frame = (
        pl.scan_parquet(str(parquet_path))
        .filter((pl.col("status") == "scored") & (pl.col("flags") != ""))
        .select(pl.col("flags").str.split("|").alias("flag"))
        .explode("flag")
        .group_by("flag")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect()
    )
    return {"flags": frame.to_dicts()}
