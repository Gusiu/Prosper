"""Forecast quality month by month, for one prediction run."""

from typing import Any

from fastapi import APIRouter, HTTPException

from prosper.api.validation import (
    require_interval,
    require_model_type,
    require_run_timestamp,
    require_symbol,
)
from prosper.config import get_settings

router = APIRouter()


@router.get("/api/stability/{symbol}")
def run_stability_api(
    symbol: str,
    start: str = "2021-01-01",
    end: str = "2026-06-30",
    model_type: str | None = None,
    timestamp: str | None = None,
    interval: str | None = None,
) -> dict[str, Any]:
    """Return one run's quality as a time series, shaped for charting.

    This is the only view in the project that can distinguish a consistently
    mediocre model from one that worked in a single regime and stopped, which a
    whole-run aggregate hides by construction. Returns are not part of it —
    `/api/backtest` owns those.
    """
    symbol = require_symbol(symbol)
    # Optional run identity: absent means "the latest run", present means a path.
    model_type = require_model_type(model_type) if model_type else None
    interval = require_interval(interval) if interval else None
    timestamp = require_run_timestamp(timestamp) if timestamp else None

    # Imported lazily so the heavy evaluation stack stays out of app startup.
    from prosper.eval.stability import HORIZON_NAMES, eval_stability

    try:
        report = eval_stability(
            symbol,
            start,
            end,
            settings=get_settings(),
            model_type=model_type,
            timestamp=timestamp,
            interval=interval,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    months = report["months"]
    # One array per horizon per metric, aligned to `labels`, with null for a
    # month a horizon could not be scored in — the chart needs the gap to stay
    # a gap rather than close over it.
    series: dict[str, dict[str, list[Any]]] = {}
    for name in HORIZON_NAMES:
        series[name] = {
            metric: [month["horizons"][name][metric] for month in months]
            for metric in ("logloss", "brier", "accuracy")
        }
        series[name]["samples"] = [month["horizons"][name]["samples"] for month in months]
        series[name]["sparse"] = [month["horizons"][name]["sparse"] for month in months]

    return {
        "run": report["run"],
        "symbol": report["symbol"],
        "start": report["start"],
        "end": report["end"],
        "min_month_samples": report["min_month_samples"],
        "labels": [month["month"] for month in months],
        "overall": report["overall"],
        "series": series,
    }
