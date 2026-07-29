"""Data-lake inventory, chart data and symbol deletion."""

import datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from prosper.api.commands import _is_valid_symbol
from prosper.inventory import DataManager

router = APIRouter()


@router.get("/api/data/dates")
def get_dates():
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)

    return {
        "today": today.strftime("%Y-%m-%d"),
        "yesterday": yesterday.strftime("%Y-%m-%d"),
        "default_start_month": "2017-08",
        "yesterday_month": yesterday.strftime("%Y-%m"),
    }


@router.get("/api/data/inventory")
def get_inventory(refresh: bool = False):
    return DataManager().get_inventory(refresh=refresh)


@router.get("/api/data/klines/{symbol}/{interval}")
def get_kline_chart_data(
    symbol: str,
    interval: str,
    limit: int = 1000,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 5000")

    try:
        payload = DataManager().get_chart_data(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start=start,
            end=end,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if payload["rows"] == 0:
        raise HTTPException(status_code=404, detail="No chart data found")
    return payload


@router.delete("/api/data/{symbol}")
def delete_symbol_data(symbol: str):
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    deleted_count = DataManager().delete_symbol_data(symbol)
    return {"status": "deleted", "paths_removed": deleted_count}
