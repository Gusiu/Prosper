"""Binance symbol metadata."""

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from prosper.config import get_settings

router = APIRouter()


@router.get("/api/meta/symbols")
def get_symbol_metadata() -> dict[str, Any]:
    settings = get_settings()
    symbols_path = settings.meta_dir / "binance_spot_symbols.json"
    if not symbols_path.exists():
        return {"symbols": [], "base_assets": [], "quote_assets": []}

    try:
        payload = json.loads(symbols_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Cannot read symbols metadata: {e}") from e

    symbols_raw = payload.get("symbols", []) if isinstance(payload, dict) else payload
    symbols: set[str] = set()
    base_assets: set[str] = set()
    quote_assets: set[str] = set()
    known_quotes = [
        "USDT",
        "FDUSD",
        "USDC",
        "TUSD",
        "BUSD",
        "BTC",
        "ETH",
        "BNB",
        "EUR",
        "TRY",
        "BRL",
        "DAI",
    ]

    for item in symbols_raw if isinstance(symbols_raw, list) else []:
        if isinstance(item, str):
            symbol = item.upper()
            symbols.add(symbol)
            for quote in sorted(known_quotes, key=len, reverse=True):
                if symbol.endswith(quote) and len(symbol) > len(quote):
                    base_assets.add(symbol[: -len(quote)])
                    quote_assets.add(quote)
                    break
        elif isinstance(item, dict):
            symbol = str(item.get("symbol", "")).upper()
            base = str(item.get("baseAsset", item.get("base_asset", ""))).upper()
            quote = str(item.get("quoteAsset", item.get("quote_asset", ""))).upper()
            if symbol:
                symbols.add(symbol)
            if base:
                base_assets.add(base)
            if quote:
                quote_assets.add(quote)

    return {
        "symbols": sorted(symbols),
        "base_assets": sorted(base_assets),
        "quote_assets": sorted(quote_assets),
    }
