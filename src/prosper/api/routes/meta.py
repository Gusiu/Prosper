"""Binance symbol metadata."""

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from prosper.config import get_settings
from prosper.domain import DEFAULT_HORIZONS, HORIZON_NAMES
from prosper.predict.defaults import (
    DEFAULT_EPOCH_BUDGET,
    EPOCHLESS_MODELS,
)

router = APIRouter()


def _horizon_label(days: int) -> str:
    """A human span for a horizon, derived rather than written down."""
    if days % 364 == 0 and days >= 364:
        years = days // 364
        return "1 year" if years == 1 else f"{years} years"
    if days % 7 == 0:
        weeks = days // 7
        return "1 week" if weeks == 1 else f"{weeks} weeks"
    return f"{days} days"


@router.get("/api/meta/backtest-defaults")
def get_backtest_defaults() -> dict[str, Any]:
    """Defaults the backtest form pre-fills from.

    Read from the same place the simulation reads them, for the reason the epoch
    field taught: a literal in the HTML becomes a second definition, and the two
    drift without anything failing. The exposure grades are sent in scale order
    so the form can render them as one axis from fully invested to flat.
    """
    from prosper.eval.backtest import (
        EXPOSURE_KEYS,
        EXPOSURE_POLICY,
        MIN_REBALANCE_FRACTION,
        MOMENTUM_LOOKBACK_BARS,
    )

    settings = get_settings()
    return {
        "initial_capital": 10000.0,
        "fee_rate": settings.trading_fee_rate,
        "slippage_rate": settings.trading_slippage_proxy_rate,
        "round_trip_cost": settings.planner_round_trip_cost,
        "min_rebalance_fraction": MIN_REBALANCE_FRACTION,
        "momentum_lookback_bars": MOMENTUM_LOOKBACK_BARS,
        "horizons": list(HORIZON_NAMES),
        # key -> {label, default}; `Hold` is absent because it carries the
        # previous exposure rather than a target, and must not be settable.
        "exposure": [
            {"key": key, "label": label, "default": EXPOSURE_POLICY[label]}
            for key, label in EXPOSURE_KEYS.items()
        ],
    }


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


@router.get("/api/meta/training-defaults")
def get_training_defaults() -> dict[str, Any]:
    """Defaults the training form should pre-fill from.

    The form used to carry `value="5"` in the HTML while the CLI defaulted to
    20, so the same model trained from the dashboard ran a different
    configuration. There is one definition now and the browser reads it.
    """
    return {
        # Names *and* spans: the dashboard used to write "≈52w (365d)" into
        # markup, which was already wrong once the year became 364 days. A label
        # that restates a value is a second definition of it.
        "horizons": [
            {
                "name": h.name,
                "forward_days": h.forward_days,
                "weeks": h.forward_days / 7,
                "label": _horizon_label(h.forward_days),
            }
            for h in DEFAULT_HORIZONS
        ],
        "epoch_budget": dict(DEFAULT_EPOCH_BUDGET),
        "epochless_models": sorted(EPOCHLESS_MODELS),
    }
