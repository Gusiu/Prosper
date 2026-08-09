"""Trading simulation over a single prediction run."""

from typing import Any

from fastapi import APIRouter, HTTPException

from prosper.api.commands import _is_valid_symbol
from prosper.config import get_settings

router = APIRouter()


@router.get("/api/backtest/{symbol}")
def run_backtest_api(
    symbol: str,
    start: str = "2021-01-01",
    end: str = "2024-06-30",
    model_type: str | None = None,
    timestamp: str | None = None,
    interval: str | None = None,
    horizon: str = "short",
    capital: float = 10000.0,
    fee_rate: float | None = None,
    slippage_rate: float | None = None,
    round_trip_cost: float | None = None,
    min_rebalance: float | None = None,
    momentum_lookback: int | None = None,
    exposure: str | None = None,
) -> dict[str, Any]:
    """Simulate one prediction run and return its capital curve and stats.

    The run identity is part of the response: a backtest number is meaningless
    unless you know which model produced the signals behind it.
    """
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    settings = get_settings()
    # Import lazily to avoid pulling heavy ML/runtime deps during module import
    from prosper.eval.backtest import (
        MOMENTUM_LOOKBACK_BARS,
        parse_exposure_policy,
        run_backtest,
    )

    # A malformed exposure string is the caller's mistake, not a server fault, so
    # it must come back as 400 with the parser's own message rather than 500.
    try:
        policy = parse_exposure_policy(exposure)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        results = run_backtest(
            symbol,
            start,
            end,
            initial_capital=capital,
            settings=settings,
            model_type=model_type,
            timestamp=timestamp,
            interval=interval,
            horizon=horizon,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            round_trip_cost=round_trip_cost,
            min_rebalance_fraction=min_rebalance,
            momentum_lookback=(
                MOMENTUM_LOOKBACK_BARS if momentum_lookback is None else int(momentum_lookback)
            ),
            exposure_policy=policy,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])

    history = results.get("chart_data", [])

    return {
        "run": results["run"],
        "horizon": results["horizon"],
        "roi": round(results["roi_pct"], 2),
        "max_drawdown": round(results["max_drawdown_pct"], 2),
        "final_capital": round(results["final_value"], 2),
        "initial_capital": results["initial_capital"],
        "total_trades": results["total_trades"],
        "days_tested": results["days_tested"],
        "skipped_untrained": results["skipped_untrained"],
        "sharpe": results["sharpe"],
        "turnover_ratio": results["turnover_ratio"],
        "time_in_market_pct": results["time_in_market_pct"],
        "benchmark": results["benchmark"],
        # Buy & hold alone cannot separate a forecast from a reduced-beta
        # position, so the browser gets the whole comparison set too. Leaving it
        # out of the response is how the CLI ends up the only honest surface.
        "nulls": results.get("nulls", {}),
        "mean_exposure": results.get("mean_exposure"),
        "config": results.get("config", {}),
        "assumptions": results["assumptions"],
        "chart_data": {
            "dates": [h["date"] for h in history],
            "capital": [h["value"] for h in history],
            "close": [h["price"] for h in history],
        },
    }
