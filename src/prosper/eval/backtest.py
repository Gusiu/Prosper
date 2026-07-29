"""Trading simulator over one versioned prediction run.

Execution model
---------------
A signal derived from bar ``t`` is executed at the **open of bar t+1**. Trading
on bar ``t``'s own close would use a price that is only known once the bar has
finished, which quietly inflates every result.

The strategy is long-only by design, not by omission: the platform targets
Binance **SPOT**, where a short position cannot be opened. A `short` forecast
therefore means "hold no inventory", and the honest benchmark for that is buy
and hold — reported alongside every run.

Exposure is scaled by conviction rather than being all-or-nothing, and each
rebalance pays fees and slippage on the traded notional only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from rich.console import Console

from prosper.config import Settings, get_settings
from prosper.planner.windows import calculate_edge, calculate_risk_metric, map_to_recommendation
from prosper.storage.layout import get_backtest_report_path
from prosper.storage.runs import load_run_predictions, resolve_run
from prosper.utils.time import parse_date

console = Console()

# Target share of equity held in the asset, per recommendation. "Hold" keeps
# whatever the previous bar left, so a neutral forecast is not a trade signal.
EXPOSURE_POLICY: dict[str, float] = {
    "Strong Buy": 1.0,
    "Buy": 0.7,
    "Accumulate": 0.4,
    "Reduce": 0.2,
    "Sell": 0.0,
    "Strong Sell": 0.0,
}

# Rebalancing for a hair of drift costs more in fees than it gains.
MIN_REBALANCE_FRACTION = 0.05

TRADING_DAYS_PER_YEAR = 365.0

ASSUMPTIONS = [
    "Signals from bar t execute at the open of bar t+1.",
    "Long-only: Binance SPOT cannot open a short, so a 'short' forecast means "
    "hold no inventory. Buy and hold is reported as the benchmark.",
    "Exposure is scaled by conviction (Strong Buy 100% … Sell 0%); 'Hold' keeps "
    "the previous exposure.",
    "Fees and slippage are charged on the traded notional at every rebalance.",
    "Orders are assumed to fill in full at the open price; market impact beyond "
    "the flat slippage rate is not modelled.",
]


@dataclass
class BacktestConfig:
    """Execution assumptions for a simulation."""

    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.001
    horizon: str = "short"
    min_rebalance_fraction: float = MIN_REBALANCE_FRACTION
    exposure_policy: dict[str, float] = field(
        default_factory=lambda: dict(EXPOSURE_POLICY)
    )

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> BacktestConfig:
        base = {
            "fee_rate": settings.trading_fee_rate,
            "slippage_rate": settings.trading_slippage_proxy_rate,
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)


def _load_price_bars(
    symbol: str, settings: Settings, interval: str = "1d"
) -> dict[str, dict[str, float]]:
    """Return ``{date: {open, close}}`` for the whole available series."""
    base = settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
    files = [p for p in sorted(base.glob("**/*.parquet")) if p.is_file() and p.stat().st_size > 0]
    if not files:
        return {}

    frame = (
        pl.concat([pl.read_parquet(f, hive_partitioning=False) for f in files])
        .sort("open_time")
        .unique(subset=["open_time"], keep="first")
        .select(["open_time", "open", "close"])
    )
    return {
        row["open_time"].date().isoformat(): {
            "open": float(row["open"]),
            "close": float(row["close"]),
        }
        for row in frame.iter_rows(named=True)
    }


def _max_drawdown(equity: list[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def _sharpe(returns: list[float]) -> float | None:
    """Annualised Sharpe of the bar-to-bar returns, risk-free rate assumed 0."""
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if variance <= 0:
        return None
    return (mean / math.sqrt(variance)) * math.sqrt(TRADING_DAYS_PER_YEAR)


def run_backtest(
    symbol: str,
    start: str,
    end: str,
    initial_capital: float = 10000.0,
    fee_rate: float | None = None,
    settings: Settings | None = None,
    model_type: str | None = None,
    timestamp: str | None = None,
    interval: str | None = None,
    horizon: str = "short",
    slippage_rate: float | None = None,
    save_report: bool = False,
) -> dict[str, Any]:
    """Simulate the exposure-scaled strategy over a single prediction run."""
    if settings is None:
        settings = get_settings()
    if horizon not in ("short", "medium", "long"):
        return {"error": f"Unknown horizon: {horizon}"}

    config = BacktestConfig.from_settings(
        settings,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        horizon=horizon,
    )

    try:
        run = resolve_run(
            symbol,
            model_type=model_type,
            timestamp=timestamp,
            interval=interval,
            settings=settings,
        )
    except FileNotFoundError as e:
        return {"error": str(e)}

    start_dt = parse_date(start).date()
    end_dt = parse_date(end).date()
    predictions = [
        pred
        for pred in load_run_predictions(run)
        if start_dt <= parse_date(str(pred["date"])[:10]).date() <= end_dt
    ]
    if not predictions:
        return {"error": f"Run {run.slug} has no predictions in {start}..{end}."}

    # The strategy trades daily bars regardless of the run's own interval.
    prices = _load_price_bars(symbol, settings)
    if not prices:
        return {"error": f"No 1d klines found for {symbol}."}
    trading_days = sorted(prices)

    cash = config.initial_capital
    units = 0.0
    exposure = 0.0
    cost_rate = config.fee_rate + config.slippage_rate

    history: list[dict[str, Any]] = []
    equity_curve: list[float] = []
    bar_returns: list[float] = []
    trades = 0
    traded_notional = 0.0
    skipped_untrained = 0
    bars_in_market = 0
    previous_equity: float | None = None

    for pred in predictions:
        date_str = str(pred["date"])[:10]
        payload = pred.get(config.horizon) or {}
        if payload.get("trained") is False:
            skipped_untrained += 1
            continue

        # Execution happens on the *next* trading bar's open.
        position = _index_of(trading_days, date_str)
        if position is None or position + 1 >= len(trading_days):
            continue
        exec_date = trading_days[position + 1]
        exec_price = prices[exec_date]["open"]
        mark_price = prices[exec_date]["close"]
        if exec_price <= 0 or mark_price <= 0:
            continue

        edge = calculate_edge(payload["P_long"], payload["P_short"])
        risk = calculate_risk_metric(
            payload["P_long"],
            payload["P_short"],
            payload.get("depth_long_bins", {}),
            payload.get("depth_short_bins", {}),
        )
        recommendation = map_to_recommendation(edge, risk)
        # "Hold" is not a signal; it leaves the book untouched.
        target = config.exposure_policy.get(recommendation, exposure)

        equity_at_open = cash + units * exec_price
        action = "HOLD"
        if equity_at_open > 0:
            target_units = (target * equity_at_open) / exec_price
            delta_units = target_units - units
            delta_notional = abs(delta_units) * exec_price
            if delta_notional >= config.min_rebalance_fraction * equity_at_open:
                cost = delta_notional * cost_rate
                cash -= delta_units * exec_price + cost
                units = target_units
                exposure = target
                trades += 1
                traded_notional += delta_notional
                action = "BUY" if delta_units > 0 else "SELL"

        equity = cash + units * mark_price
        equity_curve.append(equity)
        if previous_equity and previous_equity > 0:
            bar_returns.append(equity / previous_equity - 1.0)
        previous_equity = equity
        if units > 0:
            bars_in_market += 1

        history.append(
            {
                "date": exec_date,
                "signal_date": date_str,
                "action": action,
                "price": mark_price,
                "exec_price": exec_price,
                "exposure": exposure,
                "value": equity,
                "recommendation": recommendation,
            }
        )

    if not history:
        return {
            "error": (
                f"Run {run.slug} produced no tradable bars in {start}..{end} "
                f"({skipped_untrained} untrained rows skipped)."
            )
        }

    final_value = equity_curve[-1]
    roi = (final_value - config.initial_capital) / config.initial_capital

    # Buy and hold over exactly the simulated window, same costs on entry.
    first_price = history[0]["exec_price"]
    last_price = history[-1]["price"]
    hold_units = config.initial_capital * (1 - cost_rate) / first_price
    hold_final = hold_units * last_price
    hold_roi = (hold_final - config.initial_capital) / config.initial_capital

    result = {
        "symbol": symbol,
        "run": run.to_dict(),
        "horizon": config.horizon,
        "start": start,
        "end": end,
        "initial_capital": config.initial_capital,
        "final_value": final_value,
        "roi_pct": roi * 100,
        "max_drawdown_pct": _max_drawdown(equity_curve) * 100,
        "sharpe": _sharpe(bar_returns),
        "total_trades": trades,
        "turnover_ratio": traded_notional / config.initial_capital,
        "time_in_market_pct": 100.0 * bars_in_market / len(history),
        "days_tested": len(history),
        "skipped_untrained": skipped_untrained,
        "fee_rate": config.fee_rate,
        "slippage_rate": config.slippage_rate,
        "benchmark": {
            "name": "buy_and_hold",
            "final_value": hold_final,
            "roi_pct": hold_roi * 100,
            "excess_roi_pct": (roi - hold_roi) * 100,
        },
        "assumptions": ASSUMPTIONS,
        "chart_data": history,
    }

    if save_report:
        report_path = get_backtest_report_path(symbol, settings=settings, run_slug=run.slug)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        result["report_path"] = str(report_path)

    return result


def _index_of(sorted_days: list[str], day: str) -> int | None:
    """Binary search for *day* in the sorted trading calendar."""
    lo, hi = 0, len(sorted_days) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_days[mid] == day:
            return mid
        if sorted_days[mid] < day:
            lo = mid + 1
        else:
            hi = mid - 1
    return None
