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
from prosper.domain import DEFAULT_TRADED_HORIZON
from prosper.planner.windows import (
    SequenceGrader,
    calculate_edge,
    calculate_risk_metric,
    expected_move,
)
from prosper.predict.defaults import HORIZON_NAMES
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
    "Buy and hold is the weakest of four comparisons. Also reported: the same "
    "average exposure held throughout, the same exposure path mistimed by every "
    "circular shift, and a 7-bar momentum rule containing no model at all.",
]


@dataclass
class BacktestConfig:
    """Execution assumptions for a simulation."""

    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.001
    horizon: str = DEFAULT_TRADED_HORIZON
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


# Bars of trailing price change the naive trend benchmark looks at. A week, so
# it is the crudest rule anyone would try and no part of it is fitted.
MOMENTUM_LOOKBACK_BARS = 7


def _replay(prices: list[float], exposures: list[float], cost_rate: float) -> float:
    """ROI of holding `exposures[t]` into the return of bar t+1, as a fraction.

    Deliberately simpler than the simulation above — no rebalance floor, no
    open/close distinction — because it exists to compare an exposure path
    against alternatives, and every arm goes through this same function. Its
    absolute value is not comparable to `roi_pct`.
    """
    if len(prices) < 2:
        return 0.0
    equity = 1.0
    held_previous = 0.0
    for i in range(len(prices) - 1):
        if prices[i] <= 0:
            continue
        bar_return = prices[i + 1] / prices[i] - 1.0
        turnover = abs(exposures[i] - held_previous)
        equity *= 1.0 + exposures[i] * bar_return - turnover * cost_rate
        held_previous = exposures[i]
        if equity <= 0:
            return -1.0
    return equity - 1.0


def _timing_test(prices: list[float], exposures: list[float], cost_rate: float) -> dict[str, Any]:
    """Does this exposure path beat mistimed copies of itself?

    Buy and hold is not a sufficient benchmark, which is what invariant 7 was
    written before anyone had measured. A long-only strategy that sits flat
    through high-volatility months finishes ahead of holding by compounding
    alone: on ETHUSDT tft over a 2023-start window the strategy beat holding by
    11.06 pp while its cumulative monthly excess was negative in all 40 months,
    and it was flat during the largest months in *both* directions.

    The null keeps the exposure sequence exactly as it is — same values, same
    turnover, same average — and only changes when it applies, by circular
    shift. `percentile` is the share of misalignments the real path beat: ~50
    means the timing explained nothing.

    A random permutation would be the wrong null. It destroys the exposure
    autocorrelation and pays turnover the strategy never paid; at 1h, 51k bars
    of churn drive it to total loss, so it measures "does this avoid
    over-trading" instead.

    Read it as necessary, not sufficient. A plain 7-bar momentum rule with no
    model in it scores a median 90.5 across these runs, so clearing this bar is
    the beginning of an argument and `naive_momentum` below is the rest of it.
    """
    if len(prices) < 3:
        return {"samples": 0, "percentile": None, "median_roi_pct": None}

    actual = _replay(prices, exposures, cost_rate)
    shifted = [
        _replay(prices, exposures[k:] + exposures[:k], cost_rate)
        for k in range(1, len(exposures))
    ]
    # Mid-rank: ties count half. A constant exposure path is identical under
    # every shift, so counting only strict wins reported it at percentile 0 —
    # reading as "the worst possible timing" when the truth is that a path with
    # no variation carries no timing information at all. Mid-rank puts it at 50,
    # which is what "explains nothing" is supposed to look like here.
    tolerance = 1e-12
    beaten = sum(1 for roi in shifted if roi < actual - tolerance)
    tied = sum(1 for roi in shifted if abs(roi - actual) <= tolerance)
    ordered = sorted(shifted)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return {
        "samples": len(shifted),
        "percentile": 100.0 * (beaten + 0.5 * tied) / len(shifted),
        "median_roi_pct": median * 100.0,
        "replay_roi_pct": actual * 100.0,
    }


def _naive_momentum(prices: list[float], cost_rate: float, lookback: int) -> dict[str, Any]:
    """Fully invested when the last *lookback* bars rose, flat otherwise.

    The comparison that matters most and was missing. It contains no forecast,
    no training and no fitted parameter, so anything the models cannot beat here
    they have not earned. Measured across the 1d runs it returns a median of
    -147 pp against buy & hold where the models' own policy returns -246 pp.
    """
    if len(prices) < lookback + 2:
        return {"roi_pct": None, "mean_exposure": None}
    exposures = [
        1.0 if i >= lookback and prices[i] > prices[i - lookback] else 0.0
        for i in range(len(prices))
    ]
    return {
        "roi_pct": _replay(prices, exposures, cost_rate) * 100.0,
        "mean_exposure": sum(exposures) / len(exposures),
        "lookback_bars": lookback,
    }


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
    horizon: str = DEFAULT_TRADED_HORIZON,
    slippage_rate: float | None = None,
    save_report: bool = False,
) -> dict[str, Any]:
    """Simulate the exposure-scaled strategy over a single prediction run."""
    if settings is None:
        settings = get_settings()
    if horizon not in HORIZON_NAMES:
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
    # One grader for the whole walk: its risk gates are quantiles of what it has
    # already seen, so it has to be built once and fed in order.
    grader = SequenceGrader(settings.planner_round_trip_cost)

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

        depth_long = payload.get("depth_long_bins", {})
        depth_short = payload.get("depth_short_bins", {})
        edge = calculate_edge(payload["P_long"], payload["P_short"])
        risk = calculate_risk_metric(
            payload["P_long"], payload["P_short"], depth_long, depth_short
        )
        # Graded by the same object the planner uses, so a bar cannot be
        # `Strong Buy` in the recommendation report and `Buy` in the simulation
        # that is supposed to be trading it. The grader also carries the cost
        # gate, which this loop skipped entirely.
        move = (
            expected_move(payload["P_long"], payload["P_short"], depth_long, depth_short)
            if depth_long and depth_short
            else None
        )
        recommendation = grader.grade(edge, risk, move)
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

    # Three more comparisons, because buy & hold alone cannot separate a
    # forecast from a reduced-beta position. See `_timing_test`.
    marks = [float(row["price"]) for row in history]
    path = [float(row["exposure"]) for row in history]
    mean_exposure = sum(path) / len(path)
    # Same average exposure, entered once and held: what carrying less beta
    # explains on its own, with no timing at all.
    constant_units = mean_exposure * (1.0 - cost_rate) / marks[0]
    constant_roi = constant_units * marks[-1] + (1.0 - mean_exposure) - 1.0

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
        "mean_exposure": mean_exposure,
        # Ordered weakest claim to strongest: beating buy & hold may be beta,
        # beating constant exposure may be volatility, beating your own
        # mistimed copies may still be a trend rule, and `naive_momentum` is
        # what a trend rule with no model in it achieves.
        "nulls": {
            "constant_exposure": {
                "roi_pct": constant_roi * 100,
                "excess_roi_pct": (roi - constant_roi) * 100,
                "mean_exposure": mean_exposure,
            },
            "mistimed_replay": _timing_test(marks, path, cost_rate),
            "naive_momentum": _naive_momentum(marks, cost_rate, MOMENTUM_LOOKBACK_BARS),
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
