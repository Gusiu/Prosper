"""Trading simulator over one versioned prediction run.

Scope note: this is still the original MVP strategy — long-only, all-in, signal
executed on the same bar's close. Those limitations are reported in the result
payload under ``limitations`` so nothing downstream presents the numbers as a
realistic performance estimate. Rebuilding the execution model is tracked
separately.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console

from prosper.config import Settings, get_settings
from prosper.planner.windows import calculate_edge, calculate_risk_metric, map_to_recommendation
from prosper.storage.layout import get_backtest_report_path, get_parquet_file_path
from prosper.storage.parquet import load_parquet
from prosper.storage.runs import load_run_predictions, resolve_run
from prosper.utils.time import parse_date

console = Console()

BUY_SIGNALS = ("Buy", "Strong Buy", "Accumulate")
SELL_SIGNALS = ("Sell", "Strong Sell", "Reduce")

LIMITATIONS = [
    "Signals execute at the same bar's close, so entries use a price that is "
    "only known once the bar has completed.",
    "Long-only: 'short' forecasts close the position rather than opening a short.",
    "All-in sizing: every entry commits the full balance.",
    "Slippage is not modelled; only the flat fee rate is applied.",
]


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
    save_report: bool = False,
) -> dict[str, Any]:
    """Simulate the MVP strategy over a single prediction run."""
    if settings is None:
        settings = get_settings()
    if fee_rate is None:
        fee_rate = settings.trading_fee_rate

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

    if horizon not in ("short", "medium", "long"):
        return {"error": f"Unknown horizon: {horizon}"}

    # Load daily klines to get close prices for execution
    months = {
        (parse_date(str(p["date"])[:10]).year, parse_date(str(p["date"])[:10]).month)
        for p in predictions
    }

    prices: dict[str, float] = {}
    for y, m in sorted(months):
        path = get_parquet_file_path(symbol, "1d", y, m, settings=settings)
        if path.exists():
            df = load_parquet(path)
            dates = df["open_time"].dt.date().to_list()
            closes = df["close"].to_list()
            for d, c in zip(dates, closes, strict=True):
                prices[d.isoformat()] = c

    if not prices:
        return {"error": f"No 1d klines found for {symbol} in the requested range."}

    capital = initial_capital
    position = 0.0

    history = []
    last_price: float | None = None
    peak_value = capital
    max_drawdown = 0.0
    trades = 0
    skipped_untrained = 0

    for pred in predictions:
        date_str = str(pred["date"])[:10]

        # execution uses next day price roughly, but for MVP we use that day's close
        if date_str not in prices:
            continue

        h_data = pred[horizon]
        if h_data.get("trained") is False:
            skipped_untrained += 1
            continue

        current_price = prices[date_str]
        last_price = current_price

        edge = calculate_edge(h_data["P_long"], h_data["P_short"])
        risk = calculate_risk_metric(
            h_data["P_long"],
            h_data["P_short"],
            h_data.get("depth_long_bins", {}),
            h_data.get("depth_short_bins", {}),
        )

        rec = map_to_recommendation(edge, risk)

        action = "HOLD"
        if rec in BUY_SIGNALS and capital > 0:
            # Buy all
            size = capital * (1 - fee_rate) / current_price
            capital = 0.0
            position += size
            trades += 1
            action = "BUY"
        elif rec in SELL_SIGNALS and position > 0:
            # Sell all
            capital = position * current_price * (1 - fee_rate)
            position = 0.0
            trades += 1
            action = "SELL"

        current_value = capital + (position * current_price)
        peak_value = max(peak_value, current_value)
        dd = (peak_value - current_value) / peak_value
        max_drawdown = max(max_drawdown, dd)

        history.append(
            {
                "date": date_str,
                "action": action,
                "price": current_price,
                "value": current_value,
                "recommendation": rec,
            }
        )

    if not history:
        return {
            "error": (
                f"Run {run.slug} produced no tradable bars in {start}..{end} "
                f"({skipped_untrained} untrained rows skipped)."
            )
        }

    final_mark_price = last_price if last_price is not None else 0.0
    final_value = capital + (position * final_mark_price)
    roi = (final_value - initial_capital) / initial_capital

    result = {
        "symbol": symbol,
        "run": run.to_dict(),
        "horizon": horizon,
        "start": start,
        "end": end,
        "initial_capital": initial_capital,
        "final_value": final_value,
        "roi_pct": roi * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "total_trades": trades,
        "days_tested": len(history),
        "skipped_untrained": skipped_untrained,
        "fee_rate": fee_rate,
        "limitations": LIMITATIONS,
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
