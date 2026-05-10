import json
from typing import Any

from rich.console import Console

from prosper.config import Settings, get_settings
from prosper.planner.windows import calculate_edge, calculate_risk_metric, map_to_recommendation
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import load_parquet
from prosper.utils.time import parse_date

console = Console()


def run_backtest(
    symbol: str,
    start: str,
    end: str,
    initial_capital: float = 10000.0,
    fee_rate: float | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    if settings is None:
        settings = get_settings()
    if fee_rate is None:
        fee_rate = settings.trading_fee_rate

    predictions_dir = settings.reports_predictions_dir / symbol / "daily"
    if not predictions_dir.exists():
        return {"error": f"No predictions found for {symbol}."}

    start_dt = parse_date(start).date()
    end_dt = parse_date(end).date()

    predictions: list[dict[str, Any]] = []
    for jsonl_file in sorted(predictions_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pred = json.loads(line)
                pred_date = parse_date(pred["date"]).date()
                if start_dt <= pred_date <= end_dt:
                    predictions.append(pred)

    if not predictions:
        return {"error": "No predictions loaded for given date range."}

    predictions.sort(key=lambda x: x["date"])

    # Load daily klines to get close prices for execution
    months = set()
    for p in predictions:
        dt = parse_date(p["date"])
        months.add((dt.year, dt.month))

    prices = {}
    for y, m in months:
        path = get_parquet_file_path(symbol, "1d", y, m, settings=settings)
        if path.exists():
            df = load_parquet(path)
            dates = df["open_time"].dt.date().to_list()
            closes = df["close"].to_list()
            for d, c in zip(dates, closes):
                prices[d.isoformat()] = c

    capital = initial_capital
    position = 0.0

    history = []
    last_price: float | None = None
    peak_value = capital
    max_drawdown = 0.0
    trades = 0

    for pred in predictions:
        date_str = pred["date"]

        # execution uses next day price roughly, but for MVP we use that day's close
        if date_str not in prices:
            continue

        current_price = prices[date_str]
        last_price = current_price

        # We will follow the short horizon recommendations for trading
        h_data = pred["short"]
        edge = calculate_edge(h_data["P_long"], h_data["P_short"])
        risk = calculate_risk_metric(
            h_data["P_long"],
            h_data["P_short"],
            h_data.get("depth_long_bins", {}),
            h_data.get("depth_short_bins", {}),
        )

        rec = map_to_recommendation(edge, risk)

        action = "HOLD"
        if rec in ("Buy", "Strong Buy", "Accumulate") and capital > 0:
            # Buy all
            size = capital * (1 - fee_rate) / current_price
            capital = 0.0
            position += size
            trades += 1
            action = "BUY"
        elif rec in ("Sell", "Strong Sell", "Reduce") and position > 0:
            # Sell all
            capital = position * current_price * (1 - fee_rate)
            position = 0.0
            trades += 1
            action = "SELL"

        current_value = capital + (position * current_price)
        if current_value > peak_value:
            peak_value = current_value
        dd = (peak_value - current_value) / peak_value
        if dd > max_drawdown:
            max_drawdown = dd

        history.append(
            {
                "date": date_str,
                "action": action,
                "price": current_price,
                "value": current_value,
                "recommendation": rec,
            }
        )

    final_mark_price = (
        last_price if last_price is not None else prices.get(predictions[-1]["date"], 0)
    )
    final_value = capital + (position * final_mark_price)
    roi = (final_value - initial_capital) / initial_capital

    return {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_value": final_value,
        "roi_pct": roi * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "total_trades": trades,
        "days_tested": len(history),
        "chart_data": history,
    }
