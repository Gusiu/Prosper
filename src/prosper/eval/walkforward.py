from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import DEFAULT_HORIZONS, direction_from_return
from prosper.storage.layout import (
    get_eval_walkforward_summary_path,
    get_parquet_file_path,
)
from prosper.storage.runs import RunRef, load_run_predictions, resolve_run
from prosper.utils.time import generate_month_range

ALLOWED_RECOMMENDATIONS = {
    "Strong Buy",
    "Buy",
    "Accumulate",
    "Hold",
    "Reduce",
    "Sell",
    "Strong Sell",
}

EXPOSURE_POLICY: dict[str, float] = {
    "Strong Buy": 1.0,
    "Buy": 0.7,
    "Accumulate": 0.4,
    "Hold": 0.0,
    "Reduce": 0.2,
    "Sell": 0.0,
    "Strong Sell": 0.0,
}


def multiclass_logloss_brier(
    probs: dict[str, float],
    y_true: str,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Logloss and Brier over whatever direction classes *probs* carries.

    Taking a dict rather than one argument per class is what lets a two-class
    run and a legacy three-class one be scored by the same code.
    """
    p_true = max(eps, float(probs.get(y_true, 0.0)))
    logloss = -math.log(p_true)
    brier = sum(
        (float(prob) - (1.0 if name == y_true else 0.0)) ** 2 for name, prob in probs.items()
    )
    return logloss, brier


def load_daily_closes(symbol: str, start_ym: str, end_ym: str, settings: Settings) -> pl.DataFrame:
    """Load daily klines parquet across months and return open_time/date, close arrays."""
    from prosper.utils.time import generate_month_range

    parts: list[pl.DataFrame] = []
    for y, m in generate_month_range(start_ym, end_ym):
        path = get_parquet_file_path(symbol, "1d", y, month=m, settings=settings)
        if path.exists():
            parts.append(pl.read_parquet(path, hive_partitioning=False))

    if not parts:
        raise FileNotFoundError(f"No 1d parquet found for {symbol} in {start_ym}..{end_ym}")

    df = pl.concat(parts).sort("open_time").unique(subset=["open_time"], keep="first")
    df = df.with_columns(pl.col("open_time").dt.strftime("%Y-%m-%d").alias("_date"))
    return df


def load_run_predictions_in_range(
    run: RunRef, start_ym: str, end_ym: str
) -> list[dict[str, Any]]:
    """Load a run's predictions restricted to the requested months."""
    months = {f"{y:04d}-{m:02d}" for y, m in generate_month_range(start_ym, end_ym)}
    return [row for row in load_run_predictions(run) if str(row["date"])[:7] in months]


def load_recommendations_windows(
    symbol: str, start_ym: str, end_ym: str, settings: Settings, run_slug: str | None = None
) -> list[dict[str, Any]]:
    """Load recommendation windows JSON for requested months of one run."""
    from prosper.storage.layout import get_recommendation_report_path

    windows: list[dict[str, Any]] = []
    for y, m in generate_month_range(start_ym, end_ym):
        p = get_recommendation_report_path(symbol, y, m, settings=settings, run_slug=run_slug)
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        windows.extend(payload.get("windows", []) or [])
    return windows


@dataclass(frozen=True)
class Window:
    horizon: str
    start_date: str
    end_date: str
    recommendation: str

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Window:
        return Window(
            horizon=str(d.get("horizon", "")),
            start_date=str(d.get("start_date", "")),
            end_date=str(d.get("end_date", "")),
            recommendation=str(d.get("recommendation", "")),
        )

    def contains(self, date_str: str) -> bool:
        return self.start_date <= date_str <= self.end_date


def eval_walkforward(
    symbol: str,
    start: str,
    end: str,
    train_months: int = 24,
    step_months: int = 1,
    settings: Settings | None = None,
    model_type: str | None = None,
    timestamp: str | None = None,
    interval: str | None = None,
) -> dict[str, Any]:
    """
    Walk-forward evaluation of one versioned prediction run.

    Notes:
    - Predictions and recommendations are precomputed by earlier steps.
    - `train_months` / `step_months` define test-month slices; we do not retrain in this MVP.
    - The run is identified explicitly so the report can be attributed to a model.
    """
    if settings is None:
        settings = get_settings()

    run = resolve_run(
        symbol, model_type=model_type, timestamp=timestamp, interval=interval, settings=settings
    )

    # Load data needed for evaluation
    daily_df = load_daily_closes(symbol, start, end, settings)
    daily_dates = daily_df["_date"].to_list()
    closes = daily_df["close"].to_list()

    predictions = load_run_predictions_in_range(run, start, end)
    if not predictions:
        raise FileNotFoundError(
            f"Run {run.slug} has no predictions in {start}..{end}"
        )

    windows_payload = load_recommendations_windows(
        symbol, start, end, settings, run_slug=run.slug
    )
    windows: list[Window] = [Window.from_dict(w) for w in windows_payload if w.get("horizon")]
    for w in windows:
        if w.recommendation not in ALLOWED_RECOMMENDATIONS:
            raise ValueError(f"Unexpected recommendation: {w.recommendation}")

    # Pre-index recommendations by horizon for O(1) per date (small N for smoke).
    windows_by_horizon: dict[str, list[Window]] = {}
    for w in windows:
        windows_by_horizon.setdefault(w.horizon, []).append(w)

    # Realised forward returns are read off the 1d series, so the horizons are
    # measured in daily bars regardless of the run's own interval.
    horizon_specs: dict[str, int] = {h.name: h.steps("1d") for h in DEFAULT_HORIZONS}

    # Map prediction row date to index in daily data
    date_to_idx = {d: i for i, d in enumerate(daily_dates)}

    def forward_return_idx(i: int, forward_days: int) -> float | None:
        j = i + forward_days
        if j >= len(closes):
            return None
        if closes[i] == 0:
            return None
        return float(closes[j] / closes[i] - 1.0)

    def exposure_for(horizon: str, date_str: str) -> float:
        for w in windows_by_horizon.get(horizon, []):
            if w.contains(date_str):
                return float(EXPOSURE_POLICY.get(w.recommendation, 0.0))
        return 0.0

    # Walk-forward months slicing
    month_list = generate_month_range(start, end)
    if not month_list:
        raise ValueError("Invalid start/end month range")

    test_months = []
    for idx in range(0, len(month_list), max(1, step_months)):
        test_months.append(month_list[idx])

    horizons = ("short", "medium", "long")
    horizon_results: dict[str, dict[str, Any]] = {
        h: {"n_samples": 0, "logloss_sum": 0.0, "brier_sum": 0.0, "pnl_sum": 0.0} for h in horizons
    }

    fee = float(settings.trading_fee_rate)
    slippage = float(settings.trading_slippage_proxy_rate)
    cost_rate = fee + slippage

    # For summary by test month
    month_summaries: list[dict[str, Any]] = []

    for yy, mm in test_months:
        ym = f"{yy:04d}-{mm:02d}"
        month_rows = [p for p in predictions if str(p["date"]).startswith(ym)]

        month_out: dict[str, Any] = {"month": ym, "horizons": {}}
        for h in horizons:
            month_out["horizons"][h] = {
                "n_samples": 0,
                "logloss": None,
                "brier": None,
                "mean_pnl": None,
            }

        for pred in month_rows:
            date_str = str(pred["date"])[:10]
            if date_str not in date_to_idx:
                continue
            i = date_to_idx[date_str]

            for h in horizons:
                probs = pred[h]
                if probs.get("trained") is False:
                    continue

                forward_days = horizon_specs[h]
                r = forward_return_idx(i, forward_days)
                if r is None:
                    continue
                y_true = direction_from_return(r)
                if y_true is None:
                    # An exactly flat forward return has no side to score.
                    continue

                row_probs = {
                    key[2:]: float(value)
                    for key, value in probs.items()
                    if key.startswith("P_") and value is not None
                }
                logloss, brier = multiclass_logloss_brier(row_probs, y_true)

                exposure = exposure_for(h, date_str)
                pnl = exposure * r - exposure * cost_rate

                horizon_results[h]["n_samples"] += 1
                horizon_results[h]["logloss_sum"] += logloss
                horizon_results[h]["brier_sum"] += brier
                horizon_results[h]["pnl_sum"] += pnl

                month_h = month_out["horizons"][h]
                month_h["n_samples"] += 1
                month_h["logloss"] = (
                    logloss if month_h["logloss"] is None else month_h["logloss"] + logloss
                )
                month_h["brier"] = brier if month_h["brier"] is None else month_h["brier"] + brier
                month_h["mean_pnl"] = (
                    pnl if month_h["mean_pnl"] is None else month_h["mean_pnl"] + pnl
                )

        # finalize month metrics
        for h in horizons:
            month_h = month_out["horizons"][h]
            n = int(month_h["n_samples"])
            if n > 0:
                month_h["logloss"] = month_h["logloss"] / n
                month_h["brier"] = month_h["brier"] / n
                month_h["mean_pnl"] = month_h["mean_pnl"] / n
            else:
                month_h["logloss"] = None
                month_h["brier"] = None
                month_h["mean_pnl"] = None

        month_summaries.append(month_out)

    # finalize horizon metrics
    horizons_out: dict[str, Any] = {}
    for h in horizons:
        n = int(horizon_results[h]["n_samples"])
        if n > 0:
            horizons_out[h] = {
                "n_samples": n,
                "logloss": horizon_results[h]["logloss_sum"] / n,
                "brier": horizon_results[h]["brier_sum"] / n,
                "mean_pnl": horizon_results[h]["pnl_sum"] / n,
            }
        else:
            horizons_out[h] = {"n_samples": 0, "logloss": None, "brier": None, "mean_pnl": None}

    report = {
        "symbol": symbol,
        "run": run.to_dict(),
        "model_type": run.model_type,
        "interval": run.interval,
        "timestamp": run.timestamp,
        "start": start,
        "end": end,
        "train_months": train_months,
        "step_months": step_months,
        "trading_cost_rate": cost_rate,
        "generated_at": datetime.now(UTC).isoformat(),
        "horizons": horizons_out,
        "walkforward_test_months": month_summaries,
    }

    out_path = get_eval_walkforward_summary_path(symbol, settings=settings, run_slug=run.slug)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return report
