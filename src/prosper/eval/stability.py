"""Is a run's forecast quality stable over time, or was it good in one year?

This replaces `eval walkforward`, which reported three things and only owned
one of them. Its per-horizon logloss and Brier duplicated `eval predictions`,
which already scores the whole run. Its PnL duplicated `eval backtest`, and did
so incorrectly: it carried a second copy of `EXPOSURE_POLICY` that mapped
`Hold` to 0.0 while the backtest leaves the previous exposure in place, so the
two disagreed on every neutral bar. It also called itself walk-forward while
retraining nothing — `train_months` was accepted and ignored.

What only this module can answer is whether quality *holds up*. A single
aggregate over six years hides the difference between a model that is
consistently mediocre and one that was excellent in 2021 and useless since,
and the second is worth knowing before anything goes in a thesis. So the report
is a per-month time series of proper scores per horizon, and nothing else.

Because the recommendations are no longer needed, this reads only the run's own
predictions and the realised closes — the planner does not have to have run.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import DEFAULT_HORIZONS, direction_from_return
from prosper.storage.layout import get_eval_stability_summary_path
from prosper.storage.runs import RunRef, load_run_predictions, resolve_run
from prosper.utils.time import parse_date

HORIZON_NAMES: tuple[str, ...] = tuple(h.name for h in DEFAULT_HORIZONS)

# A month with fewer scored bars than this is reported but flagged: a mean over
# a handful of rows moves for reasons that have nothing to do with the model.
MIN_MONTH_SAMPLES = 5


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


def load_closes(symbol: str, interval: str, settings: Settings) -> pl.DataFrame:
    """Closes at *interval*, sorted and keyed by ``open_time``.

    It must be the run's own interval, not 1d. Reading daily closes for an
    hourly run makes all 24 bars of a date resolve to one candle and measures
    the horizon from the daily close rather than from the bar the forecast was
    made on — the defect invariant 4 exists to prevent, and one this module
    inherited from `eval walkforward` before it was rewritten.

    The whole series is loaded rather than the requested months, because a
    forward return reaches past `end` by up to a year.
    """
    base = settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
    files = sorted(base.rglob("*.parquet")) if base.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No {interval} parquet found for {symbol}: {base}")

    df = (
        pl.concat([pl.read_parquet(path, hive_partitioning=False) for path in files])
        .select(["open_time", "close"])
        .with_columns(pl.col("open_time").cast(pl.Datetime("us", "UTC")))
        .sort("open_time")
        .unique(subset=["open_time"], keep="first")
    )
    return df.with_columns(
        pl.col("open_time").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("_key")
    )


def bar_key(value: object) -> str:
    """Normalise an ``open_time`` to the key `load_closes` produces."""
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1]
    # Drop a timezone offset and any sub-second part; both sides are UTC.
    if "+" in text[10:]:
        text = text[: 10 + text[10:].index("+")]
    if "." in text:
        text = text[: text.index(".")]
    if "T" not in text and len(text) == 10:
        return f"{text}T00:00:00"
    return text[:19]


def load_run_predictions_in_range(
    run: RunRef, start: str, end: str
) -> list[dict[str, Any]]:
    """A run's predictions restricted to [start, end], inclusive, by date."""
    return [row for row in load_run_predictions(run) if start <= str(row["date"])[:10] <= end]


def eval_stability(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    model_type: str | None = None,
    timestamp: str | None = None,
    interval: str | None = None,
) -> dict[str, Any]:
    """Month-by-month forecast quality for one versioned prediction run.

    *start* and *end* are ``YYYY-MM-DD``, as everywhere else in `eval`; the old
    command took ``YYYY-MM`` here and ``YYYY-MM-DD`` for its sibling backtest.

    Untrained horizons contribute nothing (invariant 3), and a bar whose
    realised forward return is exactly zero has no side to score.
    """
    if settings is None:
        settings = get_settings()

    run = resolve_run(
        symbol, model_type=model_type, timestamp=timestamp, interval=interval, settings=settings
    )

    # Bounds are parsed so a malformed date fails here rather than silently
    # matching nothing through string comparison.
    start_iso = parse_date(start).date().isoformat()
    end_iso = parse_date(end).date().isoformat()

    bars = load_closes(symbol, run.interval, settings)
    closes = bars["close"].to_list()
    key_to_idx = {key: i for i, key in enumerate(bars["_key"].to_list())}

    predictions = load_run_predictions_in_range(run, start_iso, end_iso)
    if not predictions:
        raise FileNotFoundError(f"Run {run.slug} has no predictions in {start_iso}..{end_iso}")

    # Horizons are calendar spans converted to bars of the run's own interval
    # (invariant 2), so a 1h run and a 1d run of `long` both mean one year.
    forward_steps = {h.name: h.steps(run.interval) for h in DEFAULT_HORIZONS}

    # month -> horizon -> accumulators
    months: dict[str, dict[str, dict[str, float]]] = {}
    totals: dict[str, dict[str, float]] = {
        name: {"n": 0.0, "logloss": 0.0, "brier": 0.0, "correct": 0.0} for name in HORIZON_NAMES
    }

    for pred in predictions:
        # Keyed on open_time, not date: a sub-daily run puts many bars on one
        # calendar date and they are different forecasts (invariant 4).
        index = key_to_idx.get(bar_key(pred.get("open_time") or pred["date"]))
        if index is None:
            continue
        month_key = str(pred["date"])[:7]

        for name in HORIZON_NAMES:
            horizon = pred.get(name)
            if not horizon or horizon.get("trained") is False:
                continue

            ahead = index + forward_steps[name]
            if ahead >= len(closes) or not closes[index]:
                continue
            realised = direction_from_return(closes[ahead] / closes[index] - 1.0)
            if realised is None:
                continue

            probs = {
                key[2:]: float(value)
                for key, value in horizon.items()
                if key.startswith("P_") and value is not None
            }
            if not probs:
                continue
            logloss, brier = multiclass_logloss_brier(probs, realised)
            predicted = max(probs, key=lambda cls: probs[cls])

            bucket = months.setdefault(month_key, {}).setdefault(
                name, {"n": 0.0, "logloss": 0.0, "brier": 0.0, "correct": 0.0}
            )
            for target in (bucket, totals[name]):
                target["n"] += 1
                target["logloss"] += logloss
                target["brier"] += brier
                target["correct"] += 1.0 if predicted == realised else 0.0

    def finalise(bucket: dict[str, float]) -> dict[str, Any]:
        n = int(bucket["n"])
        if n == 0:
            return {"samples": 0, "logloss": None, "brier": None, "accuracy": None}
        return {
            "samples": n,
            "logloss": bucket["logloss"] / n,
            "brier": bucket["brier"] / n,
            "accuracy": bucket["correct"] / n,
        }

    series = [
        {
            "month": month_key,
            "horizons": {
                name: {
                    **finalise(months[month_key].get(name, {"n": 0.0})),
                    "sparse": 0 < int(months[month_key].get(name, {"n": 0.0})["n"])
                    < MIN_MONTH_SAMPLES,
                }
                for name in HORIZON_NAMES
            },
        }
        for month_key in sorted(months)
    ]

    report = {
        "symbol": symbol,
        "run": run.to_dict(),
        "start": start_iso,
        "end": end_iso,
        "generated_at": datetime.now(UTC).isoformat(),
        "min_month_samples": MIN_MONTH_SAMPLES,
        # The whole-run figures are here only so a reader can see what the
        # monthly series is varying around. `eval predictions` remains the
        # authority on run-level quality, and `eval backtest` on returns.
        "overall": {name: finalise(totals[name]) for name in HORIZON_NAMES},
        "months": series,
    }

    out_path = get_eval_stability_summary_path(symbol, settings=settings, run_slug=run.slug)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    report["report_path"] = str(out_path)
    return report
