"""The benchmark: smoothed frequencies of past outcomes, no model at all."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import DEFAULT_DEPTH_BINS_STR, DIR_TO_IDX, DIRECTION_CLASSES, parse_depth_bins
from prosper.labels.forward import UNKNOWN, ForwardLabels, forward_labels
from prosper.predict.walk_forward import Month, RunPlan, plan_run, run
from prosper.predict.window import MIN_TRAIN_SAMPLES
from prosper.storage.parquet import load_parquet


def _laplace_prob(count: int, total: int, k: int, alpha: float) -> float:
    return (count + alpha) / (total + alpha * k) if total >= 0 else 0.0


class BaselineForecaster:
    """Laplace-smoothed frequencies over the same window a model would train on.

    Nothing is fitted, so there is nothing to do at the start of a month: each
    bar counts the outcomes that were already realised before it. Counting the
    most recent ``forward_days`` bars would mean reading returns that had not
    happened yet, which is exactly the advantage a benchmark must not have.
    """

    name = "baseline"

    def __init__(self, labels: ForwardLabels, depth_labels: list[str], alpha: float) -> None:
        self.labels = labels
        self.depth_labels = depth_labels
        self.alpha = alpha
        self.plan: RunPlan | None = None

    def fit_month(self, plan: RunPlan, month: Month, bars: list[int]) -> None:
        self.plan = plan

    def predict(self, bar: int, horizon: str) -> dict[str, Any] | None:
        assert self.plan is not None
        # Only outcomes realised strictly before this bar are observable.
        start, end = self.plan.windows.bounds(bar, horizon)
        sides = self.labels.direction[horizon][start:end]
        depths = self.labels.depth[horizon][start:end]
        known = sides != UNKNOWN
        total = int(np.count_nonzero(known))
        if total < MIN_TRAIN_SAMPLES:
            return None

        k_dir, k_depth = len(DIRECTION_CLASSES), len(self.depth_labels)
        payload: dict[str, Any] = {
            f"P_{side}": _laplace_prob(
                int(np.count_nonzero(sides == DIR_TO_IDX[side])), total, k_dir, self.alpha
            )
            for side in DIRECTION_CLASSES
        }
        for side in ("long", "short"):
            in_side = depths[(sides == DIR_TO_IDX[side]) & (depths != UNKNOWN)]
            counts = np.bincount(in_side, minlength=k_depth)
            payload[f"depth_{side}_bins"] = {
                label: _laplace_prob(int(counts[b]), len(in_side), k_depth, self.alpha)
                for b, label in enumerate(self.depth_labels)
            }
        payload["trained"] = True
        return payload

    def summary(self) -> dict[str, Any]:
        return {}

    def after_write(self, run_dir: Path) -> None:
        return None


def _load_klines(symbol: str, interval: str, settings: Settings) -> pl.DataFrame | None:
    """Every stored month, not just the requested range.

    The frequency window looks back from the first predicted bar. Loading only
    [start, end] left the early rows with almost no history, so the benchmark
    got weaker the narrower the range - while the models, which read all their
    features and filter on output, did not.
    """
    base = settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
    parts = [
        load_parquet(path)
        for path in sorted(base.glob("**/*.parquet"))
        if path.is_file() and path.stat().st_size > 0
    ]
    if not parts:
        return None
    # `maintain_order` keeps the sort: polars does not otherwise promise it.
    return (
        pl.concat(parts)
        .sort("open_time")
        .unique(subset=["open_time"], keep="first", maintain_order=True)
        .with_columns(pl.col("open_time").dt.date().alias("_date"))
    )


def predict_baseline(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    horizons_selected: str | None = None,
    rolling_window_days: int | None = None,
    train_window_by_horizon: str | None = None,
    depth_bins_str: str = DEFAULT_DEPTH_BINS_STR,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Baseline probabilistic predictions: rolling empirical frequencies with Laplace smoothing.

    Available at every interval the models are, since everything below is in
    bars of *interval*: pinning it to 1d only meant a 1h comparison had no
    benchmark to be measured against.
    """
    if settings is None:
        settings = get_settings()
    if rolling_window_days is None:
        rolling_window_days = settings.baseline_rolling_window_days

    bars = _load_klines(symbol, interval, settings)
    if bars is None or bars.is_empty():
        return {"error": f"No {interval} parquet found for {symbol}", "symbol": symbol}

    depth_bins, depth_labels = parse_depth_bins(depth_bins_str)
    # The benchmark counts frequencies over the same window the models train
    # on, or it would not be answering the same question.
    plan = plan_run(
        symbol,
        interval,
        start,
        end,
        open_times=bars["open_time"].to_list(),
        dates=bars["_date"].to_list(),
        depth_labels=depth_labels,
        train_window_days=rolling_window_days,
        train_window_by_horizon=train_window_by_horizon,
        horizons_selected=horizons_selected,
    )
    labels = forward_labels(bars["close"].to_list(), plan.steps, depth_bins)
    return run(BaselineForecaster(labels, depth_labels, alpha), plan, settings)
