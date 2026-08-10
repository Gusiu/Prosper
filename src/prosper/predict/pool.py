"""Training rows drawn from several symbols at once.

The project's binding constraint is not model capacity, it is independent
observations: a 364-day horizon inside nine years of one symbol is worth about
three of them, and no window, optimiser or epoch budget changes that. Every
other lever trades one side of the problem for the other — widening the training
window takes bars away from the evaluation, narrowing it takes them away from the
fit. Pooling symbols is the only lever that adds observations without spending
any, because a second symbol brings its own history rather than borrowing from
the first's.

Two things make it correct rather than merely convenient.

**Rows are selected by timestamp, never by row index.** Every predictor here
slices its training set with `training_bounds`, which is arithmetic on positions
in one sorted frame. That is exact for one symbol and meaningless across
several: row 400 of BTCUSDT is September 2018 and row 400 of SOLUSDT is
September 2021, so an index-based window would train a 2021 forecast on rows
that had not happened yet. The bounds here are instants — everything from
``cutoff - window`` up to ``cutoff - horizon`` — and each symbol contributes
whatever it happens to have in that span, which is nothing at all for a symbol
that had not launched yet.

**The pooled slice is ordered by time.** The calibrator is fitted on the newest
part of the training region, and "newest" has to mean newest overall. Stacking
symbols end to end and taking the last rows would hand the calibrator one
symbol's recent history and call it a held-out slice.

Pooling only means anything because the feature set is scale-free
(`domain.TRAINING_FEATURE_COLUMNS`). Before that, `ma_10` differed by 553x
between BTCUSDT and SOLUSDT and a pooled model would have spent its capacity
learning which symbol it was looking at.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl

from prosper.config import Settings
from prosper.domain import (
    DIR_TO_IDX,
    assign_depth_bin,
    direction_from_return,
    effective_symbols,
    interval_minutes,
    training_features,
)
from prosper.predict.window import TrainingWindows
from prosper.storage.layout import get_features_parquet_path


@dataclass(frozen=True)
class SymbolFrame:
    """One symbol's features, labels and timestamps, sorted ascending."""

    symbol: str
    # For slicing: UTC instants with the zone dropped, because numpy comparison
    # needs a naive dtype. Never write these out.
    open_time: np.ndarray  # datetime64[us]
    # For writing: the timestamps exactly as the parquet holds them, zone and
    # all. Predictions are keyed by `open_time` (invariant 4), and a run whose
    # keys lost their `+00:00` no longer joins against one that kept it — which
    # is a corrupted run, not a formatting quibble.
    timestamps: list[datetime]
    close: np.ndarray
    features: np.ndarray
    # Per horizon: direction class index, or -1 where the outcome is unrealised
    # or the return is exactly zero (invariant 9: a zero return has no side).
    direction: dict[str, np.ndarray]
    depth: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.open_time)


@dataclass(frozen=True)
class PooledSlice:
    """The training rows for one horizon at one instant, oldest first."""

    features: np.ndarray
    direction: np.ndarray
    depth: np.ndarray
    open_time: np.ndarray
    symbol: np.ndarray

    def __len__(self) -> int:
        return len(self.direction)

    @property
    def symbol_counts(self) -> dict[str, int]:
        names, counts = np.unique(self.symbol, return_counts=True)
        return {str(name): int(count) for name, count in zip(names, counts, strict=True)}


class TrainingPool:
    """Several symbols' frames, sliceable by instant.

    The pool always contains the symbol being predicted; the others only ever
    contribute training rows. Nothing here reads a bar at or after the cutoff, so
    invariant 6 holds per symbol and therefore across the pool.
    """

    def __init__(self, frames: Sequence[SymbolFrame], interval: str) -> None:
        if not frames:
            raise ValueError("A training pool needs at least one symbol")
        self.frames = tuple(frames)
        self.interval = interval
        self._minutes = interval_minutes(interval)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(frame.symbol for frame in self.frames)

    def span(self, steps: int) -> timedelta:
        """The real duration of *steps* bars at this interval."""
        return timedelta(minutes=self._minutes * steps)

    def training_slice(
        self,
        cutoff: datetime,
        horizon: str,
        windows: TrainingWindows,
    ) -> PooledSlice:
        """Every pooled row usable for *horizon* when predicting at *cutoff*.

        A label at instant ``t`` needs the close at ``t + horizon``, so it is
        usable only once ``t + horizon < cutoff``. The window reaches back
        ``window`` before the cutoff. Both bounds are instants, which is what
        lets symbols with different launch dates share a training set.
        """
        forward = self.span(windows.forward_steps[horizon])
        window = self.span(windows.window_steps[horizon])
        at = _as_utc_naive(cutoff)
        oldest = np.datetime64(at - window, "us")
        # Strictly before, matching `training_bounds`' exclusive end.
        newest = np.datetime64(at - forward, "us")

        features: list[np.ndarray] = []
        direction: list[np.ndarray] = []
        depth: list[np.ndarray] = []
        times: list[np.ndarray] = []
        symbols: list[np.ndarray] = []

        for frame in self.frames:
            keep = (frame.open_time >= oldest) & (frame.open_time < newest)
            if not keep.any():
                continue
            features.append(frame.features[keep])
            direction.append(frame.direction[horizon][keep])
            depth.append(frame.depth[horizon][keep])
            times.append(frame.open_time[keep])
            symbols.append(np.full(int(keep.sum()), frame.symbol, dtype=object))

        if not features:
            width = self.frames[0].features.shape[1]
            return PooledSlice(
                features=np.empty((0, width), dtype=self.frames[0].features.dtype),
                direction=np.empty(0, dtype=np.int64),
                depth=np.empty(0, dtype=np.int64),
                open_time=np.empty(0, dtype="datetime64[us]"),
                symbol=np.empty(0, dtype=object),
            )

        stacked_times = np.concatenate(times)
        # Chronological, and stable so a tie between symbols always resolves the
        # same way — otherwise two runs of the same configuration could order the
        # calibration split differently and stop disagreeing only by luck.
        order = np.argsort(stacked_times, kind="stable")

        return PooledSlice(
            features=np.concatenate(features)[order],
            direction=np.concatenate(direction)[order],
            depth=np.concatenate(depth)[order],
            open_time=stacked_times[order],
            symbol=np.concatenate(symbols)[order],
        )

    def return_correlation(self, forward_steps: int) -> float:
        """Mean pairwise correlation of forward returns across the pool.

        This is what decides whether pooling bought anything. `effective_symbols`
        turns it into a count, and the count is nothing like the number of
        symbols: at 0.76 — the measured figure for annual returns here — three
        symbols are worth 1.19, and no number of them exceeds 1.3.

        Returns 0.0 for a pool of one, where there is nothing to correlate and
        pooling is a no-op by construction.
        """
        if len(self.frames) < 2:
            return 0.0

        series: list[dict[np.datetime64, float]] = []
        for frame in self.frames:
            close = frame.close
            usable = len(close) - forward_steps
            if usable <= 1:
                continue
            ahead = close[forward_steps:]
            base = close[:usable]
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = np.where(base != 0, ahead / base - 1.0, np.nan)
            series.append(dict(zip(frame.open_time[:usable], returns, strict=True)))

        correlations: list[float] = []
        for left, right in combinations(series, 2):
            shared = sorted(set(left) & set(right))
            if len(shared) < 30:
                continue
            x = np.array([left[key] for key in shared])
            y = np.array([right[key] for key in shared])
            usable = np.isfinite(x) & np.isfinite(y)
            if usable.sum() < 30 or x[usable].std() == 0 or y[usable].std() == 0:
                continue
            correlations.append(float(np.corrcoef(x[usable], y[usable])[0, 1]))

        return float(np.mean(correlations)) if correlations else 0.0


def build_frame(
    symbol: str,
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    horizon_steps: dict[str, int],
    depth_bins: Sequence[float],
) -> SymbolFrame:
    """Turn one symbol's feature frame into labelled arrays.

    Labels are computed inside the symbol — the close *k* bars ahead of a BTCUSDT
    bar is a BTCUSDT close — so pooling never mixes one symbol's price path into
    another's outcome.
    """
    close = df["close"].to_numpy().astype(float)
    n = len(close)
    direction: dict[str, np.ndarray] = {}
    depth: dict[str, np.ndarray] = {}

    for name, forward in horizon_steps.items():
        dir_arr = np.full(n, -1, dtype=np.int64)
        depth_arr = np.full(n, -1, dtype=np.int64)
        for i in range(n - forward):
            base, ahead = close[i], close[i + forward]
            if not base or not ahead:
                continue
            change = ahead / base - 1.0
            side = direction_from_return(change)
            if side is None:
                continue
            dir_arr[i] = DIR_TO_IDX[side]
            bin_idx = assign_depth_bin(abs(change) * 100.0, depth_bins)
            if bin_idx is not None:
                depth_arr[i] = bin_idx
        direction[name] = dir_arr
        depth[name] = depth_arr

    # Via `to_list`, not `to_numpy`: the zero-copy path segfaults the
    # interpreter on this build's Datetime columns. Every other module here
    # already goes through `to_list` for `open_time`, which is presumably how
    # the project has never hit it.
    timestamps = df["open_time"].to_list()

    return SymbolFrame(
        symbol=symbol,
        # datetime64 cannot carry a zone, so normalise to UTC first and drop it
        # deliberately rather than letting numpy drop it with a warning. Symbols
        # are pooled by instant, so they must all be on one clock.
        open_time=np.array(
            [_as_utc_naive(value) for value in timestamps], dtype="datetime64[us]"
        ),
        timestamps=timestamps,
        close=close,
        features=df.select(list(feature_cols)).to_numpy(),
        direction=direction,
        depth=depth,
    )


def _as_utc_naive(value: datetime) -> datetime:
    """A tz-aware instant as naive UTC; a naive one is assumed to be UTC already.

    Binance publishes in UTC and the lake stores what it publishes, so the
    assumption holds — but it is written down here rather than left implicit in
    a `datetime64` cast that silently discards whatever zone it was given.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def load_pool(
    symbols: Iterable[str],
    interval: str,
    horizon_steps: dict[str, int],
    depth_bins: Sequence[float],
    settings: Settings,
    feature_cols: Sequence[str] | None = None,
) -> tuple[TrainingPool, list[str]]:
    """Load every symbol's features into one pool.

    Returns the pool and the feature columns it settled on. A symbol whose
    parquet is missing a column the others have would silently change what the
    model sees per symbol, so the columns are intersected once and used for all.
    """
    frames: list[SymbolFrame] = []
    loaded: list[pl.DataFrame] = []
    names: list[str] = []

    for symbol in symbols:
        path = get_features_parquet_path(symbol, interval, settings=settings)
        if not path.exists():
            raise FileNotFoundError(
                f"No features parquet for {symbol} at {interval}. Run `features build` for it "
                "before pooling it into a training set."
            )
        df = pl.read_parquet(path, hive_partitioning=False).sort("open_time")
        if df.is_empty():
            raise ValueError(f"The features parquet for {symbol} at {interval} is empty")
        loaded.append(df)
        names.append(symbol)

    shared = feature_cols
    if shared is None:
        available: set[str] | None = None
        for df in loaded:
            columns = set(training_features(df.columns))
            available = columns if available is None else (available & columns)
        # Keep the canonical order rather than a set's, so two runs agree.
        shared = training_features(sorted(available or set()))

    for symbol, df in zip(names, loaded, strict=True):
        frames.append(build_frame(symbol, df, shared, horizon_steps, depth_bins))

    return TrainingPool(frames, interval), list(shared)


def parse_symbols(raw: str | None, target: str) -> list[str]:
    """Parse ``'BTCUSDT,ETHUSDT'`` into a training pool containing *target*.

    The predicted symbol is always in the pool and always first: a run that
    trained on everything *except* the symbol it forecasts would be a different
    experiment, and one nobody asked for by leaving a flag out.
    """
    target = target.upper()
    if raw is None or not raw.strip():
        return [target]

    requested = [part.strip().upper() for part in raw.split(",") if part.strip()]
    pooled = [target] + [name for name in requested if name != target]
    seen: set[str] = set()
    return [name for name in pooled if not (name in seen or seen.add(name))]


def describe_pool(
    slices: dict[str, PooledSlice],
    pool: TrainingPool | None = None,
    forward_steps: dict[str, int] | None = None,
) -> dict[str, Any]:
    """What each symbol actually contributed, for the run summary.

    Worth recording because it is rarely what the flag says: a symbol that
    launched in 2020 contributes nothing to a 2018 window, and a pooled run whose
    extra symbols turn out to have supplied fifty rows is not the experiment its
    command line describes.

    With *pool* and *forward_steps* it also records what the pool was *worth*.
    Row counts flatter pooling badly — three symbols triple the rows while
    carrying 1.2 to 1.35 symbols' worth of independent information, because
    crypto returns share one dominant factor. A summary reporting only the rows
    would support a claim the data does not.
    """
    described: dict[str, Any] = {
        horizon: {"rows": len(slice_), "by_symbol": slice_.symbol_counts}
        for horizon, slice_ in slices.items()
    }
    if pool is None or forward_steps is None:
        return described

    for horizon, entry in described.items():
        steps = forward_steps.get(horizon)
        if steps is None:
            continue
        correlation = pool.return_correlation(steps)
        entry["return_correlation"] = round(correlation, 4)
        entry["effective_symbols"] = round(effective_symbols(len(pool.frames), correlation), 3)
    return described
