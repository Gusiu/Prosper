"""Write every horizon's labels for one symbol and interval to a parquet file.

The file is a view of the same labels the models train on, computed by the same
function (`prosper.labels.forward.forward_labels`), for the dashboard to draw
and for anyone to inspect. The models do not read it back: each predictor
labels the bars it loads, so a stale or missing file can never change a
forecast.

It holds exactly the models' horizons, converted to bars of the file's own
interval: shifting by rows rather than by calendar span would make "one day"
one hour on hourly data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    DIR_TO_IDX,
    DIRECTION_CLASSES,
    IDX_TO_DIR,
    HorizonSpec,
    parse_depth_bins,
)
from prosper.labels.forward import UNKNOWN, forward_labels
from prosper.storage.layout import get_labels_parquet_path
from prosper.storage.parquet import load_parquet, save_parquet


def label_columns(horizon: str) -> tuple[str, str, str]:
    """Names of one horizon's return, direction and depth columns."""
    return f"return_{horizon}", f"direction_{horizon}", f"depth_{horizon}"


def _load_klines(symbol: str, interval: str, settings: Settings) -> pl.DataFrame | None:
    base_path = settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
    frames = [
        frame
        for path in sorted(base_path.glob("**/*.parquet"))
        if not (frame := load_parquet(path).select("open_time", "close")).is_empty()
    ]
    if not frames:
        return None
    # Deduplicate, then sort. The other order relied on `unique` keeping rows
    # where `sort` put them, which polars does not promise.
    return pl.concat(frames, how="diagonal_relaxed").unique(subset=["open_time"], keep="first").sort("open_time")


def build_labels(
    symbol: str,
    base_interval: str = "1d",
    depth_bins_str: str = DEFAULT_DEPTH_BINS_STR,
    settings: Settings | None = None,
    horizons: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """Label every bar of *symbol* at *base_interval* for every horizon.

    Returns per-horizon statistics, or ``{"error": ...}`` when there is nothing
    to label.
    """
    if settings is None:
        settings = get_settings()

    klines = _load_klines(symbol, base_interval, settings)
    if klines is None:
        return {"error": f"No data found for {symbol} at interval {base_interval}"}
    if len(klines) < 2:
        return {"error": f"Insufficient data for labels: {len(klines)} bar(s)"}

    depth_bins, depth_labels = parse_depth_bins(depth_bins_str)
    steps = {h.name: h.steps(base_interval) for h in horizons}
    labels = forward_labels(klines["close"].to_numpy(), steps, depth_bins)

    # Index -1 (`UNKNOWN`) selects the trailing None, so a lookup is one take.
    side_names = np.array([IDX_TO_DIR[i] for i in range(len(IDX_TO_DIR))] + [None], dtype=object)
    bin_names = np.array([*depth_labels, None], dtype=object)
    assert UNKNOWN == -1

    columns: list[pl.Series] = []
    by_horizon: dict[str, Any] = {}
    for name, forward in steps.items():
        return_col, direction_col, depth_col = label_columns(name)
        columns += [
            pl.Series(return_col, labels.returns[name]).fill_nan(None),
            pl.Series(direction_col, side_names[labels.direction[name]], dtype=pl.Utf8),
            pl.Series(depth_col, bin_names[labels.depth[name]], dtype=pl.Utf8),
        ]

        counts = {
            side: int(np.count_nonzero(labels.direction[name] == DIR_TO_IDX[side]))
            for side in DIRECTION_CLASSES
        }
        labelled = sum(counts.values())
        by_horizon[name] = {
            "forward_bars": forward,
            "labelled": labelled,
            "direction_distribution": counts,
            "long_share": counts["long"] / labelled if labelled else None,
        }

    frame = klines.with_columns(columns)
    save_parquet(frame, get_labels_parquet_path(symbol, base_interval, settings=settings))

    return {
        "symbol": symbol,
        "base_interval": base_interval,
        "bars": len(frame),
        "depth_bins": depth_labels,
        "horizons": by_horizon,
    }
