"""Build direction and depth labels for daily predictions."""

from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DIRECTION_CLASSES,
    assign_depth_bin,
    direction_from_return,
    parse_depth_bins,
)
from prosper.storage.layout import get_labels_parquet_path
from prosper.storage.parquet import load_parquet, save_parquet


def build_labels(
    symbol: str,
    base_interval: str = "1d",
    forward_days: int = 1,
    depth_bins_str: str = DEFAULT_DEPTH_BINS_STR,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Build direction and depth labels from klines.

    Direction is the sign of the forward return: `long` or `short`. There is no
    flat class and no threshold — a label says what the market did, and whether
    that is worth trading after costs is a decision made later, against
    `Settings.planner_round_trip_cost`. How far the move went is the depth
    label's job, and its lowest bin covers what `flat` used to absorb.

    Args:
        symbol: Trading symbol
        base_interval: Base interval for labels (e.g., "1d")
        forward_days: Forward horizon in days (N)
        depth_bins_str: Comma-separated depth bin definitions
        settings: Settings instance (defaults to global)

    Returns:
        Dictionary with label statistics
    """
    if settings is None:
        settings = get_settings()

    # Load daily klines

    # Need to load all daily data, not just one month
    # For simplicity, we'll load from processed directory
    base_path = settings.processed_binance_spot_klines_dir / base_interval / f"symbol={symbol}"

    if not base_path.exists():
        return {"error": f"No data found for {symbol} at interval {base_interval}"}

    # Load all Parquet files
    parquet_files = list(base_path.glob("**/*.parquet"))
    if not parquet_files:
        return {"error": f"No Parquet files found for {symbol} at interval {base_interval}"}

    dfs: list[pl.DataFrame] = []
    for pf in parquet_files:
        df = load_parquet(pf)
        if not df.is_empty():
            dfs.append(df)

    if not dfs:
        return {"error": "No data loaded"}

    df_all = pl.concat(dfs).sort("open_time").unique(subset=["open_time"], keep="first")

    if len(df_all) < 2:
        return {"error": "Insufficient data for labels (need at least 2 days)"}

    forward_days = int(forward_days)
    if forward_days < 1:
        raise ValueError("forward_days must be >= 1")

    # forward return: r_fwd = close[t+N]/close[t] - 1
    df_all = df_all.with_columns(
        (pl.col("close").shift(-forward_days) / pl.col("close") - 1.0).alias("return_fwd")
    )
    # Remove tail rows with null forward return
    df_all = df_all.filter(pl.col("return_fwd").is_not_null())

    # Parse depth bins
    depth_bins, _depth_labels = parse_depth_bins(depth_bins_str)

    # Build labels
    def assign_direction(return_val: float) -> str | None:
        return direction_from_return(return_val)

    def assign_depth(return_val: float, direction: str | None) -> int | None:
        """Assign a depth bin. Every classifiable row has one."""
        if direction is None:
            return None
        return assign_depth_bin(abs(return_val) * 100.0, depth_bins)

    # Assign direction labels
    df_labels = df_all.with_columns(
        pl.col("return_fwd")
        .map_elements(assign_direction, return_dtype=pl.Utf8)
        .alias("direction"),
    )

    # Assign depth bins - need to use a wrapper function for map_elements
    def assign_depth_wrapper(row: dict[str, Any]) -> int | None:
        """Wrapper for assign_depth to work with map_elements."""
        return_val = row.get("return_fwd", 0.0)
        return assign_depth(return_val, row.get("direction"))

    df_labels = df_labels.with_columns(
        pl.struct(["return_fwd", "direction"])
        .map_elements(assign_depth_wrapper, return_dtype=pl.Int64)
        .alias("depth_bin"),
    )

    # Select relevant columns
    df_labels = df_labels.select(
        [
            "open_time",
            "close",
            "return_fwd",
            "direction",
            "depth_bin",
        ]
    )

    # Calculate statistics
    direction_counts = df_labels["direction"].value_counts().to_dicts()
    direction_stats = {row["direction"]: row["count"] for row in direction_counts}

    depth_counts = (
        df_labels.filter(pl.col("depth_bin").is_not_null())["depth_bin"].value_counts().to_dicts()
    )
    depth_stats = {str(row["depth_bin"]): row["count"] for row in depth_counts}

    total = len(df_labels)
    stats = {
        "symbol": symbol,
        "base_interval": base_interval,
        "forward_days": forward_days,
        "total_labels": total,
        "direction_distribution": {d: direction_stats.get(d, 0) for d in DIRECTION_CLASSES},
        "direction_percentages": {
            d: direction_stats.get(d, 0) / total * 100.0 for d in DIRECTION_CLASSES
        },
        "depth_distribution": depth_stats,
        "imbalance_ratio": (
            max(direction_stats.get("long", 0), direction_stats.get("short", 0))
            / max(min(direction_stats.get("long", 0), direction_stats.get("short", 0)), 1)
            if min(direction_stats.get("long", 0), direction_stats.get("short", 0)) > 0
            else float("inf")
        ),
    }

    # Save labels
    labels_path = get_labels_parquet_path(symbol, base_interval, settings=settings)
    save_parquet(df_labels, labels_path)

    return stats
