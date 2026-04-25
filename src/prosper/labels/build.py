"""Build direction and depth labels for daily predictions."""

import json
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_labels_parquet_path
from prosper.storage.parquet import load_parquet, save_parquet


def parse_depth_bins(bins_str: str) -> list[tuple[float, float | None]]:
    """
    Parse depth bin string into list of (min, max) tuples.

    Format: "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+"
    Returns: [(1.0, 2.0), (2.0, 3.0), ..., (34.0, None)]

    Args:
        bins_str: Comma-separated bin definitions

    Returns:
        List of (min, max) tuples, where None means infinity
    """
    bins: list[tuple[float, float | None]] = []
    parts = bins_str.split(",")

    for part in parts:
        part = part.strip()
        if part.endswith("+"):
            # Open-ended bin
            min_val = float(part[:-1])
            bins.append((min_val, None))
        elif "-" in part:
            # Range bin
            min_str, max_str = part.split("-", 1)
            bins.append((float(min_str), float(max_str)))
        else:
            raise ValueError(f"Invalid bin format: {part}")

    return bins


def assign_depth_bin(value: float, bins: list[tuple[float, float | None]]) -> int | None:
    """
    Assign value to depth bin index.

    Args:
        value: Value to bin (absolute return percentage)
        bins: List of (min, max) tuples

    Returns:
        Bin index (0-based) or None if no match
    """
    for idx, (min_val, max_val) in enumerate(bins):
        if max_val is None:
            if value >= min_val:
                return idx
        else:
            if min_val <= value < max_val:
                return idx
    return None


def build_labels(
    symbol: str,
    base_interval: str = "1d",
    forward_days: int = 1,
    flat_threshold: float = 0.01,
    depth_bins_str: str = "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Build direction and depth labels from daily klines.

    direction labels for forward horizon:
    - flat: |r_fwd| <= flat_threshold
    - long: r_fwd > flat_threshold
    - short: r_fwd < -flat_threshold

    Depth labels (conditional):
    - If long: bin based on |r_fwd|
    - If short: bin based on |r_fwd|
    - If flat: depth = None

    Args:
        symbol: Trading symbol
        base_interval: Base interval for labels (e.g., "1d")
        forward_days: Forward horizon in days (N)
        flat_threshold: Threshold for flat label (default 0.01 = 1%)
        depth_bins_str: Comma-separated depth bin definitions
        settings: Settings instance (defaults to global)

    Returns:
        Dictionary with label statistics
    """
    if settings is None:
        settings = get_settings()

    # Load daily klines
    from prosper.storage.layout import get_parquet_file_path

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
    depth_bins = parse_depth_bins(depth_bins_str)

    # Build labels
    def assign_direction(return_val: float) -> str:
        """Assign direction label."""
        abs_return = abs(return_val)
        if abs_return <= flat_threshold:
            return "flat"
        elif return_val > flat_threshold:
            return "long"
        else:
            return "short"

    def assign_depth(return_val: float, direction: str) -> int | None:
        """Assign depth bin."""
        if direction == "flat":
            return None
        abs_return_pct = abs(return_val) * 100.0  # Convert to percentage
        return assign_depth_bin(abs_return_pct, depth_bins)

    # Assign direction labels
    df_labels = df_all.with_columns(
        pl.col("return_fwd").map_elements(assign_direction, return_dtype=pl.Utf8).alias("direction"),
    )

    # Assign depth bins - need to use a wrapper function for map_elements
    def assign_depth_wrapper(row: dict[str, Any]) -> int | None:
        """Wrapper for assign_depth to work with map_elements."""
        return_val = row.get("return_fwd", 0.0)
        direction = row.get("direction", "flat")
        return assign_depth(return_val, direction)

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
        df_labels.filter(pl.col("depth_bin").is_not_null())["depth_bin"]
        .value_counts()
        .to_dicts()
    )
    depth_stats = {str(row["depth_bin"]): row["count"] for row in depth_counts}

    total = len(df_labels)
    stats = {
        "symbol": symbol,
        "base_interval": base_interval,
        "forward_days": forward_days,
        "flat_threshold": flat_threshold,
        "total_labels": total,
        "direction_distribution": {
            "long": direction_stats.get("long", 0),
            "flat": direction_stats.get("flat", 0),
            "short": direction_stats.get("short", 0),
        },
        "direction_percentages": {
            "long": direction_stats.get("long", 0) / total * 100.0,
            "flat": direction_stats.get("flat", 0) / total * 100.0,
            "short": direction_stats.get("short", 0) / total * 100.0,
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
