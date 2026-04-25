"""Baseline prediction model using rolling window frequencies."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_prediction_report_path
from prosper.storage.parquet import load_parquet
from prosper.utils.time import parse_date, timestamp_to_utc_datetime


def parse_depth_bins(bins_str: str) -> list[tuple[float, float | None]]:
    """Parse depth bin string (same as in labels.build)."""
    bins: list[tuple[float, float | None]] = []
    parts = bins_str.split(",")

    for part in parts:
        part = part.strip()
        if part.endswith("+"):
            min_val = float(part[:-1])
            bins.append((min_val, None))
        elif "-" in part:
            min_str, max_str = part.split("-", 1)
            bins.append((float(min_str), float(max_str)))
        else:
            raise ValueError(f"Invalid bin format: {part}")

    return bins


def calculate_rolling_frequencies(
    df_labels: pl.DataFrame,
    current_date: datetime,
    window_days: int = 180,
) -> dict[str, Any]:
    """
    Calculate rolling window frequencies for direction and depth.

    Args:
        df_labels: DataFrame with labels (open_time, direction, depth_bin)
        current_date: Current date for rolling window
        window_days: Rolling window size in days

    Returns:
        Dictionary with probabilities
    """
    if df_labels.is_empty():
        return {}

    # Filter to rolling window
    window_start = current_date - timedelta(days=window_days)
    df_window = df_labels.filter(
        (pl.col("open_time") >= window_start) & (pl.col("open_time") < current_date)
    )

    if df_window.is_empty():
        return {}

    total = len(df_window)

    # Direction frequencies
    direction_counts = df_window["direction"].value_counts().to_dicts()
    direction_freq = {row["direction"]: row["count"] / total for row in direction_counts}

    # Depth frequencies (conditional on direction)
    df_long = df_window.filter(pl.col("direction") == "long")
    df_short = df_window.filter(pl.col("direction") == "short")

    depth_long_counts = (
        df_long.filter(pl.col("depth_bin").is_not_null())["depth_bin"].value_counts().to_dicts()
        if not df_long.is_empty()
        else []
    )
    depth_short_counts = (
        df_short.filter(pl.col("depth_bin").is_not_null())["depth_bin"].value_counts().to_dicts()
        if not df_short.is_empty()
        else []
    )

    depth_long_freq = {str(row["depth_bin"]): row["count"] / len(df_long) for row in depth_long_counts} if len(df_long) > 0 else {}
    depth_short_freq = {str(row["depth_bin"]): row["count"] / len(df_short) for row in depth_short_counts} if len(df_short) > 0 else {}

    return {
        "P_long": direction_freq.get("long", 0.0),
        "P_flat": direction_freq.get("flat", 0.0),
        "P_short": direction_freq.get("short", 0.0),
        "depth_long_bins": depth_long_freq,
        "depth_short_bins": depth_short_freq,
        "window_size": total,
    }


def predict_baseline(
    symbol: str,
    start: str,
    end: str,
    horizon: str = "short",
    window_days: int = 180,
    depth_bins_str: str = "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Generate baseline predictions for daily labels.

    Args:
        symbol: Trading symbol
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        horizon: Prediction horizon (short/medium/long) - not used in baseline
        window_days: Rolling window size in days
        depth_bins_str: Depth bin definitions
        settings: Settings instance (defaults to global)

    Returns:
        Dictionary with prediction results
    """
    if settings is None:
        settings = get_settings()

    from prosper.storage.layout import get_labels_parquet_path

    # Load labels
    labels_path = get_labels_parquet_path(symbol, "1d", settings=settings)
    if not labels_path.exists():
        return {"error": f"Labels not found for {symbol}. Run 'prosper labels build' first."}

    df_labels = load_parquet(labels_path)

    if df_labels.is_empty():
        return {"error": "Empty labels DataFrame"}

    # Parse dates
    start_date = parse_date(start)
    end_date = parse_date(end)

    # Generate predictions for each day
    predictions: list[dict[str, Any]] = []
    current_date = start_date

    while current_date <= end_date:
        # Calculate rolling frequencies
        probs = calculate_rolling_frequencies(df_labels, current_date, window_days)

        if probs:
            pred = {
                "date": current_date.strftime("%Y-%m-%d"),
                "P_long": probs.get("P_long", 0.0),
                "P_flat": probs.get("P_flat", 0.0),
                "P_short": probs.get("P_short", 0.0),
                "depth_long_bins": probs.get("depth_long_bins", {}),
                "depth_short_bins": probs.get("depth_short_bins", {}),
                "window_size": probs.get("window_size", 0),
            }
            predictions.append(pred)

        current_date += timedelta(days=1)

    # Save predictions to JSONL
    year = start_date.year
    month = start_date.month
    report_path = get_prediction_report_path(symbol, year, month, settings=settings)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, sort_keys=True) + "\n")

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "predictions_count": len(predictions),
        "report_path": str(report_path),
    }
