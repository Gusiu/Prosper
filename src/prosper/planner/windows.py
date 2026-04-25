"""Action window planner based on baseline predictions."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_prediction_report_path, get_recommendation_report_path
from prosper.utils.time import parse_date


def calculate_edge(p_long: float, p_short: float) -> float:
    """
    Calculate edge as difference between long and short probabilities.

    Args:
        p_long: Probability of long
        p_short: Probability of short

    Returns:
        Edge value (positive favors long, negative favors short)
    """
    return p_long - p_short


def calculate_risk_metric(
    p_long: float,
    p_short: float,
    depth_long_bins: dict[str, float],
    depth_short_bins: dict[str, float],
) -> float:
    """
    Calculate risk metric based on probability of large moves.

    Args:
        p_long: Probability of long
        p_short: Probability of short
        depth_long_bins: Depth bin probabilities for long
        depth_short_bins: Depth bin probabilities for short

    Returns:
        Risk metric (higher = more risk)
    """
    # Risk = probability mass in large depth bins.
    # Spec bins: 13-21, 21-34, 34+
    large_bins = {"13-21", "21-34", "34+"}

    long_large = sum(prob for bin_str, prob in depth_long_bins.items() if bin_str in large_bins)
    short_large = sum(prob for bin_str, prob in depth_short_bins.items() if bin_str in large_bins)

    return p_long * long_large + p_short * short_large


def map_to_recommendation(edge: float, risk: float) -> str:
    """
    Map edge and risk to recommendation.

    Recommendations:
    - Strong Buy: edge > 0.3, risk < 0.2
    - Buy: edge > 0.15, risk < 0.3
    - Accumulate: edge > 0.05, risk < 0.4
    - Hold: |edge| <= 0.05
    - Reduce: edge < -0.05, risk < 0.4
    - Sell: edge < -0.15, risk < 0.3
    - Strong Sell: edge < -0.3, risk < 0.2

    Args:
        edge: Edge value (P_long - P_short)
        risk: Risk metric

    Returns:
        Recommendation string
    """
    if edge > 0.3 and risk < 0.2:
        return "Strong Buy"
    elif edge > 0.15 and risk < 0.3:
        return "Buy"
    elif edge > 0.05 and risk < 0.4:
        return "Accumulate"
    elif abs(edge) <= 0.05:
        return "Hold"
    elif edge < -0.05 and risk < 0.4:
        return "Reduce"
    elif edge < -0.15 and risk < 0.3:
        return "Sell"
    elif edge < -0.3 and risk < 0.2:
        return "Strong Sell"
    else:
        # Default based on edge
        if edge > 0.1:
            return "Buy"
        elif edge < -0.1:
            return "Sell"
        else:
            return "Hold"


def segment_windows(
    predictions: list[dict[str, Any]],
    edge_threshold: float = 0.1,
    risk_threshold: float = 0.3,
    min_window_days: int = 7,
) -> list[dict[str, Any]]:
    """
    Segment predictions into action windows based on similar edge and risk.

    Args:
        predictions: List of prediction dictionaries
        edge_threshold: Threshold for edge similarity
        risk_threshold: Threshold for risk similarity
        min_window_days: Minimum window size in days

    Returns:
        List of window dictionaries
    """
    if not predictions:
        return []

    windows: list[dict[str, Any]] = []
    current_window_start = None
    current_window_end = None
    current_edge = None
    current_risk = None

    for pred in predictions:
        edge = calculate_edge(pred["P_long"], pred["P_short"])
        risk = calculate_risk_metric(
            pred["P_long"],
            pred["P_short"],
            pred.get("depth_long_bins", {}),
            pred.get("depth_short_bins", {}),
        )

        if current_window_start is None:
            # Start new window
            current_window_start = pred["date"]
            current_window_end = pred["date"]
            current_edge = edge
            current_risk = risk
        else:
            # Check if similar enough to extend window
            edge_diff = abs(edge - current_edge) if current_edge is not None else float("inf")
            risk_diff = abs(risk - current_risk) if current_risk is not None else float("inf")

            if edge_diff <= edge_threshold and risk_diff <= risk_threshold:
                # Extend window
                current_window_end = pred["date"]
                # Update running averages
                if current_edge is not None:
                    current_edge = (current_edge + edge) / 2.0
                if current_risk is not None:
                    current_risk = (current_risk + risk) / 2.0
            else:
                # Close current window and start new one
                if current_window_start and current_window_end:
                    start_date = parse_date(current_window_start)
                    end_date = parse_date(current_window_end)
                    window_days = (end_date - start_date).days + 1

                    if window_days >= min_window_days:
                        recommendation = map_to_recommendation(
                            current_edge or 0.0, current_risk or 0.0
                        )
                        windows.append(
                            {
                                "start_date": current_window_start,
                                "end_date": current_window_end,
                                "weeks": window_days / 7.0,
                                "edge": current_edge,
                                "risk": current_risk,
                                "recommendation": recommendation,
                            }
                        )

                # Start new window
                current_window_start = pred["date"]
                current_window_end = pred["date"]
                current_edge = edge
                current_risk = risk

    # Close last window
    if current_window_start and current_window_end:
        start_date = parse_date(current_window_start)
        end_date = parse_date(current_window_end)
        window_days = (end_date - start_date).days + 1

        if window_days >= min_window_days:
            recommendation = map_to_recommendation(current_edge or 0.0, current_risk or 0.0)
            windows.append(
                {
                    "start_date": current_window_start,
                    "end_date": current_window_end,
                    "weeks": window_days / 7.0,
                    "edge": current_edge,
                    "risk": current_risk,
                    "recommendation": recommendation,
                }
            )

    return windows


def plan_windows(
    symbol: str,
    short_weeks: tuple[int, int] = (1, 26),
    medium_weeks: tuple[int, int] = (13, 52),
    long_weeks: tuple[int, int] = (26, 104),
    start: str | None = None,
    end: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Plan action windows based on baseline predictions.

    Args:
        symbol: Trading symbol
        short_weeks: Short horizon weeks range (min, max)
        medium_weeks: Medium horizon weeks range (min, max)
        long_weeks: Long horizon weeks range (min, max)
        settings: Settings instance (defaults to global)

    Returns:
        Dictionary with window recommendations
    """
    if settings is None:
        settings = get_settings()

    predictions_dir = settings.reports_predictions_dir / symbol / "daily"
    if not predictions_dir.exists():
        return {"error": f"No predictions found for {symbol}. Run `prosper predict baseline` first."}

    start_dt = parse_date(start) if start else None
    end_dt = parse_date(end) if end else None

    predictions: list[dict[str, Any]] = []
    for jsonl_file in sorted(predictions_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pred = json.loads(line)
                pred_date = parse_date(pred["date"])
                if start_dt is not None and pred_date < start_dt:
                    continue
                if end_dt is not None and pred_date > end_dt:
                    continue
                predictions.append(pred)

    if not predictions:
        return {"error": "No predictions loaded for given date range"}

    predictions.sort(key=lambda x: x["date"])

    horizons = ["short", "medium", "long"]
    # (horizon, iso_year, iso_week) -> accumulators
    buckets: dict[tuple[str, int, int], dict[str, Any]] = {}

    for pred in predictions:
        dt = parse_date(pred["date"])
        iso_year, iso_week = dt.isocalendar()[0], dt.isocalendar()[1]
        week_start = dt - timedelta(days=dt.weekday())
        week_end = week_start + timedelta(days=6)
        week_start_str = week_start.date().isoformat()
        week_end_str = week_end.date().isoformat()

        for h in horizons:
            p_long = float(pred[h]["P_long"])
            p_short = float(pred[h]["P_short"])
            edge = calculate_edge(p_long=p_long, p_short=p_short)
            risk = calculate_risk_metric(
                p_long=p_long,
                p_short=p_short,
                depth_long_bins=pred[h].get("depth_long_bins", {}),
                depth_short_bins=pred[h].get("depth_short_bins", {}),
            )

            key = (h, iso_year, iso_week)
            b = buckets.get(key)
            if b is None:
                b = {
                    "horizon": h,
                    "iso_year": iso_year,
                    "iso_week": iso_week,
                    "week_start": week_start_str,
                    "week_end": week_end_str,
                    "edge_sum": 0.0,
                    "risk_sum": 0.0,
                    "count": 0,
                }
                buckets[key] = b
            b["edge_sum"] += edge
            b["risk_sum"] += risk
            b["count"] += 1

    # Build window objects and group them by YYYY-MM
    month_to_windows: dict[str, list[dict[str, Any]]] = {}
    all_windows: list[dict[str, Any]] = []
    for b in buckets.values():
        edge_mean = b["edge_sum"] / max(b["count"], 1)
        risk_mean = b["risk_sum"] / max(b["count"], 1)
        recommendation = map_to_recommendation(edge_mean, risk_mean)

        w = {
            "start_date": b["week_start"],
            "end_date": b["week_end"],
            "horizon": b["horizon"],
            "recommendation": recommendation,
            "diagnostics": {"edge_mean": edge_mean, "risk_mean": risk_mean},
        }
        all_windows.append(w)

        month_key = b["week_start"][:7]
        month_to_windows.setdefault(month_key, []).append(w)

    # Write one report per month
    for month_key, windows in month_to_windows.items():
        year = int(month_key[:4])
        month = int(month_key[5:7])
        report_path = get_recommendation_report_path(symbol, year, month, settings=settings)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "symbol": symbol,
            "month": month_key,
            "windows": windows,
            "summary": {"total_windows": len(windows)},
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)

    return {"symbol": symbol, "total_windows": len(all_windows), "windows": all_windows}
