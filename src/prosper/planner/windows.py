"""Action window planner based on baseline predictions."""

import json
from datetime import timedelta
from typing import Any

from prosper.config import Settings, get_settings
from prosper.domain import depth_bin_midpoints, depth_scheme_of, is_large_move_bin
from prosper.storage.layout import get_recommendation_report_path
from prosper.storage.runs import load_run_predictions, resolve_run
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
    Probability of a large move against the position the forecast implies.

    Args:
        p_long: Probability of long
        p_short: Probability of short
        depth_long_bins: Depth bin probabilities for long
        depth_short_bins: Depth bin probabilities for short

    Returns:
        Risk metric (higher = more risk)
    """
    # Which labels count as "large" is derived from the bin edges, not from a
    # fixed set of names: naming them pinned the metric to one bin scheme, so
    # extending the tail (34+ -> 34-55, 55-89, 89-144, 144+) would have
    # silently dropped the whole tail for every run written under the old one.
    long_large = sum(prob for label, prob in depth_long_bins.items() if is_large_move_bin(label))
    short_large = sum(prob for label, prob in depth_short_bins.items() if is_large_move_bin(label))

    # Risk is the chance of a large move *against* the position the edge
    # implies — not of a large move in general. Summing both sides counted an
    # expected rally as a reason not to buy: on strongly bullish bars of the ml
    # run, mean risk was 0.85 and 98% of it was the upside term, so the planner
    # vetoed the very signals it was most confident about.
    #
    # Reading the side from `p_long >= p_short` rather than from a separate
    # argument is what keeps `map_to_recommendation` symmetric: mirroring a bar
    # swaps both the probabilities and the distributions, so the risk of the
    # mirrored bar is unchanged while the edge flips sign.
    if p_long > p_short:
        return p_short * short_large
    if p_short > p_long:
        return p_long * long_large
    # No directional lean, so neither side is "against" the position. Take the
    # worse tail: anything else makes the metric asymmetric at exactly the tie.
    return max(p_short * short_large, p_long * long_large)


def expected_move(
    p_long: float,
    p_short: float,
    depth_long_bins: dict[str, float],
    depth_short_bins: dict[str, float],
) -> float:
    """Probability-weighted forward move, as a fraction (0.05 == 5%).

    The direction head says which way; the depth head says how far. `edge`
    alone answers only the first question, so a bar whose long mass all sits in
    the 1-2% bin scores exactly like one whose mass sits in 144+. This combines
    both, which is what a cost comparison needs.
    """
    long_mid = depth_bin_midpoints(depth_scheme_of(depth_long_bins))
    short_mid = depth_bin_midpoints(depth_scheme_of(depth_short_bins))

    up = sum(prob * long_mid.get(label, 0.0) for label, prob in depth_long_bins.items())
    down = sum(prob * short_mid.get(label, 0.0) for label, prob in depth_short_bins.items())

    return (p_long * up - p_short * down) / 100.0


def clears_cost(expected: float, round_trip_cost: float) -> bool:
    """Whether an expected move is worth acting on once costs are paid.

    Costs are a property of the trade, not of the market: one round trip is
    paid whether the position is held for a month or a year, so this threshold
    is deliberately *not* scaled by horizon. It is symmetric because entering
    and exiting both cost the same — see `map_to_recommendation`.
    """
    return abs(expected) > round_trip_cost


def map_to_recommendation(edge: float, risk: float, *, cost_cleared: bool = True) -> str:
    """
    Map edge and risk to recommendation.

    *cost_cleared* is the verdict of `clears_cost` on the expected move: a
    signal whose expected magnitude cannot pay for its own round trip is
    `Hold`, however confident the direction head is. It defaults to True so
    that callers with no depth distribution to work from keep the old
    behaviour rather than silently holding everything.

    Both sides are tested strongest-first and are mirror images of each other:

    - Strong Buy / Strong Sell: |edge| > 0.3, risk < 0.2
    - Buy / Sell:               |edge| > 0.15, risk < 0.3
    - Accumulate / Reduce:      |edge| > 0.05, risk < 0.4
    - Hold:                     |edge| <= 0.05, or conviction the risk does not support

    The ordering matters. An earlier version listed the sell branches
    weakest-first, so `Reduce` (edge < -0.05, risk < 0.4) shadowed both `Sell`
    and `Strong Sell`: any risk low enough to qualify as a strong signal also
    satisfied the weaker branch, which was checked first. `Strong Sell` was
    unreachable for every (edge, risk) pair and the advertised seven-point scale
    was really a five-point one.

    Args:
        edge: Edge value (P_long - P_short)
        risk: Risk metric

    Returns:
        Recommendation string
    """
    if abs(edge) <= 0.05:
        return "Hold"
    if not cost_cleared:
        return "Hold"

    bullish = edge > 0
    if abs(edge) > 0.3 and risk < 0.2:
        return "Strong Buy" if bullish else "Strong Sell"
    if abs(edge) > 0.15 and risk < 0.3:
        return "Buy" if bullish else "Sell"
    if abs(edge) > 0.05 and risk < 0.4:
        return "Accumulate" if bullish else "Reduce"

    # Conviction the risk metric does not support: stay flat rather than act on
    # a signal whose downside distribution is fat.
    return "Hold"


def plan_windows(
    symbol: str,
    short_weeks: tuple[int, int] = (1, 26),
    medium_weeks: tuple[int, int] = (13, 52),
    long_weeks: tuple[int, int] = (26, 104),
    start: str | None = None,
    end: str | None = None,
    settings: Settings | None = None,
    model_type: str | None = None,
    timestamp: str | None = None,
    interval: str | None = None,
) -> dict[str, Any]:
    """
    Plan action windows from one versioned prediction run.

    Args:
        symbol: Trading symbol
        short_weeks: Short horizon weeks range (min, max)
        medium_weeks: Medium horizon weeks range (min, max)
        long_weeks: Long horizon weeks range (min, max)
        start: Optional lower date bound (YYYY-MM-DD)
        end: Optional upper date bound (YYYY-MM-DD)
        settings: Settings instance (defaults to global)
        model_type: Model that produced the predictions; latest run if omitted
        timestamp: Run timestamp; latest matching run if omitted
        interval: Run interval; latest matching run if omitted

    Returns:
        Dictionary with window recommendations, tagged with the source run
    """
    if settings is None:
        settings = get_settings()

    try:
        run = resolve_run(
            symbol,
            model_type=model_type,
            timestamp=timestamp,
            interval=interval,
            settings=settings,
        )
    except FileNotFoundError as e:
        return {"error": str(e)}

    round_trip_cost = settings.planner_round_trip_cost
    start_dt = parse_date(start) if start else None
    end_dt = parse_date(end) if end else None

    predictions: list[dict[str, Any]] = []
    for pred in load_run_predictions(run):
        pred_date = parse_date(str(pred["date"])[:10])
        if start_dt is not None and pred_date < start_dt:
            continue
        if end_dt is not None and pred_date > end_dt:
            continue
        predictions.append(pred)

    if not predictions:
        return {"error": f"No predictions in run {run.slug} for the given date range"}

    predictions.sort(key=lambda x: str(x.get("open_time") or x["date"]))

    horizons = ["short", "medium", "long"]
    # (horizon, iso_year, iso_week) -> accumulators
    buckets: dict[tuple[str, int, int], dict[str, Any]] = {}

    skipped_untrained = 0
    for pred in predictions:
        dt = parse_date(str(pred["date"])[:10])
        iso_year, iso_week = dt.isocalendar()[0], dt.isocalendar()[1]
        week_start = dt - timedelta(days=dt.weekday())
        week_end = week_start + timedelta(days=6)
        week_start_str = week_start.date().isoformat()
        week_end_str = week_end.date().isoformat()

        for h in horizons:
            # An untrained horizon carries the uniform prior. Averaging it into
            # a window would dilute real signal toward "Hold" for free.
            if pred[h].get("trained") is False:
                skipped_untrained += 1
                continue

            p_long = float(pred[h]["P_long"])
            p_short = float(pred[h]["P_short"])
            depth_long = pred[h].get("depth_long_bins", {})
            depth_short = pred[h].get("depth_short_bins", {})
            edge = calculate_edge(p_long=p_long, p_short=p_short)
            risk = calculate_risk_metric(
                p_long=p_long,
                p_short=p_short,
                depth_long_bins=depth_long,
                depth_short_bins=depth_short,
            )
            move = expected_move(
                p_long=p_long,
                p_short=p_short,
                depth_long_bins=depth_long,
                depth_short_bins=depth_short,
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
                    "move_sum": 0.0,
                    "count": 0,
                }
                buckets[key] = b
            b["edge_sum"] += edge
            b["risk_sum"] += risk
            b["move_sum"] += move
            b["count"] += 1

    # Build window objects and group them by YYYY-MM
    month_to_windows: dict[str, list[dict[str, Any]]] = {}
    all_windows: list[dict[str, Any]] = []
    for b in buckets.values():
        edge_mean = b["edge_sum"] / max(b["count"], 1)
        risk_mean = b["risk_sum"] / max(b["count"], 1)
        move_mean = b["move_sum"] / max(b["count"], 1)
        cost_cleared = clears_cost(move_mean, round_trip_cost)
        recommendation = map_to_recommendation(edge_mean, risk_mean, cost_cleared=cost_cleared)

        w = {
            "start_date": b["week_start"],
            "end_date": b["week_end"],
            "horizon": b["horizon"],
            "recommendation": recommendation,
            "diagnostics": {
                "edge_mean": edge_mean,
                "risk_mean": risk_mean,
                "expected_move": move_mean,
                "round_trip_cost": round_trip_cost,
                "cost_cleared": cost_cleared,
            },
        }
        all_windows.append(w)

        month_key = b["week_start"][:7]
        month_to_windows.setdefault(month_key, []).append(w)

    # Write one report per month, scoped to the run that produced it
    for month_key, windows in month_to_windows.items():
        year = int(month_key[:4])
        month = int(month_key[5:7])
        report_path = get_recommendation_report_path(
            symbol, year, month, settings=settings, run_slug=run.slug
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "symbol": symbol,
            "model_type": run.model_type,
            "interval": run.interval,
            "timestamp": run.timestamp,
            "month": month_key,
            "windows": windows,
            "summary": {"total_windows": len(windows)},
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)

    return {
        "symbol": symbol,
        "run": run.to_dict(),
        "total_windows": len(all_windows),
        "skipped_untrained_horizons": skipped_untrained,
        "windows": all_windows,
    }
