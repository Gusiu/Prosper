"""Risk is the chance of a move against the position, not of any large move.

`risk = p_long * P(large|long) + p_short * P(large|short)` counted an expected
rally as a reason not to buy. On strongly bullish bars of the recomputed ml run
mean risk was 0.85, of which 0.83 was the upside term, so the planner vetoed
the signals it was most confident about and held ~90% of medium and long bars.
"""

from __future__ import annotations

import pytest
from prosper.planner.windows import calculate_edge, calculate_risk_metric, map_to_recommendation

# A confident forecast of a large move, in one direction only.
BIG = {"34-55": 0.5, "55-89": 0.3, "89-144": 0.2}
SMALL = {"1-2": 0.6, "2-3": 0.4}


def test_an_expected_rally_is_not_a_reason_to_avoid_buying() -> None:
    risk = calculate_risk_metric(0.9, 0.05, depth_long_bins=BIG, depth_short_bins=SMALL)

    # The whole long tail is ignored; only the (tiny, small-move) short side counts.
    assert risk == pytest.approx(0.0)
    assert map_to_recommendation(calculate_edge(0.9, 0.05), risk) == "Strong Buy"


def test_a_fat_adverse_tail_still_blocks_the_signal() -> None:
    """The gate must keep working — this is the case it exists for."""
    risk = calculate_risk_metric(0.55, 0.45, depth_long_bins=SMALL, depth_short_bins=BIG)

    assert risk == pytest.approx(0.45)
    assert map_to_recommendation(calculate_edge(0.55, 0.45), risk) == "Hold"


def test_risk_mirrors_when_the_bar_does() -> None:
    """Invariant 8: a mirrored bar must produce the mirrored recommendation."""
    bull_risk = calculate_risk_metric(0.9, 0.05, BIG, SMALL)
    bear_risk = calculate_risk_metric(0.05, 0.9, SMALL, BIG)

    assert bull_risk == pytest.approx(bear_risk)
    assert map_to_recommendation(calculate_edge(0.9, 0.05), bull_risk) == "Strong Buy"
    assert map_to_recommendation(calculate_edge(0.05, 0.9), bear_risk) == "Strong Sell"


@pytest.mark.parametrize("p_long", [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
def test_the_metric_is_symmetric_across_the_whole_range(p_long: float) -> None:
    p_short = 1.0 - p_long
    forward = calculate_risk_metric(p_long, p_short, BIG, SMALL)
    mirrored = calculate_risk_metric(p_short, p_long, SMALL, BIG)

    assert forward == pytest.approx(mirrored)


def test_a_bearish_signal_is_not_blocked_by_its_own_downside() -> None:
    """The naive downside-only fix would have made `Strong Sell` unreachable."""
    risk = calculate_risk_metric(0.05, 0.9, depth_long_bins=SMALL, depth_short_bins=BIG)

    assert risk == pytest.approx(0.0)
    assert map_to_recommendation(calculate_edge(0.05, 0.9), risk) == "Strong Sell"


def test_every_verdict_on_the_scale_is_still_reachable() -> None:
    seen = set()
    for p_long in [i / 20 for i in range(21)]:
        for adverse_tail in [i / 10 for i in range(11)]:
            p_short = 1.0 - p_long
            depth_long = {"34-55": adverse_tail, "1-2": 1.0 - adverse_tail}
            depth_short = {"34-55": adverse_tail, "1-2": 1.0 - adverse_tail}
            risk = calculate_risk_metric(p_long, p_short, depth_long, depth_short)
            seen.add(map_to_recommendation(calculate_edge(p_long, p_short), risk))

    assert seen == {
        "Strong Buy",
        "Buy",
        "Accumulate",
        "Hold",
        "Reduce",
        "Sell",
        "Strong Sell",
    }
