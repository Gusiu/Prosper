"""A signal has to pay for its own round trip before the planner acts on it.

`edge` is `P_long - P_short`: it says which way, never how far. A bar whose
long mass sits entirely in the 1-2% bin produced exactly the same edge as one
whose mass sits in 144+, so conviction alone could order a trade whose expected
move was smaller than the fee. The depth head already knew the difference; this
is the decision layer finally reading it.

The cost is a property of the trade, not of the market, so it is deliberately
not scaled by horizon: one round trip is paid whether a position is held for a
month or a year.
"""

from __future__ import annotations

import pytest
from prosper.config import Settings
from prosper.planner.windows import (
    clears_cost,
    expected_move,
    map_to_recommendation,
)

CHEAP = {"1-2": 1.0}  # midpoint 1.5%
RICH = {"34-55": 1.0}  # midpoint 44.5%
COST = 0.004  # 0.4% round trip, the Binance SPOT default


def test_a_confident_but_tiny_move_does_not_clear_costs() -> None:
    move = expected_move(p_long=0.9, p_short=0.05, depth_long_bins=CHEAP, depth_short_bins=CHEAP)

    # 0.9 * 1.5% - 0.05 * 1.5% = 1.275%, which does clear 0.4%...
    assert move == pytest.approx(0.01275)
    assert clears_cost(move, COST)

    # ...but a venue with a 2% round trip must not act on it.
    assert not clears_cost(move, 0.02)


def test_a_large_expected_move_clears_costs() -> None:
    move = expected_move(p_long=0.9, p_short=0.05, depth_long_bins=RICH, depth_short_bins=RICH)

    assert move == pytest.approx(0.37825)
    assert clears_cost(move, COST)


def test_depth_decides_where_edge_cannot() -> None:
    """Same direction probabilities, opposite verdicts — the point of the gate."""
    tiny = expected_move(0.9, 0.05, CHEAP, CHEAP)
    large = expected_move(0.9, 0.05, RICH, RICH)

    assert not clears_cost(tiny, 0.02)
    assert clears_cost(large, 0.02)


def test_the_gate_is_symmetric() -> None:
    """Exiting costs what entering costs; invariant 8 must survive the gate."""
    bullish = expected_move(0.9, 0.05, RICH, RICH)
    bearish = expected_move(0.05, 0.9, RICH, RICH)

    assert bullish == pytest.approx(-bearish)
    assert clears_cost(bullish, COST) == clears_cost(bearish, COST)


def test_an_uncleared_signal_becomes_hold_however_strong() -> None:
    assert map_to_recommendation(0.9, 0.05, cost_cleared=True) == "Strong Buy"
    assert map_to_recommendation(0.9, 0.05, cost_cleared=False) == "Hold"
    assert map_to_recommendation(-0.9, 0.05, cost_cleared=True) == "Strong Sell"
    assert map_to_recommendation(-0.9, 0.05, cost_cleared=False) == "Hold"


def test_the_gate_defaults_open() -> None:
    """Callers with no depth distribution keep the old behaviour."""
    assert map_to_recommendation(0.9, 0.05) == "Strong Buy"


def test_an_empty_distribution_yields_no_move() -> None:
    assert expected_move(0.5, 0.5, {}, {}) == 0.0
    assert not clears_cost(0.0, COST)


def test_the_open_top_bin_still_contributes() -> None:
    """`144+` has no midpoint; it must not silently count as zero."""
    move = expected_move(1.0, 0.0, {"144+": 1.0}, {})
    assert move == pytest.approx(1.80)  # 144 * 1.25, as a fraction


def test_a_legacy_run_is_measured_on_its_own_bins() -> None:
    """`34+` is absent from the current scheme but must still carry weight."""
    move = expected_move(1.0, 0.0, {"34+": 1.0}, {})
    assert move == pytest.approx(0.425)  # 34 * 1.25 / 100


def test_the_cost_is_configurable_per_market() -> None:
    settings = Settings(data_root=".", planner_round_trip_cost=0.02)
    assert settings.planner_round_trip_cost == 0.02

    move = expected_move(0.9, 0.05, CHEAP, CHEAP)
    assert clears_cost(move, Settings(data_root=".").planner_round_trip_cost)
    assert not clears_cost(move, settings.planner_round_trip_cost)
