"""Tests for planner window utilities."""

from prosper.planner.windows import calculate_edge, calculate_risk_metric, map_to_recommendation


def test_calculate_edge() -> None:
    assert abs(calculate_edge(0.7, 0.2) - 0.5) < 1e-12
    assert abs(calculate_edge(0.1, 0.4) + 0.3) < 1e-12


def test_calculate_risk_metric_large_bins_only() -> None:
    p_long = 0.6
    p_short = 0.3
    depth_long = {
        "1-2": 0.2,
        "8-13": 0.2,
        "13-21": 0.3,
        "21-34": 0.2,
        "34+": 0.1,
    }
    depth_short = {
        "1-2": 0.1,
        "5-8": 0.2,
        "13-21": 0.4,
        "21-34": 0.2,
        "34+": 0.1,
    }
    expected = p_long * (0.3 + 0.2 + 0.1) + p_short * (0.4 + 0.2 + 0.1)
    assert abs(calculate_risk_metric(p_long, p_short, depth_long, depth_short) - expected) < 1e-12


ALLOWED = {"Strong Buy", "Buy", "Accumulate", "Hold", "Reduce", "Sell", "Strong Sell"}


def test_map_to_recommendation_valid_domain() -> None:
    for edge in (-0.8, -0.2, 0.0, 0.2, 0.8):
        for risk in (0.05, 0.2, 0.4, 0.8):
            rec = map_to_recommendation(edge, risk)
            assert rec in ALLOWED


def test_every_step_of_the_scale_is_reachable() -> None:
    """`Strong Sell` used to be unreachable for every (edge, risk) pair.

    The sell branches were ordered weakest-first, so `Reduce` shadowed both
    `Sell` and `Strong Sell`: any risk low enough for a strong signal also
    satisfied the weaker branch, which was tested first.
    """
    reachable = {
        map_to_recommendation(edge / 100, risk / 100)
        for edge in range(-100, 101, 1)
        for risk in range(0, 101, 1)
    }
    assert reachable == ALLOWED


def test_the_scale_is_symmetric_around_zero_edge() -> None:
    mirror = {
        "Strong Buy": "Strong Sell",
        "Buy": "Sell",
        "Accumulate": "Reduce",
        "Hold": "Hold",
    }
    for edge in range(1, 101):
        for risk in range(0, 101):
            bullish = map_to_recommendation(edge / 100, risk / 100)
            bearish = map_to_recommendation(-edge / 100, risk / 100)
            assert mirror[bullish] == bearish, f"edge={edge / 100} risk={risk / 100}"


def test_conviction_the_risk_does_not_support_falls_back_to_hold() -> None:
    # A huge edge with a fat downside distribution must not become a Buy.
    assert map_to_recommendation(0.9, 0.95) == "Hold"
    assert map_to_recommendation(-0.9, 0.95) == "Hold"

