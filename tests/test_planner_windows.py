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


def test_map_to_recommendation_valid_domain() -> None:
    allowed = {"Strong Buy", "Buy", "Accumulate", "Hold", "Reduce", "Sell", "Strong Sell"}
    for edge in (-0.8, -0.2, 0.0, 0.2, 0.8):
        for risk in (0.05, 0.2, 0.4, 0.8):
            rec = map_to_recommendation(edge, risk)
            assert rec in allowed

