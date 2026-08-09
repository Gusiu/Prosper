"""The risk gate is relative to what this market has already looked like.

`calculate_risk_metric` counts only the adverse side, so it is bounded by 0.5:
the losing direction's probability cannot exceed a half. The cut-offs
0.2/0.3/0.4 were chosen when the metric summed both sides and could reach 1.0,
and nobody re-derived them. Measured on the recomputed runs the metric peaks at
0.365, so the outer gate blocked 0 of 2129 bars — an advertised three-stage
gate that was really two.

Reading the observed maximum and lowering the constant to suit would be tuning
on the evaluation set. Quantiles of the risk already seen say what the gate is
for, keep it reachable, and survive the next change to the bin scheme.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from prosper.config import Settings
from prosper.domain import HORIZON_NAMES
from prosper.planner.windows import (
    RISK_WARMUP_WINDOWS,
    STATIC_RISK_GATES,
    calculate_edge,
    calculate_risk_metric,
    map_to_recommendation,
    plan_windows,
    risk_gates,
)
from prosper.storage.layout import get_recommendation_report_path

SYMBOL = "TESTUSDT"

BIG = {"34-55": 0.5, "55-89": 0.3, "89-144": 0.2}
SMALL = {"1-2": 0.6, "2-3": 0.4}


def _run_with_rising_risk(tmp_path, days: int = 700) -> Settings:
    """A run whose adverse tail grows steadily, so the gates have to move."""
    settings = Settings(data_root=tmp_path)
    run_dir = settings.reports_predictions_dir / SYMBOL / "ml_1d_20240101000000"
    run_dir.mkdir(parents=True, exist_ok=True)

    start = datetime(2020, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(days):
        # Confident long throughout; only the short side's tail thickens.
        heavy = 0.05 + 0.9 * i / days
        horizon = {
            "P_long": 0.7,
            "P_short": 0.3,
            "depth_long_bins": {"5-8": 1.0},
            "depth_short_bins": {"5-8": 1.0 - heavy, "55-89": heavy},
            "trained": True,
        }
        moment = start + timedelta(days=i)
        rows.append(
            {
                "open_time": moment.isoformat(),
                "date": moment.date().isoformat(),
                "symbol": SYMBOL,
                **{name: dict(horizon) for name in HORIZON_NAMES},
            }
        )
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return settings


def test_the_metric_cannot_reach_the_old_outer_gate() -> None:
    """Why the constants had to go: 0.4 is unreachable by construction.

    Risk is `min(p_long, p_short) * P(large move | that side)`, and the smaller
    of two probabilities summing to 1 is at most 0.5. Even a certain large
    adverse move at a perfect coin flip lands on 0.5, and every real bar is
    below it.
    """
    worst = calculate_risk_metric(0.5, 0.5, depth_long_bins=BIG, depth_short_bins=BIG)

    assert worst == pytest.approx(0.5)
    assert all(
        calculate_risk_metric(p, 1.0 - p, depth_long_bins=BIG, depth_short_bins=BIG) <= 0.5
        for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    )


def test_a_short_history_keeps_the_historical_constants() -> None:
    assert risk_gates([]) == STATIC_RISK_GATES
    assert risk_gates([0.1] * (RISK_WARMUP_WINDOWS - 1)) == STATIC_RISK_GATES


def test_the_gates_become_quantiles_once_there_is_history() -> None:
    history = [i / 100.0 for i in range(RISK_WARMUP_WINDOWS)]
    strong, moderate, weak = risk_gates(history)

    assert strong < moderate < weak
    assert strong == pytest.approx(0.40 * (len(history) - 1) / 100.0)
    assert weak == pytest.approx(0.90 * (len(history) - 1) / 100.0)


def test_the_gates_stay_ordered_for_any_history() -> None:
    for history in ([0.2] * 60, [0.0, 0.5] * 40, [i / 60 for i in range(60)]):
        strong, moderate, weak = risk_gates(history)
        assert strong <= moderate <= weak


def test_a_calm_market_still_gets_a_binding_gate() -> None:
    """The point of the change: the strongest grade stays earnable.

    Every one of these bars would have cleared the old 0.2 constant, so the
    gate expressed no preference between them. Against its own distribution the
    quieter half is still separated from the noisier half.
    """
    history = [0.05 + i * 0.001 for i in range(RISK_WARMUP_WINDOWS)]
    strong, _, _ = risk_gates(history)

    assert strong < STATIC_RISK_GATES[0]
    assert min(history) < strong < max(history)


def test_a_window_does_not_influence_its_own_gate(tmp_path) -> None:
    """The gates are part of the decision, so they may only see the past.

    Exercised through `plan_windows`, because the ordering is what carries the
    property: a unit test of `risk_gates` alone could only restate that a pure
    function ignores arguments it was not given.
    """
    settings = _run_with_rising_risk(tmp_path)
    result = plan_windows(SYMBOL, settings=settings, model_type="ml", interval="1d")

    short = sorted(
        (w for w in result["windows"] if w["horizon"] == HORIZON_NAMES[0]),
        key=lambda w: w["start_date"],
    )
    assert len(short) > RISK_WARMUP_WINDOWS, "not enough windows to leave warm-up"

    for position, window in enumerate(short):
        diagnostics = window["diagnostics"]
        assert diagnostics["risk_history_windows"] == position, "the walk is chronological"

        expected = risk_gates([w["diagnostics"]["risk_mean"] for w in short[:position]])
        assert tuple(diagnostics["risk_gates"]) == pytest.approx(expected)

    # Risk climbs over the run, so a late window's own value is above every
    # gate it was judged by — proof the gate never absorbed it.
    assert short[-1]["diagnostics"]["risk_mean"] > max(short[-1]["diagnostics"]["risk_gates"])


def test_the_gates_are_recorded_in_the_report(tmp_path) -> None:
    """A report has to be readable without replaying the walk that graded it."""
    settings = _run_with_rising_risk(tmp_path)
    plan_windows(SYMBOL, settings=settings, model_type="ml", interval="1d")

    path = get_recommendation_report_path(
        SYMBOL, 2021, 6, settings=settings, run_slug="ml_1d_20240101000000"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    for window in payload["windows"]:
        gates = window["diagnostics"]["risk_gates"]
        assert len(gates) == 3
        assert gates[0] <= gates[1] <= gates[2]


def test_custom_gates_keep_the_scale_symmetric() -> None:
    """Invariant 13 must survive the new parameter."""
    gates = (0.15, 0.25, 0.33)
    bull_risk = calculate_risk_metric(0.9, 0.05, BIG, SMALL)
    bear_risk = calculate_risk_metric(0.05, 0.9, SMALL, BIG)

    assert bull_risk == pytest.approx(bear_risk)
    assert map_to_recommendation(calculate_edge(0.9, 0.05), bull_risk, gates=gates) == "Strong Buy"
    assert map_to_recommendation(calculate_edge(0.05, 0.9), bear_risk, gates=gates) == "Strong Sell"


def test_tighter_gates_can_only_downgrade() -> None:
    """A stricter gate never promotes a bar; it demotes or leaves it alone."""
    order = [
        "Strong Sell", "Sell", "Reduce", "Hold", "Accumulate", "Buy", "Strong Buy",
    ]
    loose = (0.40, 0.50, 0.60)
    tight = (0.10, 0.20, 0.30)

    for edge in (-0.9, -0.4, -0.2, -0.08, 0.08, 0.2, 0.4, 0.9):
        for risk in (0.0, 0.05, 0.15, 0.25, 0.35, 0.45):
            wide = map_to_recommendation(edge, risk, gates=loose)
            narrow = map_to_recommendation(edge, risk, gates=tight)
            if edge > 0:
                assert order.index(narrow) <= order.index(wide)
            else:
                assert order.index(narrow) >= order.index(wide)


def test_the_default_gates_are_the_historical_constants() -> None:
    """A caller that passes nothing must behave exactly as before."""
    assert map_to_recommendation(0.5, 0.25) == map_to_recommendation(
        0.5, 0.25, gates=STATIC_RISK_GATES
    )
    assert STATIC_RISK_GATES == (0.20, 0.30, 0.40)
