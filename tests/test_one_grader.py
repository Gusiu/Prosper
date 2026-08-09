"""One object decides a verdict, and nothing else may.

`map_to_recommendation` was called from three places. The quantile risk gates
and the cost gate were added to `plan_windows` only, so the same bar could be
`Accumulate` in the simulation, `Hold` in the recommendation report, and a third
thing in the evaluation artifacts — three answers to one question, none of them
flagged. Each was found separately, hours apart, by tripping over a symptom.

This file exists so a fourth path cannot appear quietly. The rule: production
grading goes through `SequenceGrader`, which is the only thing allowed to decide
what the gates are.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from prosper.planner.windows import RECOMMENDATIONS, STATIC_RISK_GATES, SequenceGrader

SRC = Path(__file__).resolve().parents[1] / "src" / "prosper"

# `windows.py` defines both, and the evaluation keeps a documented fallback for
# callers with no chronological context to offer.
ALLOWED_DIRECT_CALLERS = {
    Path("planner/windows.py"),
    Path("eval/predictions.py"),
}


def _direct_call_sites(path: Path) -> list[int]:
    # utf-8-sig: two source files carry a byte-order mark. Python's importer
    # tolerates one, `ast.parse` on decoded text does not.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "map_to_recommendation"
    ]


def test_nothing_new_grades_a_bar_on_its_own() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC)
        if relative in ALLOWED_DIRECT_CALLERS:
            continue
        for line in _direct_call_sites(path):
            offenders.append(f"{relative}:{line}")

    assert not offenders, (
        "these call map_to_recommendation directly and so bypass the risk and "
        "cost gates; go through SequenceGrader:\n  " + "\n  ".join(offenders)
    )


def test_the_grader_starts_on_the_historical_constants() -> None:
    """Warm-up behaviour has to be the documented one, or a short run silently
    grades against quantiles of almost nothing."""
    grader = SequenceGrader(round_trip_cost=0.004)
    assert grader.gates == STATIC_RISK_GATES
    assert grader.observed == 0


def test_a_bar_is_graded_before_its_risk_joins_the_history() -> None:
    grader = SequenceGrader(round_trip_cost=0.0)
    gates_before = grader.gates

    grader.grade(edge=0.5, risk=0.49, expected=0.5)

    assert grader.observed == 1
    assert gates_before == STATIC_RISK_GATES, "the gates used knew nothing of that bar"


def test_the_cost_gate_holds_a_move_too_small_to_pay_for_itself() -> None:
    grader = SequenceGrader(round_trip_cost=0.004)
    # Strong direction, tiny magnitude: not worth the round trip.
    assert grader.grade(edge=0.9, risk=0.0, expected=0.001) == "Hold"


def test_no_expected_move_leaves_the_cost_gate_open() -> None:
    """A caller with no depth distribution must not have everything held."""
    grader = SequenceGrader(round_trip_cost=0.004)
    assert grader.grade(edge=0.9, risk=0.0, expected=None) == "Strong Buy"


def test_the_grader_is_symmetric() -> None:
    """Invariant 13 has to survive the wrapper too."""
    bull = SequenceGrader(round_trip_cost=0.0).grade(edge=0.9, risk=0.05, expected=0.5)
    bear = SequenceGrader(round_trip_cost=0.0).grade(edge=-0.9, risk=0.05, expected=-0.5)
    assert (bull, bear) == ("Strong Buy", "Strong Sell")


@pytest.mark.parametrize("risk", [0.0, 0.1, 0.25, 0.35, 0.5])
def test_grading_never_returns_something_outside_the_scale(risk: float) -> None:
    grader = SequenceGrader(round_trip_cost=0.0)
    for edge in (-0.9, -0.2, -0.02, 0.0, 0.02, 0.2, 0.9):
        assert grader.grade(edge, risk, expected=edge) in RECOMMENDATIONS
