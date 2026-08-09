"""The execution assumptions are inputs, and the comparisons are not.

Fees, slippage, the cost hurdle, the rebalance floor, the exposure targets and
the trend benchmark's lookback used to be constants in the module or fields in
Settings. That made the simulation a verdict when it should be an instrument.

The risk in opening them is obvious: a knob-twiddling machine that produces a
flattering number. What prevents it is that `nulls` travels with every result
whatever the settings — measured on one run, an aggressive policy with zero fees
*lowered* ROI from 74.05% to 61.71% and moved the mistimed-replay percentile from
45.5 to 40.0, because changing the size of the bets cannot create timing.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from prosper.eval.backtest import (
    EXPOSURE_KEYS,
    EXPOSURE_POLICY,
    MOMENTUM_LOOKBACK_BARS,
    parse_exposure_policy,
    run_backtest,
)

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def test_no_policy_means_the_default_is_used_untouched() -> None:
    assert parse_exposure_policy(None) is None
    assert parse_exposure_policy("   ") is None


def test_a_partial_policy_merges_onto_the_default() -> None:
    policy = parse_exposure_policy("buy=0.5")

    assert policy["Buy"] == 0.5
    assert policy["Strong Buy"] == EXPOSURE_POLICY["Strong Buy"], "untouched grades keep default"
    assert EXPOSURE_POLICY["Buy"] == 0.7, "the module default must not be mutated"


def test_display_names_are_accepted_too() -> None:
    assert parse_exposure_policy("Strong Buy=0.8")["Strong Buy"] == 0.8
    assert parse_exposure_policy("strong-buy=0.8")["Strong Buy"] == 0.8


def test_hold_cannot_be_given_a_target() -> None:
    """It means "leave the book alone" — the absence of a target, not zero. A
    number here would turn a neutral forecast into a trade signal."""
    with pytest.raises(ValueError, match="Hold carries the previous exposure"):
        parse_exposure_policy("hold=0")

    assert "hold" not in EXPOSURE_KEYS


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("nonsense=1", "Unknown grade"),
        ("buy=2", r"within \[0, 1\]"),
        ("buy=-0.1", r"within \[0, 1\]"),
        ("buy=abc", "must be a number"),
        ("buy", "Expected 'grade=share'"),
    ],
)
def test_a_bad_policy_is_refused_rather_than_ignored(raw: str, message: str) -> None:
    """Silently dropping one would leave the caller believing an assumption was
    applied — the failure that let the dashboard train 5 epochs while the CLI
    trained 20."""
    with pytest.raises(ValueError, match=message):
        parse_exposure_policy(raw)


def test_no_leverage_and_no_shorts_can_be_expressed() -> None:
    """Long-only on SPOT: every share stays a fraction of equity."""
    for share in ("0", "0.5", "1", "1.0"):
        assert parse_exposure_policy(f"buy={share}")["Buy"] == float(share)
    with pytest.raises(ValueError):
        parse_exposure_policy("buy=1.5")


def test_every_assumption_is_a_parameter_of_the_simulation() -> None:
    accepted = set(inspect.signature(run_backtest).parameters)
    for name in (
        "fee_rate",
        "slippage_rate",
        "round_trip_cost",
        "min_rebalance_fraction",
        "momentum_lookback",
        "exposure_policy",
        "initial_capital",
    ):
        assert name in accepted, name


def test_the_api_forwards_every_one_of_them() -> None:
    """The route builds its response and its call by hand, so a dropped
    parameter would silently fall back to the default."""
    from prosper.api.routes.backtest import run_backtest_api

    accepted = set(inspect.signature(run_backtest_api).parameters)
    for name in (
        "fee_rate",
        "slippage_rate",
        "round_trip_cost",
        "min_rebalance",
        "momentum_lookback",
        "exposure",
    ):
        assert name in accepted, name

    source = inspect.getsource(run_backtest_api)
    assert "parse_exposure_policy" in source
    # A malformed string is the caller's mistake, so 400 rather than 500.
    assert "status_code=400" in source


def test_the_cli_offers_the_same_set() -> None:
    import typer
    from prosper.cli import app

    command = typer.main.get_command(app).commands["eval"].commands["backtest"]  # type: ignore[attr-defined]
    options = {opt for param in command.params for opt in getattr(param, "opts", [])}
    for flag in (
        "--fee-rate",
        "--slippage-rate",
        "--round-trip-cost",
        "--min-rebalance",
        "--momentum-lookback",
        "--exposure",
    ):
        assert flag in options, flag


def test_the_defaults_endpoint_reads_the_real_constants() -> None:
    from prosper.api.routes.meta import get_backtest_defaults

    payload = get_backtest_defaults()

    assert payload["momentum_lookback_bars"] == MOMENTUM_LOOKBACK_BARS
    grades = {row["label"]: row["default"] for row in payload["exposure"]}
    assert grades == {label: EXPOSURE_POLICY[label] for label in EXPOSURE_KEYS.values()}
    assert "Hold" not in grades, "Hold has no target and must not be offered"


def test_the_form_carries_no_hardcoded_assumption() -> None:
    """A number typed into the markup is a second definition; the epoch field
    already cost this project a run trained at 5 epochs while reporting 20."""
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")

    for field in ("bt-fee", "bt-slippage", "bt-hurdle", "bt-rebalance", "bt-momentum",
                  "backtest-capital"):
        match = re.search(rf'id="{field}"[^>]*>', html)
        assert match, f"{field} not found"
        assert 'value="' not in match.group(), f"{field} carries a literal: {match.group()}"

    # The exposure rows are rendered from the endpoint, not written out.
    assert 'id="backtest-exposure"' in html
    assert "Strong Buy (%)" not in html


def test_the_reset_control_is_wired() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    main_js = (FRONTEND / "js" / "main.js").read_text(encoding="utf-8")

    assert 'data-action="backtest-reset"' in html
    assert '"backtest-reset": () => resetBacktestAssumptions()' in main_js
    assert "loadBacktestDefaults()" in main_js


def test_the_browser_only_sends_what_was_changed() -> None:
    """Sending every field on each call would make the request differ from a
    plain CLI invocation for no reason, and put the browser in charge of
    defaults it should not own."""
    js = (FRONTEND / "js" / "tabs" / "backtest.js").read_text(encoding="utf-8")

    assert "Math.abs(share - grade.default) > 1e-9" in js
    assert "if (value !== null && value !== \"\") params.set" in js
