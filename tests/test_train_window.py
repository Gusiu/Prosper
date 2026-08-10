"""Where the training window comes from, now that it is not a constant.

For most of the project's life every horizon trained on the same 730 days. That
is defensible for a 7-day horizon and indefensible for a 364-day one, and the
difference is not a matter of taste: a window of W bars is worth
``(W - forward) / forward`` independent observations, because consecutive labels
share all but one bar of their forward return. At 730 days that is 104
observations for the week and 1.0 for the year — and a single observation was
being reported to three decimal places.

These tests pin the rule that replaced it and, more importantly, the constraint
the rule cannot escape: the history is finite, so every bar given to training is
a bar that cannot be scored.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from prosper.domain import DEFAULT_HORIZONS, MIN_INDEPENDENT_OBSERVATIONS, effective_samples
from prosper.predict.window import (
    MIN_TRAIN_SAMPLES,
    MIN_TRAINING_ROWS,
    dynamic_window_steps,
    resolve_train_windows,
)

# Daily bars actually on disk, so the numbers below are the ones a run produces.
BTC_BARS = 3240
SOL_BARS = 2150


def _train_obs(window: int, forward: int) -> float:
    return effective_samples(max(0, window - forward), forward)


def _scored_obs(window: int, forward: int, history: int) -> float:
    return effective_samples(max(0, history - window - forward), forward)


# ── The rule ─────────────────────────────────────────────────────────────────

def test_short_horizons_are_sized_by_the_rows_a_model_needs() -> None:
    """At a 7-bar horizon almost the whole window is labelled, so observations
    are cheap and the binding constraint is having enough rows to fit at all."""
    week = dynamic_window_steps(7, BTC_BARS)

    assert week == 7 + MIN_TRAINING_ROWS
    assert _train_obs(week, 7) > 100, "observations were never the problem here"


def test_long_horizons_are_sized_by_the_observations_statistics_needs() -> None:
    """At 364 bars the row floor is met long before the observation floor is, so
    the observation floor has to be the one that decides."""
    year = dynamic_window_steps(364, BTC_BARS)

    assert year == 364 * (1 + MIN_INDEPENDENT_OBSERVATIONS)
    assert year == 1456
    assert _train_obs(year, 364) == pytest.approx(MIN_INDEPENDENT_OBSERVATIONS, abs=0.01)


def test_the_two_floors_swap_over_between_the_horizons() -> None:
    """Neither floor alone would do: one starves the long horizons of
    observations, the other starves the short ones of rows."""
    binding = {}
    for horizon in DEFAULT_HORIZONS:
        forward = horizon.steps("1d")
        rows = MIN_TRAINING_ROWS
        observations = MIN_INDEPENDENT_OBSERVATIONS * forward
        binding[horizon.name] = "rows" if rows >= observations else "observations"

    assert binding == {
        "week": "rows",
        "month": "rows",
        "quarter": "rows",
        "year": "observations",
    }


def test_every_horizon_clears_both_floors_on_the_history_we_have() -> None:
    """The point of the exercise. On BTCUSDT's nine years, all four horizons come
    out measurable on both sides — which the fixed 730-day window did not manage
    for the year at all."""
    for horizon in DEFAULT_HORIZONS:
        forward = horizon.steps("1d")
        window = dynamic_window_steps(forward, BTC_BARS)

        assert _train_obs(window, forward) >= MIN_INDEPENDENT_OBSERVATIONS, horizon.name
        assert _scored_obs(window, forward, BTC_BARS) >= MIN_INDEPENDENT_OBSERVATIONS, horizon.name


# ── The constraint the rule cannot escape ────────────────────────────────────

def test_a_horizon_the_history_cannot_support_is_not_quietly_widened() -> None:
    """SOLUSDT has 2150 daily bars. The ideal annual window is 1456 of them,
    which would leave 0.9 scored observations — so the ideal is unreachable, and
    what comes back is the balance point rather than the ideal.
    """
    window = dynamic_window_steps(364, SOL_BARS)

    assert window < dynamic_window_steps(364, None), "the ideal must not be used here"
    assert window == pytest.approx(SOL_BARS // 2, abs=1)

    train = _train_obs(window, 364)
    scored = _scored_obs(window, 364, SOL_BARS)
    assert train < MIN_INDEPENDENT_OBSERVATIONS
    assert scored < MIN_INDEPENDENT_OBSERVATIONS
    assert train == pytest.approx(scored, abs=0.1), (
        "when neither side can be satisfied, neither may be starved for the other"
    )


def test_the_balance_point_is_the_best_any_window_could_do() -> None:
    """Not an arbitrary fallback: half the history maximises whichever side is
    weaker, so no other width would have rescued the horizon either."""
    forward, history = 364, SOL_BARS
    chosen = dynamic_window_steps(forward, history)
    best = min(_train_obs(chosen, forward), _scored_obs(chosen, forward, history))

    for candidate in range(forward + MIN_TRAIN_SAMPLES, history - forward, 25):
        weaker = min(_train_obs(candidate, forward), _scored_obs(candidate, forward, history))
        assert weaker <= best + 0.05, f"{candidate} bars would have beaten the choice"


def test_more_history_is_what_actually_helps() -> None:
    """The honest conclusion from the test above: the fix for an unmeasurable
    horizon is more data, not a better setting."""
    poor = dynamic_window_steps(364, SOL_BARS)
    rich = dynamic_window_steps(364, BTC_BARS)

    assert _train_obs(rich, 364) > _train_obs(poor, 364)


# ── Resolution and precedence ────────────────────────────────────────────────

def test_each_horizon_gets_its_own_width() -> None:
    windows = resolve_train_windows(None, "1d", DEFAULT_HORIZONS, history_bars=BTC_BARS)

    widths = [windows.window_steps[h.name] for h in DEFAULT_HORIZONS]
    assert widths == sorted(widths), "a longer horizon never gets a narrower window"
    assert len(set(widths)) == len(widths), "they are genuinely different"


def test_an_explicit_width_overrides_the_rule() -> None:
    windows = resolve_train_windows(
        None, "1d", DEFAULT_HORIZONS, {"year": 900}, history_bars=BTC_BARS
    )

    assert windows.window_steps["year"] == 900
    assert windows.window_steps["week"] == dynamic_window_steps(7, BTC_BARS), "others untouched"


def test_an_explicit_width_may_be_narrower_than_the_rule_wants() -> None:
    """Reproducing an older run means asking for what that run used, including
    when the answer is now known to be too narrow."""
    windows = resolve_train_windows(730, "1d", DEFAULT_HORIZONS, history_bars=BTC_BARS)

    assert {windows.window_steps[h.name] for h in DEFAULT_HORIZONS} == {730}


def test_a_per_horizon_override_beats_a_shared_width() -> None:
    windows = resolve_train_windows(
        730, "1d", DEFAULT_HORIZONS, {"year": 1456}, history_bars=BTC_BARS
    )

    assert windows.window_steps["year"] == 1456
    assert windows.window_steps["month"] == 730


def test_the_window_scales_with_the_interval() -> None:
    """Horizons are calendar spans, so a 1h run must mean the same real duration
    (invariant 2). The window is expressed in bars and has to follow."""
    hourly = resolve_train_windows(730, "1h", DEFAULT_HORIZONS, history_bars=100_000)

    assert hourly.window_steps["week"] == 730 * 24


# ── The pairing that must not go wrong ───────────────────────────────────────

def test_bounds_pair_each_horizon_with_its_own_span() -> None:
    """The window and the forward span used to be a loose number and a dict, and
    pairing one horizon's window with another's span would train on labels the
    model could not have known. They travel together now."""
    windows = resolve_train_windows(None, "1d", DEFAULT_HORIZONS, history_bars=BTC_BARS)
    row = 3000

    for horizon in DEFAULT_HORIZONS:
        start, end = windows.bounds(row, horizon.name)
        forward = windows.forward_steps[horizon.name]

        assert start == max(0, row - windows.window_steps[horizon.name])
        # The newest label sits at end - 1 and needs close[end - 1 + forward],
        # which must already be behind the predicted bar.
        assert end - 1 + forward < row


def test_the_widest_window_covers_every_horizon() -> None:
    """Normalisation is fitted once per bar and shared, so its span has to reach
    the start of the deepest training slice."""
    windows = resolve_train_windows(None, "1d", DEFAULT_HORIZONS, history_bars=BTC_BARS)

    assert windows.widest_window_steps == max(windows.window_steps.values())
    for horizon in DEFAULT_HORIZONS:
        start, _ = windows.bounds(3000, horizon.name)
        assert 3000 - windows.widest_window_steps <= start


# ── Wired into every predictor ───────────────────────────────────────────────

PREDICTORS = ("baseline", "ml", "xgboost_model", "gru", "tft")


@pytest.mark.parametrize("module", PREDICTORS)
def test_every_predictor_reports_its_power(module: str) -> None:
    """`resolve_train_windows` deliberately does not warn, because the useful
    warning needs the scored range and that is only known after the load. So
    each predictor has to ask — and the reason this is a test is that the
    warning shipped absent for months while the annual horizon was reported to
    three decimals on one observation.
    """
    source = Path(f"src/prosper/predict/{module}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "resolve_train_windows" in called, f"{module} does not resolve its windows"
    assert "warn_low_power" in called, f"{module} never reports its statistical power"
    assert "count_scored_bars" in called, f"{module} warns without the scored side"


@pytest.mark.parametrize("module", PREDICTORS)
def test_no_predictor_slices_a_training_set_by_hand(module: str) -> None:
    """`windows.bounds` is the only way to get a training slice. A direct
    `training_bounds(...)` call takes the window as a loose argument again, which
    is exactly how one horizon's width would end up paired with another's span.
    """
    source = Path(f"src/prosper/predict/{module}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "training_bounds"
    ]

    assert not direct, f"{module} calls training_bounds directly; use windows.bounds"


def test_the_predictors_take_the_flag() -> None:
    from prosper.predict.baseline import predict_baseline
    from prosper.predict.gru import predict_gru
    from prosper.predict.ml import predict_ml
    from prosper.predict.tft import predict_tft
    from prosper.predict.xgboost_model import predict_xgboost

    for fn in (predict_baseline, predict_ml, predict_xgboost, predict_gru, predict_tft):
        assert "train_window_by_horizon" in inspect.signature(fn).parameters, fn.__name__
