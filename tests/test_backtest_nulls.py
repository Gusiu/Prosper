"""Buy and hold is the weakest of four comparisons, not the only one.

Invariant 7 requires the benchmark because a return figure alone cannot say
whether the model added anything. That turned out not to finish the job. A
long-only strategy that sits flat through high-volatility months finishes ahead
of holding by compounding alone: ETHUSDT tft on a 2023-start window beat holding
by 11.06 pp while its cumulative monthly excess was negative in **all 40
months**, and it was flat during the largest months in both directions — 2024-02
at 1% exposure into a +44.95% month, 2025-11 at 0% into a -22.78% one. That is
variance reduction, not forecasting.

So the report now carries three more comparisons, ordered weakest to strongest:

1. the same average exposure held throughout — rules out carrying less beta
2. the same exposure path mistimed by every circular shift — rules out
   volatility timing
3. a 7-bar momentum rule with no model in it — rules out "any trend rule would
   have done this", which measurement says is the binding one: across the 1d
   runs that rule scores a median 90.5 percentile against its own shifts while
   the models' own policy scores 34.1.
"""

from __future__ import annotations

import pytest
from prosper.eval.backtest import (
    MOMENTUM_LOOKBACK_BARS,
    _naive_momentum,
    _replay,
    _timing_test,
)

RISING = [100.0 * (1.03**i) for i in range(60)]


def _walk(n: int = 200, seed: int = 7) -> list[float]:
    """A seeded pseudo-random walk, deliberately not periodic.

    A short repeating series is the obvious fixture here and it silently breaks
    the test: a shift by any multiple of the period reproduces the *original*
    alignment, so half the "mistimed" copies are perfectly timed and even
    clairvoyant exposure only reaches the 75th percentile. The same trap cost
    the sub-daily stability test its first version, where a 24-hour oscillation
    cancelled out of a 672-hour horizon.
    """
    import random

    rng = random.Random(seed)
    prices, level = [], 100.0
    for _ in range(n):
        level *= 1.0 + rng.gauss(0.002, 0.03)
        prices.append(level)
    return prices


WALK = _walk()


def test_full_exposure_with_no_costs_reproduces_the_price_return() -> None:
    roi = _replay(RISING, [1.0] * len(RISING), cost_rate=0.0)
    assert roi == pytest.approx(RISING[-1] / RISING[0] - 1.0, rel=1e-9)


def test_zero_exposure_earns_nothing() -> None:
    assert _replay(RISING, [0.0] * len(RISING), cost_rate=0.001) == pytest.approx(0.0)


def test_turnover_is_charged_on_every_change() -> None:
    flat = [100.0] * 20
    # Ten round trips on a flat market can only lose the fees.
    churn = [float(i % 2) for i in range(20)]
    assert _replay(flat, churn, cost_rate=0.01) < 0.0


def test_perfect_foresight_beats_its_own_mistimed_copies() -> None:
    """The test has to be able to detect skill, or it detects nothing."""
    exposures = [
        1.0 if i + 1 < len(WALK) and WALK[i + 1] > WALK[i] else 0.0
        for i in range(len(WALK))
    ]
    result = _timing_test(WALK, exposures, cost_rate=0.0)

    assert result["percentile"] > 95.0
    assert result["samples"] == len(WALK) - 1


def test_inverted_foresight_lands_at_the_bottom() -> None:
    exposures = [
        0.0 if i + 1 < len(WALK) and WALK[i + 1] > WALK[i] else 1.0
        for i in range(len(WALK))
    ]
    assert _timing_test(WALK, exposures, cost_rate=0.0)["percentile"] < 5.0


def test_a_constant_exposure_path_reads_as_no_information() -> None:
    """Every shift of a constant path is the same path.

    Counting only strict wins put this at percentile 0, which reads as the worst
    timing achievable when the truth is that a path with no variation carries no
    timing at all. Ties count half, so it lands at 50.
    """
    for level in (0.0, 0.35, 1.0):
        result = _timing_test(RISING, [level] * len(RISING), cost_rate=0.001)
        assert result["percentile"] == pytest.approx(50.0)


def test_a_series_too_short_to_shift_is_reported_as_such() -> None:
    result = _timing_test([100.0, 101.0], [1.0, 1.0], cost_rate=0.0)
    assert result["samples"] == 0
    assert result["percentile"] is None


def test_the_momentum_rule_holds_a_rising_market_and_nothing_else() -> None:
    rising = _naive_momentum(RISING, cost_rate=0.0, lookback=MOMENTUM_LOOKBACK_BARS)
    falling = _naive_momentum(
        [100.0 * (0.97**i) for i in range(60)], cost_rate=0.0, lookback=MOMENTUM_LOOKBACK_BARS
    )

    # Fully invested once the lookback is available, so it tracks the asset.
    assert rising["mean_exposure"] > 0.8
    assert rising["roi_pct"] > 0.0
    # Never invested in a monotone decline, so it neither loses nor gains.
    assert falling["mean_exposure"] == pytest.approx(0.0)
    assert falling["roi_pct"] == pytest.approx(0.0)


def test_the_momentum_rule_contains_no_fitted_parameter() -> None:
    """Its only knob is a constant, and it never sees a label or a forecast."""
    import inspect

    source = inspect.getsource(_naive_momentum)
    for forbidden in ("predict", "recommendation", "trained", "P_long", "edge"):
        assert forbidden not in source, forbidden


def test_a_series_too_short_for_the_lookback_yields_nothing() -> None:
    result = _naive_momentum([100.0, 101.0, 102.0], cost_rate=0.0, lookback=7)
    assert result["roi_pct"] is None


def test_a_wipeout_stops_at_total_loss() -> None:
    """Leverage is not modelled, so equity must not be allowed to go negative."""
    crash = [100.0, 50.0, 1.0, 0.5]
    assert _replay(crash, [1.0] * 4, cost_rate=0.0) > -1.0
    assert _replay(crash, [1.0] * 4, cost_rate=0.9) == pytest.approx(-1.0)


# ── The comparisons have to reach a reader ───────────────────────────────────

def test_the_api_forwards_the_whole_comparison_set() -> None:
    """The route builds its response field by field, so a new one is easy to
    drop — and a benchmark the browser never sees is a benchmark nobody reads."""
    import inspect

    from prosper.api.routes import backtest as route

    source = inspect.getsource(route.run_backtest_api)
    assert '"nulls"' in source
    assert '"benchmark"' in source


def test_the_backtest_tab_shows_every_comparison() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "frontend"
    html = (root / "index.html").read_text(encoding="utf-8")
    assert 'id="backtest-nulls-tbody"' in html

    js = (root / "js" / "tabs" / "backtest.js").read_text(encoding="utf-8")
    assert "renderBacktestNulls" in js
    for key in ("constant_exposure", "mistimed_replay", "naive_momentum"):
        assert key in js, key


def test_the_ui_only_uses_css_classes_that_exist() -> None:
    """`text-yellow` and `table-scroll` were invented in a first draft and would
    have rendered unstyled — silent, and only visible by looking."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "frontend"
    css = (root / "style.css").read_text(encoding="utf-8")
    js = (root / "js" / "tabs" / "backtest.js").read_text(encoding="utf-8")

    used = set(re.findall(r'class="(text-[a-z-]+)"', js))
    used |= set(re.findall(r'\{[^}]*\?\s*"(text-[a-z-]+)"\s*:\s*"(text-[a-z-]+)"', js)[0]) if (
        re.findall(r'\{[^}]*\?\s*"(text-[a-z-]+)"\s*:\s*"(text-[a-z-]+)"', js)
    ) else set()
    assert used, "no colour classes found; the check would pass vacuously"
    for name in used:
        assert f".{name}" in css, f"{name} is not defined in style.css"
