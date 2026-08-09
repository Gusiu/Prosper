"""The horizon set: whole weeks, commensurate, and named for its spans.

It was `short`/`medium`/`long` at 28/182/364 days. Three problems, none cosmetic.

`short` meant four weeks, which no reader guesses. Worse, `short` and `long` are
also the two `DIRECTION_CLASSES`, so one word named a horizon in one place and
"the price falls" in another — a collision in the project's own vocabulary.

And the half-year was the weakest choice available. A 730-day training window
holds (730 - 182) / 182 = 3.0 independent observations at 182 days against
(730 - 91) / 91 = 7.0 at 91, because labels overlap by all but one bar. Moving to
a quarter more than doubles the evidence for free.
"""

from __future__ import annotations

import pytest
from prosper.domain import (
    DEFAULT_HORIZONS,
    DEFAULT_TRADED_HORIZON,
    DIRECTION_CLASSES,
    HORIZON_NAMES,
    days_to_steps,
)

SPANS = {h.name: h.forward_days for h in DEFAULT_HORIZONS}


def test_the_names_do_not_collide_with_the_direction_classes() -> None:
    """`short` and `long` were both horizons and directions."""
    assert not set(HORIZON_NAMES) & set(DIRECTION_CLASSES)


def test_every_horizon_is_a_whole_number_of_weeks() -> None:
    """`days_to_steps` rounds, so anything else means the 1w interval asks a
    different question than 1d — which invariant 2 forbids."""
    for horizon in DEFAULT_HORIZONS:
        assert horizon.forward_days % 7 == 0, horizon.name
        assert days_to_steps(horizon.forward_days, "1w") * 7 == horizon.forward_days


def test_the_conversion_is_exact_on_every_commensurate_interval() -> None:
    for horizon in DEFAULT_HORIZONS:
        assert days_to_steps(horizon.forward_days, "1d") == horizon.forward_days
        assert days_to_steps(horizon.forward_days, "1h") == horizon.forward_days * 24
        assert days_to_steps(horizon.forward_days, "4h") == horizon.forward_days * 6


def test_the_spans_are_commensurate() -> None:
    assert SPANS["year"] == 4 * SPANS["quarter"], "four quarters to a year"
    assert SPANS["month"] == 4 * SPANS["week"]
    assert SPANS["year"] == 52 * SPANS["week"]


def test_the_set_is_ordered_shortest_first() -> None:
    """The order is load-bearing: the calendar cell, the traded default and the
    chart palette all take the first entry as the shortest."""
    spans = [h.forward_days for h in DEFAULT_HORIZONS]
    assert spans == sorted(spans)
    assert DEFAULT_TRADED_HORIZON == DEFAULT_HORIZONS[0].name


def _independent_observations(window_days: int, forward_days: int) -> float:
    """Non-overlapping labels a window holds at this horizon.

    Written out rather than imported so this file states the arithmetic the
    horizon choice rests on: the window holds `window - forward` labelled bars,
    and two labels share part of their interval until they are `forward` apart.
    """
    return max(0, window_days - forward_days) / forward_days


def test_a_quarter_carries_more_than_twice_a_half_year() -> None:
    """The measured reason the half-year was replaced, not a preference."""
    quarter = _independent_observations(730, 91)
    half_year = _independent_observations(730, 182)

    assert quarter == pytest.approx(7.0, abs=0.1)
    assert half_year == pytest.approx(3.0, abs=0.1)
    assert quarter > 2 * half_year


def test_the_shorter_horizons_are_the_well_powered_ones() -> None:
    """Which is why `week` was added, and why the year is reported as
    unmeasurable rather than quoted to three decimal places."""
    power = {
        h.name: _independent_observations(730, h.forward_days) for h in DEFAULT_HORIZONS
    }

    assert power["week"] > 100
    assert power["month"] > 20
    assert power["quarter"] > 3
    assert power["year"] < 3, "one observation; no model or budget changes that"


def test_a_run_with_unknown_horizons_fails_loudly(tmp_path) -> None:
    """A legacy run must not score zero rows and look like a model that
    predicted nothing.

    `prediction.get(name) or {}` followed by a test for `trained is False` let a
    missing horizon fall through to `normalize_probabilities({})`, which returns
    the uniform prior — scoring the prior as a forecast, which invariant 3 exists
    to prevent.
    """
    import json
    from datetime import UTC, datetime

    import polars as pl
    from prosper.config import Settings
    from prosper.eval.predictions import evaluate_predictions
    from prosper.storage.layout import get_parquet_file_path
    from prosper.storage.parquet import save_parquet

    symbol = "LEGACYUSDT"
    settings = Settings(data_root=tmp_path)
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    save_parquet(
        pl.DataFrame(
            {
                "open_time": [moment],
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [100.0], "volume": [1000.0],
            }
        ),
        get_parquet_file_path(symbol, "1d", 2024, month=1, settings=settings),
    )

    run_dir = settings.reports_predictions_dir / symbol / "ml_1d_20240101000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    legacy = {"P_long": 0.8, "P_short": 0.2, "depth_long_bins": {}, "depth_short_bins": {}}
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "open_time": moment.isoformat(),
                "date": "2024-01-01",
                "symbol": symbol,
                "short": legacy, "medium": legacy, "long": legacy,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="none of which this build knows"):
        evaluate_predictions(
            symbol, model_type="ml", timestamp="20240101000000",
            interval="1d", settings=settings,
        )
