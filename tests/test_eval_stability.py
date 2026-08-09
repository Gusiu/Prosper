"""Quality over time, and the scoring it shares with the rest of eval.

`eval walkforward` reported three things and owned one. Its per-horizon logloss
and Brier duplicated `eval predictions`; its PnL duplicated `eval backtest` and
disagreed with it, because it carried a second copy of EXPOSURE_POLICY mapping
`Hold` to 0.0 where the backtest keeps the previous exposure. It also called
itself walk-forward while retraining nothing. What only it can answer is
whether quality holds up over time, so that is all it answers now.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from prosper.config import Settings
from prosper.domain import HORIZON_NAMES, direction_from_return
from prosper.eval.stability import (
    MIN_MONTH_SAMPLES,
    eval_stability,
    multiclass_logloss_brier,
)
from prosper.storage.layout import get_eval_stability_summary_path, get_parquet_file_path
from prosper.storage.parquet import save_parquet

SYMBOL = "TESTUSDT"
# Read from the specs so the tests survive the next change to the horizon set.
SHORTEST, LONGEST = HORIZON_NAMES[0], HORIZON_NAMES[-1]


def test_direction_is_the_sign_of_the_return() -> None:
    assert direction_from_return(0.02) == "long"
    assert direction_from_return(-0.03) == "short"
    # No threshold and no flat class: a 0.2% move is a real move. Whether it
    # is worth trading is a cost question, answered by the planner.
    assert direction_from_return(0.002) == "long"
    assert direction_from_return(-0.002) == "short"


def test_an_exactly_zero_return_has_no_side() -> None:
    """It must not be forced into a class — the row is excluded instead."""
    assert direction_from_return(0.0) is None
    assert direction_from_return(None) is None


def test_multiclass_logloss_brier_are_finite_positive() -> None:
    logloss, brier = multiclass_logloss_brier({"long": 0.7, "short": 0.3}, "long")

    assert logloss >= 0.0
    assert 0.0 <= brier <= 2.0
    assert brier == pytest.approx(0.09 + 0.09)


def test_scoring_follows_the_run_s_own_class_set() -> None:
    """A legacy three-class run stays scorable by the same function."""
    logloss, brier = multiclass_logloss_brier({"long": 0.7, "flat": 0.2, "short": 0.1}, "long")

    assert logloss == pytest.approx(-math.log(0.7))
    assert brier == pytest.approx(0.09 + 0.04 + 0.01)


def test_a_class_the_run_never_emitted_scores_as_a_miss() -> None:
    logloss, _ = multiclass_logloss_brier({"long": 0.7, "short": 0.3}, "flat")
    assert logloss > 0.0


# ── The report itself ─────────────────────────────────────────────────────────


def _horizon(p_long: float, trained: bool = True) -> dict:
    depth = {"1-2": 0.5, "2-3": 0.5}
    return {
        "P_long": p_long,
        "P_short": round(1.0 - p_long, 6),
        "depth_long_bins": depth,
        "depth_short_bins": depth,
        "trained": trained,
    }


def _lake(tmp_path, days: int = 200, rising: bool = True) -> Settings:
    settings = Settings(data_root=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    closes = [100.0 + i if rising else 300.0 - i for i in range(days)]
    df = pl.DataFrame(
        {
            "open_time": [start + timedelta(days=i) for i in range(days)],
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000.0] * days,
        }
    ).with_columns(
        pl.col("open_time").dt.year().alias("_y"), pl.col("open_time").dt.month().alias("_m")
    )
    for (year, month), group in df.group_by(["_y", "_m"]):
        save_parquet(
            group.drop(["_y", "_m"]).sort("open_time"),
            get_parquet_file_path(SYMBOL, "1d", int(year), month=int(month), settings=settings),
        )

    run_dir = settings.reports_predictions_dir / SYMBOL / "ml_1d_20240101000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(days):
        moment = start + timedelta(days=i)
        # Confidently long in January, confidently short afterwards. On a rising
        # series that makes January excellent and the rest terrible — which is
        # exactly the shape a whole-run average hides.
        p_long = 0.95 if moment.month == 1 else 0.05
        rows.append(
            {
                "open_time": moment.isoformat(),
                "date": moment.date().isoformat(),
                "symbol": SYMBOL,
                # Every horizon but the longest is trained; the longest stands
                # in for one the model could not fit (invariant 3).
                **{name: _horizon(p_long) for name in HORIZON_NAMES[:-1]},
                HORIZON_NAMES[-1]: _horizon(p_long, trained=False),
            }
        )
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return settings


def test_a_regime_change_is_visible_month_by_month(tmp_path) -> None:
    settings = _lake(tmp_path)
    report = eval_stability(
        SYMBOL, "2024-01-01", "2024-05-31", settings=settings, model_type="ml", interval="1d"
    )

    by_month = {m["month"]: m["horizons"][SHORTEST] for m in report["months"]}
    assert by_month["2024-01"]["accuracy"] == 1.0, "confidently long on a rising series"
    assert by_month["2024-03"]["accuracy"] == 0.0, "confidently short on the same series"
    # The aggregate sits between the two and describes neither.
    assert 0.0 < report["overall"][SHORTEST]["accuracy"] < 1.0
    assert by_month["2024-01"]["logloss"] < by_month["2024-03"]["logloss"]


def test_an_untrained_horizon_contributes_nothing(tmp_path) -> None:
    """Invariant 3: the uniform prior is not a forecast."""
    settings = _lake(tmp_path)
    report = eval_stability(
        SYMBOL, "2024-01-01", "2024-05-31", settings=settings, model_type="ml", interval="1d"
    )

    assert report["overall"][LONGEST]["samples"] == 0
    assert report["overall"][LONGEST]["logloss"] is None
    assert all(m["horizons"][LONGEST]["samples"] == 0 for m in report["months"])


def test_a_thin_month_is_flagged_rather_than_dropped(tmp_path) -> None:
    """A mean over three bars moves for reasons unrelated to the model, but
    hiding the month would make the series look denser than it is."""
    settings = _lake(tmp_path)
    report = eval_stability(
        SYMBOL, "2024-01-01", "2024-03-03", settings=settings, model_type="ml", interval="1d"
    )

    march = next(m for m in report["months"] if m["month"] == "2024-03")["horizons"][SHORTEST]
    assert 0 < march["samples"] < MIN_MONTH_SAMPLES
    assert march["sparse"] is True
    january = next(m for m in report["months"] if m["month"] == "2024-01")["horizons"][SHORTEST]
    assert january["sparse"] is False


def test_dates_are_full_iso_like_every_sibling_command(tmp_path) -> None:
    """The old command took YYYY-MM here and YYYY-MM-DD for its twin."""
    settings = _lake(tmp_path)
    report = eval_stability(
        SYMBOL, "2024-02-01", "2024-02-29", settings=settings, model_type="ml", interval="1d"
    )

    assert report["start"] == "2024-02-01"
    assert report["end"] == "2024-02-29"
    assert [m["month"] for m in report["months"]] == ["2024-02"]


def test_the_report_carries_no_returns(tmp_path) -> None:
    """The backtest owns PnL, and owned it correctly; this duplicated it with a
    divergent exposure policy."""
    settings = _lake(tmp_path)
    report = eval_stability(
        SYMBOL, "2024-01-01", "2024-05-31", settings=settings, model_type="ml", interval="1d"
    )

    text = json.dumps(report).lower()
    for banned in ("pnl", "exposure", "recommendation", "train_months", "step_months"):
        assert banned not in text, f"{banned} is another module's job"


def test_the_artifact_is_written_under_the_source_run(tmp_path) -> None:
    settings = _lake(tmp_path)
    report = eval_stability(
        SYMBOL, "2024-01-01", "2024-05-31", settings=settings, model_type="ml", interval="1d"
    )

    path = get_eval_stability_summary_path(
        SYMBOL, settings=settings, run_slug="ml_1d_20240101000000"
    )
    assert path.exists()
    assert report["report_path"] == str(path)
    assert json.loads(path.read_text(encoding="utf-8"))["run"]["slug"] == "ml_1d_20240101000000"


def test_it_does_not_need_the_planner_to_have_run(tmp_path) -> None:
    """Dropping the PnL dropped the recommendations dependency with it."""
    settings = _lake(tmp_path)
    assert not (settings.reports_recommendations_dir / SYMBOL).exists()

    report = eval_stability(
        SYMBOL, "2024-01-01", "2024-05-31", settings=settings, model_type="ml", interval="1d"
    )
    assert report["overall"][SHORTEST]["samples"] > 0


# ── Sub-daily runs ───────────────────────────────────────────────────────────

def _hourly_lake(tmp_path, hours: int = 24 * 90) -> Settings:
    """An hourly series whose 28-day-ahead sign differs across bars of one date.

    The oscillation period must not divide the horizon. A 24-hour sawtooth looks
    like the obvious choice and is useless here: the short horizon is 672 hours,
    exactly 28 whole days, so `close[i + 672]` sits at the identical phase and
    the oscillation cancels out of every forward return. A 100-hour period
    leaves 0.72 of a cycle over the horizon, so the sign genuinely varies with
    the bar — which is what makes a date-keyed lookup detectably wrong.
    """
    settings = Settings(data_root=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    times = [start + timedelta(hours=i) for i in range(hours)]
    closes = [
        200.0 + 0.001 * i + 20.0 * math.sin(2.0 * math.pi * i / 100.0) for i in range(hours)
    ]

    df = pl.DataFrame(
        {
            "open_time": times,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [10.0] * hours,
        }
    ).with_columns(
        pl.col("open_time").dt.year().alias("_y"), pl.col("open_time").dt.month().alias("_m")
    )
    for (year, month), group in df.group_by(["_y", "_m"]):
        save_parquet(
            group.drop(["_y", "_m"]).sort("open_time"),
            get_parquet_file_path(SYMBOL, "1h", int(year), month=int(month), settings=settings),
        )

    run_dir = settings.reports_predictions_dir / SYMBOL / "ml_1h_20240101000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "open_time": moment.isoformat(),
            "date": moment.date().isoformat(),
            "symbol": SYMBOL,
            **{name: _horizon(0.8) for name in HORIZON_NAMES},
        }
        for moment in times
    ]
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return settings


def test_an_hourly_run_is_scored_bar_by_bar_not_date_by_date(tmp_path) -> None:
    """Invariant 4. The predecessor indexed realised returns on the calendar
    date and read them off the 1d series, so all 24 hourly bars of a day shared
    one outcome and the horizon was measured from the daily close instead of
    from the bar the forecast was made on.
    """
    settings = _hourly_lake(tmp_path)
    report = eval_stability(
        SYMBOL, "2024-01-01", "2024-02-15", settings=settings, model_type="ml", interval="1h"
    )

    scored = report["overall"][SHORTEST]["samples"]
    # Far more scored rows than there are calendar dates in the window.
    assert scored > 46 * 20, f"only {scored} rows scored; dates collapsed"

    # The sawtooth makes the 28-day-ahead sign differ across bars of one date,
    # so accuracy must land strictly between the two degenerate values a
    # date-keyed lookup would produce.
    accuracy = report["overall"][SHORTEST]["accuracy"]
    assert 0.0 < accuracy < 1.0


def test_the_horizon_is_converted_to_the_run_s_own_interval(tmp_path) -> None:
    """`long` must mean a year at 1h too, not 365 hours (invariant 2).

    With 90 days of history no bar can have a realised annual return, so the
    long horizon has to be unscoreable. Treating forward_days as a bar count
    would have found 8760-hour... no: it would have found 365 *hours* ahead and
    happily scored almost every bar.
    """
    settings = _hourly_lake(tmp_path)
    report = eval_stability(
        SYMBOL, "2024-01-01", "2024-02-15", settings=settings, model_type="ml", interval="1h"
    )

    assert report["overall"][LONGEST]["samples"] == 0
    assert report["overall"][SHORTEST]["samples"] > 0


def test_the_bar_key_normalises_every_open_time_shape() -> None:
    from prosper.eval.stability import bar_key

    assert bar_key("2024-03-01T05:00:00+00:00") == "2024-03-01T05:00:00"
    assert bar_key("2024-03-01T05:00:00Z") == "2024-03-01T05:00:00"
    assert bar_key("2024-03-01T05:00:00.123456+00:00") == "2024-03-01T05:00:00"
    assert bar_key("2024-03-01 05:00:00") == "2024-03-01 05:00:00"[:19]
    # A legacy date-only row resolves to that day's first bar.
    assert bar_key("2024-03-01") == "2024-03-01T00:00:00"
