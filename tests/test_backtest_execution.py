"""Execution-model guarantees of the trading simulator.

The previous version traded on the same bar's close, went all-in on any buy
signal, ignored slippage and reported no benchmark — so a positive ROI said
nothing about whether the strategy beat simply holding the asset.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from prosper.config import Settings
from prosper.eval.backtest import run_backtest
from prosper.storage.layout import get_parquet_file_path
from prosper.storage.parquet import save_parquet

SYMBOL = "TESTUSDT"
TIMESTAMP = "20240101000000"
# Mass concentrated in the small bins: a uniform depth distribution puts ~0.375
# into the "large move" bins, which pins the risk metric high enough that no
# signal can ever reach Buy or Strong Buy.
DEPTH = {
    "1-2": 0.45,
    "2-3": 0.30,
    "3-5": 0.15,
    "5-8": 0.06,
    "8-13": 0.02,
    "13-21": 0.01,
    "21-34": 0.005,
    "34+": 0.005,
}


def _write_prices(settings: Settings, closes: list[float], opens: list[float]) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "open_time": [start + timedelta(days=i) for i in range(len(closes))],
            "open": opens,
            "high": [max(o, c) * 1.01 for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) * 0.99 for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    ).with_columns(pl.col("open_time").dt.month().alias("_m"))
    for (month,), group in df.group_by(["_m"]):
        save_parquet(
            group.drop("_m").sort("open_time"),
            get_parquet_file_path(SYMBOL, "1d", 2024, month=int(month), settings=settings),
        )


# Mass far enough up the scale that a modest edge still clears the round-trip
# cost. With DEPTH above, the probability-weighted move at edge 0.10 is 0.31%
# against a 0.4% hurdle, so the cost gate holds — correctly, and that is its own
# test below.
DEPTH_WIDER = {"3-5": 0.4, "5-8": 0.4, "8-13": 0.2}


def _write_run(
    settings: Settings,
    signals: list[tuple[float, float]],
    depth: dict[str, float] | None = None,
) -> None:
    """`signals` is one (P_long, P_short) pair per day, starting 2024-01-01."""
    run_dir = settings.reports_predictions_dir / SYMBOL / f"ml_1d_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i, (p_long, p_short) in enumerate(signals):
        ts = start + timedelta(days=i)
        horizon = {
            "P_long": p_long,
            "P_flat": max(0.0, 1.0 - p_long - p_short),
            "P_short": p_short,
            "depth_long_bins": depth or DEPTH,
            "depth_short_bins": depth or DEPTH,
            "trained": True,
        }
        rows.append(
            json.dumps(
                {
                    "open_time": ts.isoformat(),
                    "date": ts.date().isoformat(),
                    "symbol": SYMBOL,
                    "short": horizon,
                    "medium": horizon,
                    "long": horizon,
                }
            )
        )
    (run_dir / "predictions.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _backtest(settings: Settings, **kwargs):
    defaults = dict(
        symbol=SYMBOL,
        start="2024-01-01",
        end="2024-02-28",
        settings=settings,
        model_type="ml",
        timestamp=TIMESTAMP,
        interval="1d",
    )
    defaults.update(kwargs)
    return run_backtest(**defaults)


def test_signals_execute_on_the_next_bar_not_their_own(tmp_path) -> None:
    """A signal from bar t must not be filled at bar t's own close.

    The series jumps on day 2. If the signal from day 1 filled at day 1's close
    it would capture the jump for free; filling at day 2's open must not.
    """
    settings = Settings(data_root=tmp_path)
    # flat, then a large gap up on day 2, then flat again
    closes = [100.0, 100.0, 200.0, 200.0, 200.0, 200.0]
    opens = [100.0, 100.0, 200.0, 200.0, 200.0, 200.0]
    _write_prices(settings, closes, opens)
    # Strong buy on every bar
    _write_run(settings, [(0.95, 0.01)] * len(closes))

    result = _backtest(settings, end="2024-01-06", fee_rate=0.0, slippage_rate=0.0)

    assert "error" not in result
    first = result["chart_data"][0]
    # The first signal is dated 2024-01-01 and fills on 2024-01-02.
    assert first["signal_date"] == "2024-01-01"
    assert first["date"] == "2024-01-02"
    assert first["exec_price"] == 100.0

    # Entering at 100 before a double means the run roughly doubles; entering at
    # the jumped price would not. Costs are zero here, so this is exact.
    assert result["roi_pct"] == pytest.approx(100.0, abs=1e-6)


def test_exposure_scales_with_conviction(tmp_path) -> None:
    """A weak signal must not commit the whole balance."""
    settings = Settings(data_root=tmp_path)
    closes = [100.0] * 8
    _write_prices(settings, closes, closes)
    # edge 0.10 -> "Accumulate" (0.4 target), never a full allocation. The wider
    # depth distribution is what lets the expected move pay for the round trip;
    # see the next test for the case where it cannot.
    _write_run(settings, [(0.45, 0.35)] * 8, depth=DEPTH_WIDER)

    result = _backtest(settings, end="2024-01-08", fee_rate=0.0, slippage_rate=0.0)

    assert "error" not in result
    exposures = {round(row["exposure"], 3) for row in result["chart_data"]}
    assert exposures == {0.4}


def test_a_signal_too_small_to_pay_its_costs_is_not_traded(tmp_path) -> None:
    """The cost gate applies in the simulation, not only in the planner.

    `run_backtest` used to call `map_to_recommendation` directly with the static
    defaults and never computed an expected move, so it skipped this gate
    entirely — and it also missed the quantile risk gates when those arrived.
    The same bar could be `Accumulate` here and `Hold` in the recommendation
    report the simulation was supposed to be trading. Both now go through
    `SequenceGrader`.

    With DEPTH the probability-weighted move at edge 0.10 is 0.31% against the
    0.4% round-trip hurdle, so nothing should be bought.
    """
    settings = Settings(data_root=tmp_path)
    closes = [100.0] * 8
    _write_prices(settings, closes, closes)
    _write_run(settings, [(0.45, 0.35)] * 8)

    result = _backtest(settings, end="2024-01-08", fee_rate=0.0, slippage_rate=0.0)

    assert "error" not in result
    assert {round(row["exposure"], 3) for row in result["chart_data"]} == {0.0}
    assert {row["recommendation"] for row in result["chart_data"]} == {"Hold"}


def test_the_simulation_and_the_planner_agree_on_every_bar(tmp_path) -> None:
    """One grader, so the report and the simulation cannot describe different
    trades — the defect that motivated `SequenceGrader`."""
    from prosper.planner.windows import plan_windows

    settings = Settings(data_root=tmp_path)
    closes = [100.0 + i for i in range(40)]
    _write_prices(settings, closes, closes)
    _write_run(settings, [(0.9, 0.05)] * 40, depth=DEPTH_WIDER)

    result = _backtest(settings, end="2024-02-09", fee_rate=0.0, slippage_rate=0.0)
    plan = plan_windows(
        SYMBOL, settings=settings, model_type="ml", interval="1d", timestamp=TIMESTAMP
    )

    simulated = {row["recommendation"] for row in result["chart_data"]}
    planned = {
        window["recommendation"] for window in plan["windows"] if window["horizon"] == "short"
    }
    # Identical inputs every bar, so the weekly means equal the bar values and
    # both paths must reach the same verdict.
    assert simulated == planned


def test_costs_are_charged_on_traded_notional(tmp_path) -> None:
    """Fees and slippage both apply, and only to what actually changed hands."""
    settings = Settings(data_root=tmp_path)
    closes = [100.0] * 6
    _write_prices(settings, closes, closes)
    _write_run(settings, [(0.95, 0.01)] * 6)

    free = _backtest(settings, end="2024-01-06", fee_rate=0.0, slippage_rate=0.0)
    charged = _backtest(settings, end="2024-01-06", fee_rate=0.001, slippage_rate=0.002)

    # Price never moves, so the only difference is transaction cost.
    assert free["roi_pct"] == pytest.approx(0.0, abs=1e-9)
    assert charged["roi_pct"] < 0.0
    # One entry at full exposure costs 0.3% of the notional.
    assert charged["roi_pct"] == pytest.approx(-0.3, abs=0.01)
    assert charged["total_trades"] == 1


def test_tiny_drift_does_not_trigger_a_rebalance(tmp_path) -> None:
    """Churning on noise would bleed fees for no reason."""
    settings = Settings(data_root=tmp_path)
    closes = [100.0, 100.5, 100.2, 100.7, 100.3, 100.6]
    _write_prices(settings, closes, closes)
    _write_run(settings, [(0.95, 0.01)] * 6)

    result = _backtest(settings, end="2024-01-06", fee_rate=0.001, slippage_rate=0.001)

    # One entry; the sub-1% drift afterwards stays under the rebalance floor.
    assert result["total_trades"] == 1


def test_buy_and_hold_benchmark_is_reported(tmp_path) -> None:
    """ROI without a benchmark cannot answer 'was this better than holding?'."""
    settings = Settings(data_root=tmp_path)
    closes = [100.0 + i * 5 for i in range(10)]
    _write_prices(settings, closes, closes)
    # Sell signals throughout: the strategy stays in cash while the asset rises.
    _write_run(settings, [(0.01, 0.95)] * 10)

    result = _backtest(settings, end="2024-01-10", fee_rate=0.0, slippage_rate=0.0)

    assert result["roi_pct"] == pytest.approx(0.0, abs=1e-9)
    assert result["benchmark"]["roi_pct"] > 0.0
    # Sitting out a rally is a loss relative to holding, and must read as one.
    assert result["benchmark"]["excess_roi_pct"] < 0.0
    assert result["time_in_market_pct"] == 0.0


def test_metrics_cover_risk_not_just_return(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    closes = [100.0, 120.0, 90.0, 130.0, 80.0, 140.0, 100.0, 150.0]
    _write_prices(settings, closes, closes)
    _write_run(settings, [(0.95, 0.01)] * 8)

    result = _backtest(settings, end="2024-01-08")

    assert result["max_drawdown_pct"] > 0.0
    assert result["sharpe"] is not None
    assert result["turnover_ratio"] > 0.0
    assert 0.0 <= result["time_in_market_pct"] <= 100.0


def test_untrained_rows_are_not_traded(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    closes = [100.0] * 6
    _write_prices(settings, closes, closes)

    run_dir = settings.reports_predictions_dir / SYMBOL / f"ml_1d_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(6):
        ts = start + timedelta(days=i)
        # Confident-looking numbers, but flagged untrained: must be ignored.
        horizon = {
            "P_long": 0.95,
            "P_flat": 0.04,
            "P_short": 0.01,
            "depth_long_bins": DEPTH,
            "depth_short_bins": DEPTH,
            "trained": False,
        }
        rows.append(
            json.dumps(
                {
                    "open_time": ts.isoformat(),
                    "date": ts.date().isoformat(),
                    "symbol": SYMBOL,
                    "short": horizon,
                    "medium": horizon,
                    "long": horizon,
                }
            )
        )
    (run_dir / "predictions.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = _backtest(settings, end="2024-01-06")

    assert "error" in result
    assert "untrained" in result["error"]
