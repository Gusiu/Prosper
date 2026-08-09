"""A model may only train on features that mean the same thing on every symbol.

The predictors took every column in the features parquet bar a handful of names,
which fed them raw price levels. Measured across BTCUSDT and SOLUSDT over the
same period, `ma_10` differed by 553x, `atr_14` by 360x and `macd_hist` by 387x —
that last one is the interesting case, because nothing in its name says "price
units" and reading the names alone would have missed it.

On a single symbol this is harmless: the normaliser removes the level and the
model sees a shape. Pooled across symbols it becomes a symbol label, and robust
statistics over a joint window make the feature bimodal instead of comparable.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from prosper.config import Settings
from prosper.domain import TRAINING_FEATURE_COLUMNS, training_features
from prosper.features.build import build_features
from prosper.storage.layout import get_features_parquet_path, get_parquet_file_path
from prosper.storage.parquet import save_parquet

# Anything whose value scales with the price or the traded amount.
ABSOLUTE = {
    "open", "high", "low", "close", "volume", "quote_asset_volume",
    "taker_buy_base_volume", "taker_buy_quote_volume", "num_trades",
    "ma_10", "ma_30", "ma_diff", "ma_slope", "bb_mid", "bb_upper", "bb_lower",
    "atr_14", "obv", "macd_hist", "avg_trade_size",
}


def _write_klines(settings: Settings, symbol: str, multiplier: float, days: int = 400) -> None:
    """One price path, optionally scaled. The *shape* is identical either way."""
    start = datetime(2023, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(days):
        # Deterministic, non-monotonic, and never zero.
        base = 100.0 * (1.0 + 0.4 * math.sin(i / 23.0) + 0.15 * math.sin(i / 5.0))
        close = base * multiplier
        rows.append(
            {
                "open_time": start + timedelta(days=i),
                "open": close * 0.995,
                "high": close * 1.02,
                "low": close * 0.97,
                "close": close,
                "volume": 1000.0 * multiplier * (1.0 + 0.3 * math.sin(i / 11.0)),
                "quote_asset_volume": 1000.0 * close,
                "num_trades": 5000 + (i % 97),
                "taker_buy_base_volume": 500.0 * multiplier,
                "taker_buy_quote_volume": 500.0 * close,
            }
        )
    df = pl.DataFrame(rows).with_columns(pl.col("open_time").dt.month().alias("_m"),
                                         pl.col("open_time").dt.year().alias("_y"))
    for (year, month), group in df.group_by(["_y", "_m"]):
        save_parquet(
            group.drop(["_y", "_m"]).sort("open_time"),
            get_parquet_file_path(symbol, "1d", int(year), month=int(month), settings=settings),
        )


def _features(tmp_path, symbol: str, multiplier: float) -> pl.DataFrame:
    settings = Settings(data_root=tmp_path / symbol)
    _write_klines(settings, symbol, multiplier)
    build_features(symbol, "1d", "2023-01-01", "2024-02-04", settings=settings)
    return pl.read_parquet(
        get_features_parquet_path(symbol, "1d", settings=settings), hive_partitioning=False
    )


def test_the_training_set_is_invariant_to_the_price_level(tmp_path) -> None:
    """The property the whole change exists for: two identically shaped series
    a thousand times apart in price must produce identical training features."""
    cheap = _features(tmp_path, "CHEAPUSDT", 1.0)
    dear = _features(tmp_path, "DEARUSDT", 1000.0)

    columns = training_features(cheap.columns)
    assert columns, "no training columns found"

    for column in columns:
        a = cheap[column].to_list()
        b = dear[column].to_list()
        for i, (left, right) in enumerate(zip(a, b, strict=True)):
            if left is None or right is None:
                assert left is None and right is None, f"{column} row {i}: null mismatch"
                continue
            assert left == pytest.approx(right, rel=1e-6, abs=1e-9), (
                f"{column} row {i} differs with the price level: {left} vs {right}"
            )


def test_the_absolute_columns_really_do_differ(tmp_path) -> None:
    """Guards the test above from passing vacuously: if scaling the series
    changed nothing at all, the invariance check would prove nothing."""
    cheap = _features(tmp_path, "CHEAPUSDT", 1.0)
    dear = _features(tmp_path, "DEARUSDT", 1000.0)

    moved = [
        column
        for column in ABSOLUTE
        if column in cheap.columns
        and (cheap[column].abs().median() or 0) > 0
        and abs(dear[column].abs().median() / cheap[column].abs().median() - 1.0) > 0.5
    ]
    assert len(moved) >= 8, f"only {moved} moved; the fixture is not exercising scale"


def test_no_absolute_column_is_in_the_training_set() -> None:
    assert not set(TRAINING_FEATURE_COLUMNS) & ABSOLUTE


def test_every_predictor_uses_the_include_list() -> None:
    """An exclude list adopts every new column, which is how the relative
    columns would have been trained on *alongside* their absolute twins."""
    import inspect

    from prosper.predict import gru, ml, tft, xgboost_model

    for module in (ml, xgboost_model, gru, tft):
        source = inspect.getsource(module)
        assert "training_features(" in source, module.__name__
        assert "exclude_cols" not in source, f"{module.__name__} still excludes rather than includes"


def test_the_order_is_stable() -> None:
    """A model stored under one column order cannot be scored under another, so
    this must not depend on set iteration."""
    available = list(reversed(TRAINING_FEATURE_COLUMNS)) + ["close", "open_time"]
    assert training_features(available) == list(TRAINING_FEATURE_COLUMNS)


def test_a_narrower_dataset_is_allowed(tmp_path) -> None:
    """Order-flow features need kline columns older data lacks, and the intraday
    ones only exist at 1d, so a shorter list is legitimate rather than an error."""
    assert training_features(["log_return_1b", "rsi_14"]) == ["log_return_1b", "rsi_14"]
    assert training_features([]) == []


def test_the_duplicate_trade_count_column_is_gone(tmp_path) -> None:
    """`trade_count` was a float cast of `num_trades` kept as working state for
    `trade_count_rel_30`, and saving it stored the same quantity twice."""
    frame = _features(tmp_path, "CHEAPUSDT", 1.0)
    assert "trade_count" not in frame.columns
    assert "_trade_count" not in frame.columns
    assert "trade_count_rel_30" in frame.columns
