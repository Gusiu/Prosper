"""Integration test: predict CLI path with custom interval and versioned output."""

from datetime import UTC, datetime, timedelta

import polars as pl
from prosper.config import Settings
from prosper.features.build import build_features
from prosper.predict.ml import predict_ml
from prosper.storage.layout import get_parquet_file_path, parse_versioned_prediction_folder
from prosper.storage.parquet import save_parquet


def _daily_rows(start: datetime, days: int) -> pl.DataFrame:
    rows = []
    for i in range(days):
        ts = start + timedelta(days=i)
        close = 100.0 + i * 0.1
        rows.append(
            {
                "open_time": ts,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0 + i,
            }
        )
    return pl.DataFrame(rows)


def test_predict_ml_writes_interval_versioned_folder(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    symbol = "TESTUSDT"
    start = datetime(2023, 1, 1, tzinfo=UTC)

    # Enough history for walk-forward + long horizon labels
    save_parquet(
        _daily_rows(start, 450),
        get_parquet_file_path(symbol, "1d", 2023, month=1, settings=settings),
    )
    save_parquet(
        _daily_rows(start + timedelta(days=31), 120),
        get_parquet_file_path(symbol, "1d", 2023, month=2, settings=settings),
    )
    save_parquet(
        _daily_rows(start + timedelta(days=151), 120),
        get_parquet_file_path(symbol, "1d", 2023, month=6, settings=settings),
    )
    save_parquet(
        _daily_rows(start + timedelta(days=271), 120),
        get_parquet_file_path(symbol, "1d", 2023, month=10, settings=settings),
    )
    save_parquet(
        _daily_rows(datetime(2024, 1, 1, tzinfo=UTC), 60),
        get_parquet_file_path(symbol, "1d", 2024, month=1, settings=settings),
    )

    build_features(symbol, "1d", "2023-06-01", "2024-01-31", settings=settings)

    result = predict_ml(
        symbol=symbol,
        start="2023-12-01",
        end="2024-01-15",
        settings=settings,
        interval="1d",
        train_window_days=60,
    )

    assert "error" not in result
    assert result["predictions"] > 0

    pred_root = settings.reports_predictions_dir / symbol
    versioned_dirs = [
        d
        for d in pred_root.iterdir()
        if d.is_dir() and parse_versioned_prediction_folder(d.name)
    ]
    assert len(versioned_dirs) == 1

    parsed = parse_versioned_prediction_folder(versioned_dirs[0].name)
    assert parsed == ("ml", "1d", parsed[2])
    assert (versioned_dirs[0] / "predictions.jsonl").exists()
