"""Storage layout utilities for organizing data files."""

from pathlib import Path

from prosper.config import Settings, get_settings


def get_processed_parquet_path(
    symbol: str,
    interval: str,
    year: int,
    month: int | None = None,
    week: int | None = None,
    settings: Settings | None = None,
) -> Path:
    """
    Get path for processed Parquet file.

    Args:
        symbol: Trading symbol
        interval: Time interval (1m, 1h, 1d, 1w)
        year: Year
        month: Month (1-12) - required for 1m, 1h, 1d
        week: Week number - required for 1w
        settings: Settings instance (defaults to global)

    Returns:
        Path to Parquet file directory (part.parquet will be inside)
    """
    if settings is None:
        settings = get_settings()

    base_dir = settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"

    if interval == "1w":
        if week is None:
            raise ValueError("Week number required for 1w interval")
        return base_dir / f"year={year}" / f"week={week:02d}"
    else:
        if month is None:
            raise ValueError("Month required for intervals other than 1w")
        return base_dir / f"year={year}" / f"month={month:02d}"


def get_parquet_file_path(
    symbol: str,
    interval: str,
    year: int,
    month: int | None = None,
    week: int | None = None,
    settings: Settings | None = None,
) -> Path:
    """
    Get full path to Parquet file (part.parquet).

    Args:
        symbol: Trading symbol
        interval: Time interval (1m, 1h, 1d, 1w)
        year: Year
        month: Month (1-12) - required for 1m, 1h, 1d
        week: Week number - required for 1w
        settings: Settings instance (defaults to global)

    Returns:
        Path to Parquet file
    """
    dir_path = get_processed_parquet_path(symbol, interval, year, month, week, settings)
    return dir_path / "part.parquet"


def get_labels_parquet_path(symbol: str, interval: str, settings: Settings | None = None) -> Path:
    """
    Get path for labels Parquet file.

    Args:
        symbol: Trading symbol
        interval: Base interval for labels (e.g., 1d)
        settings: Settings instance (defaults to global)

    Returns:
        Path to labels Parquet file
    """
    if settings is None:
        settings = get_settings()

    return (
        settings.processed_binance_spot_labels_dir
        / interval
        / f"symbol={symbol}"
        / "labels.parquet"
    )


def get_features_parquet_path(
    symbol: str,
    interval: str,
    settings: Settings | None = None,
) -> Path:
    """
    Get path for daily/intraday engineered features parquet.

    Layout:
      data/processed/binance/spot/features/{interval}/symbol={SYMBOL}/features.parquet
    """
    if settings is None:
        settings = get_settings()

    # Reuse the base processed directory but keep a separate features root.
    features_root = settings.processed_data_dir / "binance" / "spot" / "features"
    return features_root / interval / f"symbol={symbol}" / "features.parquet"


def get_qa_report_path(
    symbol: str,
    interval: str,
    year: int,
    month: int | None = None,
    settings: Settings | None = None,
) -> Path:
    """
    Get path for QA report JSON file.

    Args:
        symbol: Trading symbol
        interval: Time interval (1m, 1h, 1d)
        year: Year
        month: Month (1-12)
        settings: Settings instance (defaults to global)

    Returns:
        Path to QA report file
    """
    if settings is None:
        settings = get_settings()

    month_str = f"{month:02d}" if month else "all"
    filename = f"{year}-{month_str}.json"
    return settings.reports_qa_dir / symbol / interval / filename


def get_recommendation_report_path(
    symbol: str,
    year: int,
    month: int,
    settings: Settings | None = None,
    run_slug: str | None = None,
) -> Path:
    """
    Get path for recommendation report JSON file.

    Layout:
      data/reports/recommendations/{SYMBOL}/{run_slug}/{YYYY-MM}.json

    ``run_slug`` identifies the prediction run the windows were derived from.
    Without it the file would not say which model produced the recommendation.

    Args:
        symbol: Trading symbol
        year: Year
        month: Month (1-12)
        settings: Settings instance (defaults to global)
        run_slug: Prediction run identity, e.g. ``ml_1d_20240501123000``

    Returns:
        Path to recommendation report file
    """
    if settings is None:
        settings = get_settings()

    month_str = f"{month:02d}"
    filename = f"{year}-{month_str}.json"
    folder = run_slug or "windows"
    return settings.reports_recommendations_dir / symbol / folder / filename


def get_eval_walkforward_summary_path(
    symbol: str,
    settings: Settings | None = None,
    run_slug: str | None = None,
) -> Path:
    """
    Get path for walk-forward evaluation summary JSON.

    Layout:
      data/reports/eval/{SYMBOL}/{run_slug}/walkforward_summary.json
    """
    if settings is None:
        settings = get_settings()
    base = settings.reports_eval_dir / symbol
    if run_slug:
        base = base / run_slug
    return base / "walkforward_summary.json"


def get_backtest_report_path(
    symbol: str,
    settings: Settings | None = None,
    run_slug: str | None = None,
) -> Path:
    """
    Get path for a trading backtest report JSON.

    Layout:
      data/reports/backtests/{SYMBOL}/{run_slug}/backtest.json
    """
    if settings is None:
        settings = get_settings()
    base = settings.reports_dir / "backtests" / symbol
    if run_slug:
        base = base / run_slug
    return base / "backtest.json"


def get_evaluation_run_dir(
    symbol: str,
    model_type: str,
    timestamp: str,
    settings: Settings | None = None,
    interval: str | None = "1d",
) -> Path:
    """
    Get directory for a versioned prediction-evaluation run.

    Layout:
      data/reports/evaluations/{SYMBOL}/{model_type}_{interval}_{timestamp}/
    """
    if settings is None:
        settings = get_settings()
    folder = versioned_prediction_folder_name(model_type, timestamp, interval)
    return settings.reports_dir / "evaluations" / symbol / folder


def parse_evaluation_folder(name: str) -> tuple[str, str, str] | None:
    """Parse evaluation folder into ``(model_type, interval, timestamp)``."""
    parsed = parse_versioned_prediction_folder(name)
    if parsed:
        return parsed
    if "_" not in name:
        return None
    parts = name.split("_")
    timestamp = parts[-1]
    if not timestamp.isdigit() or len(timestamp) < 12:
        return None
    model_type = "_".join(parts[:-1])
    if not model_type:
        return None
    return model_type, "1d", timestamp


def resolve_evaluation_run_dir(
    symbol: str,
    model_type: str,
    timestamp: str,
    settings: Settings | None = None,
    interval: str | None = "1d",
) -> Path:
    """Resolve an existing evaluation directory (interval-aware or legacy)."""
    if settings is None:
        settings = get_settings()
    candidates: list[Path] = []
    if interval:
        candidates.append(get_evaluation_run_dir(symbol, model_type, timestamp, settings, interval))
    candidates.append(get_evaluation_run_dir(symbol, model_type, timestamp, settings, None))
    if interval:
        legacy = settings.reports_dir / "evaluations" / symbol / f"{model_type}_{timestamp}"
        candidates.append(legacy)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def get_evaluation_metrics_path(
    symbol: str,
    model_type: str,
    timestamp: str,
    settings: Settings | None = None,
    interval: str | None = "1d",
) -> Path:
    """Return path to ``metrics.json`` for a versioned evaluation run."""
    return resolve_evaluation_run_dir(symbol, model_type, timestamp, settings, interval) / "metrics.json"


KNOWN_PREDICTION_INTERVALS = frozenset({"1m", "1h", "1d", "1w"})


def versioned_prediction_folder_name(
    model_type: str, timestamp: str, interval: str | None = None
) -> str:
    """Build folder name: ``{model_type}_{interval}_{timestamp}`` or legacy ``{model_type}_{timestamp}``."""
    if interval:
        return f"{model_type}_{interval}_{timestamp}"
    return f"{model_type}_{timestamp}"


def parse_versioned_prediction_folder(name: str) -> tuple[str, str, str] | None:
    """Parse versioned folder name into ``(model_type, interval, timestamp)``.

    Supports:
    - ``{model_type}_{interval}_{timestamp}`` e.g. ``ml_1d_20240501123000``
    - legacy ``{model_type}_{timestamp}`` e.g. ``xgboost_20240501123000``
    - legacy ``{model_type}_{interval}_{timestamp}`` with interval embedded in model_type
      e.g. folder created via ``ml_1d`` model_type → still resolves to model ``ml``, interval ``1d``
    """
    if "_" not in name:
        return None
    parts = name.split("_")
    if not parts[-1].isdigit() or len(parts[-1]) < 12:
        return None

    timestamp = parts[-1]
    if len(parts) >= 3 and parts[-2] in KNOWN_PREDICTION_INTERVALS:
        model_type = "_".join(parts[:-2])
        interval = parts[-2]
        # Normalize folders like ``ml_1d_2024...`` where model_type was passed as ``ml_1d``
        if model_type.endswith(f"_{interval}"):
            base = model_type[: -(len(interval) + 1)]
            if base:
                model_type = base
        return model_type, interval, timestamp

    model_type = parts[0]
    legacy_timestamp = "_".join(parts[1:])
    return model_type, "1d", legacy_timestamp


def get_versioned_prediction_dir(
    symbol: str,
    model_type: str,
    timestamp: str,
    settings: Settings | None = None,
    interval: str | None = None,
) -> Path:
    """
    Get directory for a versioned prediction run.

    Layout example:
      data/reports/predictions/{symbol}/{model_type}_{interval}_{timestamp}/
    """
    if settings is None:
        settings = get_settings()
    folder = versioned_prediction_folder_name(model_type, timestamp, interval)
    return settings.reports_predictions_dir / symbol / folder


def resolve_versioned_prediction_dir(
    symbol: str,
    model_type: str,
    timestamp: str,
    settings: Settings | None = None,
    interval: str | None = None,
) -> Path:
    """Resolve an existing versioned prediction directory (new or legacy layout)."""
    if settings is None:
        settings = get_settings()
    candidates: list[Path] = []
    if interval:
        candidates.append(get_versioned_prediction_dir(symbol, model_type, timestamp, settings, interval))
    candidates.append(get_versioned_prediction_dir(symbol, model_type, timestamp, settings, None))
    if interval:
        candidates.append(
            get_versioned_prediction_dir(symbol, f"{model_type}_{interval}", timestamp, settings, None)
        )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def get_versioned_prediction_file(
    symbol: str,
    model_type: str,
    timestamp: str,
    filename: str = "predictions.jsonl",
    settings: Settings | None = None,
    interval: str | None = None,
) -> Path:
    """Return path to a file inside a versioned prediction directory."""
    return (
        resolve_versioned_prediction_dir(symbol, model_type, timestamp, settings, interval) / filename
    )
