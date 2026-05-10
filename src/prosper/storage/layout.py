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


def get_prediction_report_path(
    symbol: str, year: int, month: int, settings: Settings | None = None
) -> Path:
    """
    Get path for prediction report JSONL file.

    Args:
        symbol: Trading symbol
        year: Year
        month: Month (1-12)
        settings: Settings instance (defaults to global)

    Returns:
        Path to prediction report file
    """
    if settings is None:
        settings = get_settings()

    month_str = f"{month:02d}"
    filename = f"{year}-{month_str}.jsonl"
    return settings.reports_predictions_dir / symbol / "daily" / filename


def get_recommendation_report_path(
    symbol: str, year: int, month: int, settings: Settings | None = None
) -> Path:
    """
    Get path for recommendation report JSON file.

    Args:
        symbol: Trading symbol
        year: Year
        month: Month (1-12)
        settings: Settings instance (defaults to global)

    Returns:
        Path to recommendation report file
    """
    if settings is None:
        settings = get_settings()

    month_str = f"{month:02d}"
    filename = f"{year}-{month_str}.json"
    return settings.reports_recommendations_dir / symbol / "windows" / filename


def get_eval_walkforward_summary_path(
    symbol: str,
    settings: Settings | None = None,
) -> Path:
    """
    Get path for walk-forward evaluation summary JSON.

    Layout:
      data/reports/eval/{SYMBOL}/walkforward_summary.json
    """
    if settings is None:
        settings = get_settings()
    return settings.reports_eval_dir / symbol / "walkforward_summary.json"
