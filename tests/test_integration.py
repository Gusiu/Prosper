"""Integration-style tests (no real network, use mocks or dry-run)."""


from prosper.pipeline.backfill import process_month
from prosper.storage.layout import get_parquet_file_path
from prosper.utils.time import generate_month_range


def test_generate_month_range() -> None:
    """Test month range generation."""
    months = generate_month_range("2024-01", "2024-03")
    assert months == [(2024, 1), (2024, 2), (2024, 3)]

    months_single = generate_month_range("2023-12", "2023-12")
    assert months_single == [(2023, 12)]


def test_process_month_dry_run() -> None:
    """Test process_month in dry-run does not download."""
    from prosper.config import Settings

    # Use default settings (creates ./data)
    settings = Settings()
    year, month, success, message = process_month(
        "BTCUSDT", 2024, 1, settings, dry_run=True, skip_existing=False
    )
    assert success is True
    assert "Dry run" in message or "skipped" in message.lower()


def test_storage_layout_paths() -> None:
    """Test that layout paths are consistent."""


    parquet_path = get_parquet_file_path("BTCUSDT", "1m", 2024, 1)
    assert "symbol=BTCUSDT" in str(parquet_path)
    assert "year=2024" in str(parquet_path)
    assert "month=01" in str(parquet_path)
    assert parquet_path.name == "part.parquet"
