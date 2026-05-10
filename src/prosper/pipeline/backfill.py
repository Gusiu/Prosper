"""Backfill pipeline for downloading and processing monthly Binance data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.console import Console

from prosper.binance.public_data import download_monthly_zip, extract_csv_from_zip
from prosper.config import Settings, get_settings
from prosper.storage.parquet import candles_to_dataframe, check_parquet_exists, save_parquet
from prosper.utils.time import generate_month_range

console = Console()


def process_month(
    symbol: str,
    year: int,
    month: int,
    settings: Settings,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> tuple[int, int, bool, str]:
    """
    Process a single month: download, extract, and save to Parquet.

    Args:
        symbol: Trading symbol
        year: Year
        month: Month (1-12)
        settings: Settings instance
        dry_run: If True, only simulate
        skip_existing: If True, skip if Parquet already exists

    Returns:
        Tuple of (year, month, success, message)
    """
    try:
        # Check if already processed
        if skip_existing and check_parquet_exists(symbol, "1m", year, month, settings=settings):
            return (year, month, True, "Already exists")

        if dry_run:
            return (year, month, True, "Dry run - skipped")

        # Download ZIP
        zip_path, checksum_path = download_monthly_zip(
            symbol,
            year,
            month,
            settings.raw_binance_spot_klines_1m_dir / symbol,
            verify=True,
            dry_run=False,
        )

        # Extract CSV
        candles = extract_csv_from_zip(zip_path, symbol)

        if not candles:
            return (year, month, False, "No candles extracted")

        # Convert to DataFrame
        df = candles_to_dataframe(candles)

        if df.is_empty():
            return (year, month, False, "Empty DataFrame")

        # Save to Parquet
        from prosper.storage.layout import get_parquet_file_path

        parquet_path = get_parquet_file_path(symbol, "1m", year, month, settings=settings)
        save_parquet(df, parquet_path)

        return (year, month, True, f"Processed {len(df)} candles")

    except Exception as e:
        return (year, month, False, f"Error: {str(e)}")


def backfill(
    symbol: str,
    start: str,
    end: str,
    workers: int = 4,
    dry_run: bool = False,
    skip_existing: bool = True,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Backfill monthly data for a symbol.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        start: Start date in YYYY-MM format
        end: End date in YYYY-MM format
        workers: Number of parallel workers
        dry_run: If True, only simulate
        skip_existing: If True, skip months that already have Parquet files
        settings: Settings instance (defaults to global)

    Returns:
        Dictionary with summary statistics
    """
    if settings is None:
        settings = get_settings()

    months = generate_month_range(start, end)
    console.print(f"[cyan]Backfilling {symbol} from {start} to {end} ({len(months)} months)[/cyan]")

    results: list[tuple[int, int, bool, str]] = []
    failed: list[tuple[int, int]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_month, symbol, year, month, settings, dry_run, skip_existing): (
                year,
                month,
            )
            for year, month in months
        }

        total_tasks = len(futures)
        completed = 0

        for future in as_completed(futures):
            completed += 1
            year, month, success, message = future.result()
            results.append((year, month, success, message))

            pct = (completed / total_tasks) * 100

            if success:
                console.print(f"[{pct:5.1f}%] [green][OK][/green] {year}-{month:02d}: {message}")
            else:
                console.print(f"[{pct:5.1f}%] [red][FAIL][/red] {year}-{month:02d}: {message}")
                failed.append((year, month))

    summary = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "total_months": len(months),
        "processed": len([r for r in results if r[2]]),
        "failed": len(failed),
        "failed_months": failed,
    }

    console.print(
        f"\n[bold]Summary:[/bold] {summary['processed']}/{summary['total_months']} months processed"
    )
    if failed:
        console.print(f"[red]Failed months: {failed}[/red]")

    return summary
