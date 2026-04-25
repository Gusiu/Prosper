"""Binance Public Data download utilities."""

from __future__ import annotations

import zipfile
import hashlib
from pathlib import Path
from typing import Iterable

import requests
from rich.console import Console

from prosper.config import Settings, get_settings
from prosper.utils.hash import read_checksum_file, verify_checksum
from prosper.utils.http import retry_with_backoff

console = Console()


def get_zip_url(symbol: str, year: int, month: int, base_url: str | None = None) -> str:
    """
    Generate URL for Binance monthly ZIP file.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        year: Year (e.g., 2024)
        month: Month (1-12)
        base_url: Base URL for Binance public data (defaults to config)

    Returns:
        URL string
    """
    if base_url is None:
        settings = get_settings()
        base_url = settings.binance_public_data_base

    month_str = f"{month:02d}"
    filename = f"{symbol}-1m-{year}-{month_str}.zip"
    return f"{base_url}/data/spot/monthly/klines/{symbol}/1m/{filename}"


def get_checksum_url(symbol: str, year: int, month: int, base_url: str | None = None) -> str:
    """
    Generate URL for Binance checksum file.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        year: Year (e.g., 2024)
        month: Month (1-12)
        base_url: Base URL for Binance public data (defaults to config)

    Returns:
        URL string
    """
    zip_url = get_zip_url(symbol, year, month, base_url)
    return f"{zip_url}.CHECKSUM"


def _iter_response_content(resp: requests.Response, chunk_size: int) -> Iterable[bytes]:
    for chunk in resp.iter_content(chunk_size=chunk_size):
        if chunk:
            yield chunk


def download_text(url: str, timeout: int) -> str:
    """Download text content with retry/backoff."""
    settings = get_settings()

    def _download() -> str:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text

    return retry_with_backoff(
        _download,
        max_retries=settings.download_retry_max,
        backoff_base=settings.download_retry_backoff,
    )


def download_to_file(url: str, output_path: Path, timeout: int, chunk_size: int = 1024 * 1024) -> None:
    """Download binary content to a file with retry/backoff."""
    settings = get_settings()

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    def _download() -> None:
        # Write to a temp file first so retries don't leave corrupted artifacts.
        sha256_hash = hashlib.sha256()
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in _iter_response_content(resp, chunk_size=chunk_size):
                    sha256_hash.update(chunk)
                    f.write(chunk)
        # Success: rename atomically
        tmp_path.replace(output_path)

    retry_with_backoff(
        _download,
        max_retries=settings.download_retry_max,
        backoff_base=settings.download_retry_backoff,
    )


def download_monthly_zip(
    symbol: str,
    year: int,
    month: int,
    output_dir: Path,
    verify: bool = True,
    dry_run: bool = False,
) -> tuple[Path, Path] | None:
    """
    Download monthly ZIP and checksum files from Binance Public Data.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        year: Year (e.g., 2024)
        month: Month (1-12)
        output_dir: Directory to save files
        verify: Whether to verify checksum
        dry_run: If True, only print what would be downloaded

    Returns:
        Tuple of (zip_path, checksum_path) or None if dry_run

    Raises:
        httpx.HTTPError: If download fails
        ValueError: If checksum verification fails
    """
    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    month_str = f"{month:02d}"
    zip_filename = f"{symbol}-1m-{year}-{month_str}.zip"
    zip_path = output_dir / zip_filename
    checksum_path = output_dir / f"{zip_filename}.CHECKSUM"

    # Skip if already exists and verified
    if zip_path.exists() and checksum_path.exists():
        if verify:
            expected_checksum = read_checksum_file(checksum_path)
            if expected_checksum and verify_checksum(zip_path, expected_checksum):
                console.print(f"[green][OK][/green] Already exists and verified: {zip_path.name}")
                return (zip_path, checksum_path)
        else:
            console.print(f"[green][OK][/green] Already exists: {zip_path.name}")
            return (zip_path, checksum_path)

    zip_url = get_zip_url(symbol, year, month)
    checksum_url = get_checksum_url(symbol, year, month)

    if dry_run:
        console.print(f"[yellow]DRY RUN[/yellow] Would download: {zip_url}")
        console.print(f"[yellow]DRY RUN[/yellow] Would download: {checksum_url}")
        return None

    timeout = settings.download_timeout
    console.print(f"Downloading checksum: {checksum_path.name}")
    checksum_content = download_text(checksum_url, timeout=timeout)
    console.print(f"Downloading ZIP: {zip_path.name}")
    download_to_file(zip_url, zip_path, timeout=timeout)

    checksum_path.write_text(checksum_content, encoding="utf-8")

    # Verify checksum
    if verify:
        expected_checksum = read_checksum_file(checksum_path)
        if expected_checksum:
            if not verify_checksum(zip_path, expected_checksum):
                zip_path.unlink()
                checksum_path.unlink()
                raise ValueError(f"Checksum verification failed for {zip_path.name}")
            console.print(f"[green][OK][/green] Verified: {zip_path.name}")
        else:
            console.print(f"[yellow]⚠[/yellow] Could not read checksum from {checksum_path.name}")

    return (zip_path, checksum_path)


def extract_csv_from_zip(zip_path: Path, symbol: str) -> list[dict[str, float | int]]:
    """
    Extract and parse CSV from Binance monthly ZIP file.

    Binance CSV format (1m klines):
    open_time,open,high,low,close,volume,close_time,quote_asset_volume,num_trades,
    taker_buy_base_volume,taker_buy_quote_volume,ignore

    Args:
        zip_path: Path to ZIP file
        symbol: Trading symbol (for filename matching)

    Returns:
        List of dictionaries with parsed candle data

    Raises:
        zipfile.BadZipFile: If ZIP is corrupted
        ValueError: If CSV format is invalid
    """
    from prosper.utils.time import normalize_timestamp_ms

    candles: list[dict[str, float | int]] = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        # Find CSV file in ZIP (should match symbol)
        csv_files = [f for f in zip_file.namelist() if f.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV file found in ZIP: {zip_path}")

        csv_filename = csv_files[0]
        with zip_file.open(csv_filename) as csv_file:
            for line_bytes in csv_file:
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 6:
                    continue  # Skip invalid lines

                try:
                    # Parse fields
                    open_time_raw = int(float(parts[0]))
                    open_time_ms = normalize_timestamp_ms(open_time_raw)
                    open_price = float(parts[1])
                    high_price = float(parts[2])
                    low_price = float(parts[3])
                    close_price = float(parts[4])
                    volume = float(parts[5])

                    candle: dict[str, float | int] = {
                        "open_time": open_time_ms,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                    }

                    # Optional fields
                    if len(parts) > 6:
                        close_time_raw = int(float(parts[6]))
                        candle["close_time"] = normalize_timestamp_ms(close_time_raw)
                    if len(parts) > 7:
                        candle["quote_asset_volume"] = float(parts[7])
                    if len(parts) > 8:
                        candle["num_trades"] = int(parts[8])
                    if len(parts) > 9:
                        candle["taker_buy_base_volume"] = float(parts[9])
                    if len(parts) > 10:
                        candle["taker_buy_quote_volume"] = float(parts[10])

                    candles.append(candle)
                except (ValueError, IndexError) as e:
                    console.print(f"[yellow]⚠[/yellow] Skipping invalid line: {line[:50]}... ({e})")
                    continue

    return candles
