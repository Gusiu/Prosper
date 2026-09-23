"""Nothing enters the lake unless its checksum was read and matched.

This is where data comes into the project, and the module had 15% test
coverage. It also accepted an archive whose CHECKSUM file could not be parsed,
with a warning: a truncated download paired with an error page in place of the
checksum went into the lake as if it had been verified.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from prosper.binance import public_data

ARCHIVE = b"PK-fake-zip-bytes" * 64
GOOD_SUM = hashlib.sha256(ARCHIVE).hexdigest()


@pytest.fixture
def network(monkeypatch):
    """Stands in for requests: records what was fetched, serves what it is told."""
    served = {"checksum": f"{GOOD_SUM}  BTCUSDT-1m-2024-01.zip\n", "archive": ARCHIVE}
    fetched: list[str] = []

    def download_text(url: str, timeout: int) -> str:
        fetched.append(url)
        return served["checksum"]

    def download_to_file(url: str, output_path: Path, timeout: int, chunk_size: int = 0) -> None:
        fetched.append(url)
        output_path.write_bytes(served["archive"])

    monkeypatch.setattr(public_data, "download_text", download_text)
    monkeypatch.setattr(public_data, "download_to_file", download_to_file)
    return served, fetched


def _download(tmp_path: Path) -> tuple[Path, Path] | None:
    return public_data.download_monthly_zip("BTCUSDT", 2024, 1, tmp_path, verify=True)


def test_a_matching_archive_is_kept(tmp_path, network) -> None:
    result = _download(tmp_path)

    assert result is not None
    zip_path, checksum_path = result
    assert zip_path.read_bytes() == ARCHIVE and checksum_path.exists()


def test_a_mismatched_archive_is_removed_and_refused(tmp_path, network) -> None:
    served, _ = network
    served["archive"] = ARCHIVE[:-10]  # truncated in transit

    with pytest.raises(ValueError, match="mismatch"):
        _download(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_unreadable_checksum_is_a_failed_check_not_a_skipped_one(tmp_path, network) -> None:
    served, _ = network
    served["checksum"] = ""  # what an empty or error response parses to

    with pytest.raises(ValueError, match="unreadable"):
        _download(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_verified_archive_on_disk_is_not_fetched_again(tmp_path, network) -> None:
    _, fetched = network
    _download(tmp_path)
    fetched.clear()

    assert _download(tmp_path) is not None
    assert fetched == []


def test_a_corrupted_archive_on_disk_is_fetched_again(tmp_path, network) -> None:
    _, fetched = network
    zip_path, _ = _download(tmp_path)
    zip_path.write_bytes(b"bit rot")
    fetched.clear()

    assert _download(tmp_path) is not None
    assert zip_path.read_bytes() == ARCHIVE
    assert len(fetched) == 2  # checksum and archive


def test_a_dry_run_fetches_nothing(tmp_path, network) -> None:
    _, fetched = network
    assert public_data.download_monthly_zip("BTCUSDT", 2024, 1, tmp_path, dry_run=True) is None
    assert fetched == []
