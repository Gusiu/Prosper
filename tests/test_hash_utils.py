"""Tests for hash utilities."""

from pathlib import Path

import pytest
from prosper.utils.hash import read_checksum_file, sha256_file, verify_checksum


def test_sha256_file(tmp_path: Path) -> None:
    """Test SHA256 file hashing."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content", encoding="utf-8")

    hash_value = sha256_file(test_file)
    assert isinstance(hash_value, str)
    assert len(hash_value) == 64  # SHA256 hex string length


def test_sha256_file_not_found() -> None:
    """Test SHA256 hashing of non-existent file."""
    non_existent = Path("/nonexistent/file.txt")
    with pytest.raises(FileNotFoundError):
        sha256_file(non_existent)


def test_read_checksum_file(tmp_path: Path) -> None:
    """Test reading checksum from CHECKSUM file."""
    checksum_file = tmp_path / "test.zip.CHECKSUM"
    checksum_file.write_text("abc123def456 test.zip\n", encoding="utf-8")

    checksum = read_checksum_file(checksum_file)
    assert checksum == "abc123def456"


def test_read_checksum_file_not_found() -> None:
    """Test reading non-existent checksum file."""
    non_existent = Path("/nonexistent/file.CHECKSUM")
    assert read_checksum_file(non_existent) is None


def test_verify_checksum(tmp_path: Path) -> None:
    """Test checksum verification."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content", encoding="utf-8")

    hash_value = sha256_file(test_file)
    assert verify_checksum(test_file, hash_value) is True
    assert verify_checksum(test_file, "invalid_hash") is False
