"""Hash utilities for checksum verification."""

import hashlib
from pathlib import Path


def sha256_file(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path: Path to file
        chunk_size: Chunk size for reading file

    Returns:
        Hexadecimal SHA256 hash string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """
    Verify file checksum against expected value.

    Args:
        file_path: Path to file
        expected_checksum: Expected SHA256 hash (hex string)

    Returns:
        True if checksum matches, False otherwise
    """
    try:
        actual_checksum = sha256_file(file_path)
        return actual_checksum.lower() == expected_checksum.lower().strip()
    except (FileNotFoundError, ValueError):
        return False


def read_checksum_file(checksum_path: Path) -> str | None:
    """
    Read checksum from CHECKSUM file.

    CHECKSUM files from Binance contain the hash as first token.

    Args:
        checksum_path: Path to CHECKSUM file

    Returns:
        Checksum string or None if file doesn't exist or is invalid
    """
    if not checksum_path.exists():
        return None

    try:
        with open(checksum_path, encoding="utf-8") as f:
            line = f.readline().strip()
            # Binance CHECKSUM format: hash filename
            parts = line.split()
            if parts:
                return parts[0]
    except (OSError, ValueError):
        pass
    return None
