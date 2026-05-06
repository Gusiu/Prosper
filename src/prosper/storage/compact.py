"""Utility for compacting/merging Parquet files into larger partitions."""
import polars as pl
from pathlib import Path

def compact_parquet_files(parquet_dir: Path, out_path: Path, pattern: str = "*.parquet"):
    files = list(parquet_dir.glob(pattern))
    if not files:
        raise FileNotFoundError("No Parquet files found to compact.")
    dfs = [pl.read_parquet(f) for f in files]
    df = pl.concat(dfs)
    df.write_parquet(out_path)
    return out_path
