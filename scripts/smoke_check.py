from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def _safe_head_tail(df: pl.DataFrame, n: int) -> dict[str, list[dict]]:
    cols = [c for c in ["open_time", "open", "high", "low", "close", "volume"] if c in df.columns]
    if not cols:
        # Fallback: just return first/last rows as dicts
        return {"head": df.head(n).to_dicts(), "tail": df.tail(n).to_dicts()}

    df2 = df.select(cols)
    if "open_time" in cols and df2["open_time"].dtype == pl.Datetime:
        df2 = df2.with_columns(pl.col("open_time").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("open_time"))

    head = df2.head(n).to_dicts()
    tail = df2.tail(n).to_dicts()
    return {"head": head, "tail": tail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to Parquet file or directory")
    parser.add_argument("--n", type=int, default=5, help="Head/tail rows to print")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"Path does not exist: {target}")

    suffix = target.suffix.lower()
    if suffix in {".json"}:
        payload = json.loads(target.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "path": str(target),
                    "keys": list(payload.keys()) if isinstance(payload, dict) else None,
                    "type": type(payload).__name__,
                },
                sort_keys=True,
                ensure_ascii=True,
            )
        )
        return

    if suffix in {".jsonl"}:
        lines = target.read_text(encoding="utf-8").splitlines()
        parsed = json.loads(lines[0]) if lines else None
        print(
            json.dumps(
                {
                    "path": str(target),
                    "lines": len(lines),
                    "first_keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                },
                sort_keys=True,
                ensure_ascii=True,
            )
        )
        return

    # Default: treat as Parquet file/directory.
    parquet_path = target
    # Disable hive partitioning to avoid "duplicate fields" when the parquet
    # already contains columns like `symbol=` derived partition keys.
    df = pl.read_parquet(parquet_path, hive_partitioning=False)

    info: dict[str, object] = {
        "path": str(parquet_path),
        "rows": int(len(df)),
        "schema": {k: str(v) for k, v in df.schema.items()},
    }

    if "open_time" in df.columns:
        # Convert to ISO string inside Polars to avoid datetime->Python conversion
        # (which can panic with timezone-aware Datetime in some Polars builds).
        df_str = df.with_columns(
            pl.col("open_time").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("_open_time_str")
        )
        min_s = df_str.select(pl.col("_open_time_str").min().alias("min")).item()
        max_s = df_str.select(pl.col("_open_time_str").max().alias("max")).item()
        info["open_time_min"] = str(min_s)
        info["open_time_max"] = str(max_s)

    info["head_tail"] = _safe_head_tail(df, args.n)

    print(json.dumps(info, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()

