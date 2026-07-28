from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import polars as pl

ALLOWED_RECOMMENDATIONS = {
    "Strong Buy",
    "Buy",
    "Accumulate",
    "Hold",
    "Reduce",
    "Sell",
    "Strong Sell",
}


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def smoke_check_parquet(path: Path, n: int = 5) -> None:
    if not path.exists():
        raise SystemExit(f"Missing Parquet: {path}")
    # Inline smoke check (avoid relying on scripts/smoke_check.py which prints large payloads)
    df = pl.read_parquet(path, hive_partitioning=False)
    cols = list(df.columns)
    print(f"Parquet ok: {path}")
    print(f"  rows={len(df)} cols={len(cols)}")
    if "open_time" in df.columns:
        df2 = df.with_columns(pl.col("open_time").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("open_time_str"))
        om = df2.select(pl.col("open_time_str").min().alias("min"), pl.col("open_time_str").max().alias("max")).to_dicts()[0]
        print(f"  open_time_min={om['min']} open_time_max={om['max']}")


def smoke_check_1w_any(root_dir: Path, symbol: str) -> None:
    base = root_dir / "processed" / "binance" / "spot" / "klines" / "1w" / f"symbol={symbol}"
    parts = sorted(base.rglob("part.parquet"))
    if not parts:
        raise SystemExit(f"No 1w parquet parts under: {base}")
    smoke_check_parquet(parts[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2020-01")  # YYYY-MM
    parser.add_argument("--end", default="2020-01")  # YYYY-MM
    parser.add_argument("--root", default="data_smoke")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.root)
    symbol = args.symbol

    # Unpack month boundaries (this script targets a single month smoke).
    start_year = int(args.start[:4])
    start_month = int(args.start[5:7])
    ym = f"{start_year:04d}-{start_month:02d}"

    run(
        [
            "prosper",
            "backfill",
            "--symbol",
            symbol,
            "--start",
            args.start,
            "--end",
            args.end,
            "--workers",
            str(args.workers),
            "--root",
            str(root),
        ]
    )

    zip_path = (
        root
        / "raw"
        / "binance"
        / "spot"
        / "klines"
        / "1m"
        / symbol
        / f"{symbol}-1m-{start_year:04d}-{start_month:02d}.zip"
    )
    cs_path = Path(str(zip_path) + ".CHECKSUM")
    for p in [zip_path, cs_path]:
        if not p.exists():
            raise SystemExit(f"Missing artifact: {p}")

    pq_1m = (
        root
        / "processed"
        / "binance"
        / "spot"
        / "klines"
        / "1m"
        / f"symbol={symbol}"
        / f"year={start_year:04d}"
        / f"month={start_month:02d}"
        / "part.parquet"
    )
    smoke_check_parquet(pq_1m)

    run(
        [
            "prosper",
            "qa",
            "check",
            "--symbol",
            symbol,
            "--interval",
            "1m",
            "--year",
            str(start_year),
            "--month",
            f"{start_month:02d}",
            "--root",
            str(root),
        ]
    )
    qa_report = root / "reports" / "qa" / symbol / "1m" / f"{start_year:04d}-{start_month:02d}.json"
    if not qa_report.exists():
        raise SystemExit(f"Missing QA report: {qa_report}")
    json.loads(qa_report.read_text(encoding="utf-8"))

    run(
        [
            "prosper",
            "aggregate",
            "--symbol",
            symbol,
            "--from",
            "1m",
            "--to",
            "1h",
            "1d",
            "1w",
            "--start",
            args.start,
            "--end",
            args.end,
            "--root",
            str(root),
        ]
    )

    pq_1h = (
        root
        / "processed"
        / "binance"
        / "spot"
        / "klines"
        / "1h"
        / f"symbol={symbol}"
        / f"year={start_year:04d}"
        / f"month={start_month:02d}"
        / "part.parquet"
    )
    smoke_check_parquet(pq_1h)

    pq_1d = (
        root
        / "processed"
        / "binance"
        / "spot"
        / "klines"
        / "1d"
        / f"symbol={symbol}"
        / f"year={start_year:04d}"
        / f"month={start_month:02d}"
        / "part.parquet"
    )
    smoke_check_parquet(pq_1d)

    smoke_check_1w_any(root, symbol)

    run(
        [
            "prosper",
            "features",
            "build",
            "--symbol",
            symbol,
            "--base_interval",
            "1d",
            "--start",
            f"{start_year:04d}-{start_month:02d}-01",
            "--end",
            f"{start_year:04d}-{start_month:02d}-31",
            "--root",
            str(root),
        ]
    )
    pq_feat = (
        root
        / "processed"
        / "binance"
        / "spot"
        / "features"
        / "1d"
        / f"symbol={symbol}"
        / "features.parquet"
    )
    smoke_check_parquet(pq_feat)

    run(
        [
            "prosper",
            "labels",
            "build",
            "--symbol",
            symbol,
            "--base_interval",
            "1d",
            "--forward_days",
            "1",
            "--root",
            str(root),
        ]
    )
    pq_labels = (
        root
        / "processed"
        / "binance"
        / "spot"
        / "labels"
        / "1d"
        / f"symbol={symbol}"
        / "labels.parquet"
    )
    smoke_check_parquet(pq_labels)

    # Baseline predictions
    run(
        [
            "prosper",
            "predict",
            "baseline",
            "--symbol",
            symbol,
            "--start",
            f"{start_year:04d}-{start_month:02d}-01",
            "--end",
            f"{start_year:04d}-{start_month:02d}-31",
            "--root",
            str(root),
        ]
    )
    # Predictions live in a versioned run folder identified by model/interval/timestamp.
    run_dirs = sorted(
        (root / "reports" / "predictions" / symbol).glob("baseline_*_*"),
        key=lambda p: p.name,
    )
    if not run_dirs:
        raise SystemExit(f"No baseline run folder under {root / 'reports' / 'predictions' / symbol}")
    run_dir = run_dirs[-1]
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        raise SystemExit(f"Missing predictions file: {pred_path}")
    lines = pred_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit("Predictions file is empty")
    first = json.loads(lines[0])
    s = first["short"]["P_long"] + first["short"]["P_flat"] + first["short"]["P_short"]
    if abs(s - 1.0) > 1e-6:
        raise SystemExit(f"Probability sum != 1.0 for first row: {s}")

    # Planner windows, scoped to the run we just produced
    model_type, interval, timestamp = run_dir.name.split("_")
    run(
        [
            "prosper",
            "planner",
            "windows",
            "--symbol",
            symbol,
            "--model-type",
            model_type,
            "--timestamp",
            timestamp,
            "--interval",
            interval,
            "--start",
            f"{start_year:04d}-{start_month:02d}-01",
            "--end",
            f"{start_year:04d}-{start_month:02d}-31",
            "--root",
            str(root),
        ]
    )
    win_path = (
        root
        / "reports"
        / "recommendations"
        / symbol
        / run_dir.name
        / f"{start_year:04d}-{start_month:02d}.json"
    )
    if not win_path.exists():
        raise SystemExit(f"Missing windows JSON: {win_path}")
    payload = json.loads(win_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != model_type:
        raise SystemExit(f"Windows report is not attributed to {model_type}: {payload}")
    for w in payload["windows"]:
        rec = w["recommendation"]
        if rec not in ALLOWED_RECOMMENDATIONS:
            raise SystemExit(f"Invalid recommendation {rec}")

    print(f"\nSmoke pipeline OK for {symbol} month {ym} (root={root})")


if __name__ == "__main__":
    main()

