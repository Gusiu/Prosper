from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from prosper.storage.parquet import save_parquet
from prosper.storage.layout import get_parquet_file_path
from prosper.config import get_settings


def generate_1m_klines(
    start: datetime,
    end_exclusive: datetime,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate a deterministic synthetic 1m OHLCV series.

    Intended only for offline smoke tests (pipeline wiring), not for realism.
    """

    rng = random.Random(seed)
    current = start
    n = int((end_exclusive - start).total_seconds() // 60)

    # Simple random walk around a base price.
    price = 20000.0

    open_times_ms: list[int] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []

    for i in range(n):
        t = current
        current = current + timedelta(minutes=1)

        # minute delta
        delta = (rng.random() - 0.5) * 0.002  # +/-0.1%
        o = price
        c = max(1e-6, price * (1.0 + delta))

        spread = abs(delta) * (0.5 + rng.random())
        hi = max(o, c) * (1.0 + spread)
        lo = min(o, c) * (1.0 - spread)
        lo = max(1e-8, lo)

        vol = 10.0 + rng.random() * 100.0

        open_times_ms.append(int(t.timestamp() * 1000))
        opens.append(float(o))
        closes.append(float(c))
        highs.append(float(hi))
        lows.append(float(lo))
        volumes.append(float(vol))

        price = c

    df = pl.DataFrame(
        {
            "open_time": pl.Series(open_times_ms, dtype=pl.Int64).cast(pl.Datetime("ms", time_zone="UTC")),
            "open": pl.Series(opens, dtype=pl.Float64),
            "high": pl.Series(highs, dtype=pl.Float64),
            "low": pl.Series(lows, dtype=pl.Float64),
            "close": pl.Series(closes, dtype=pl.Float64),
            "volume": pl.Series(volumes, dtype=pl.Float64),
        }
    )
    return df.sort("open_time")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data_smoke", help="Local data lake root (e.g., data_smoke)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--month", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings(data_root=Path(args.root))

    start = datetime(args.year, args.month, 1, tzinfo=timezone.utc)
    # end_exclusive = first day of next month
    if args.month == 12:
        end_exclusive = datetime(args.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end_exclusive = datetime(args.year, args.month + 1, 1, tzinfo=timezone.utc)

    df = generate_1m_klines(start=start, end_exclusive=end_exclusive)

    out_path = get_parquet_file_path(
        symbol=args.symbol,
        interval="1m",
        year=args.year,
        month=args.month,
        settings=settings,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(df, out_path)

    print(f"Synthetic 1m parquet written: {out_path}")
    print(f"Rows: {len(df)} (UTC minute bars)")


if __name__ == "__main__":
    main()

