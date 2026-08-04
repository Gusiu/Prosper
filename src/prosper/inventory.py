"""Core data management and inventory logic for Prosper."""

from __future__ import annotations

import calendar
import json
import math
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_features_parquet_path, get_labels_parquet_path

INTERVAL_ORDER = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
]
INTERVAL_TO_MILLISECONDS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
KLINE_SIDE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "symbol",
    "year",
    "month",
    "week",
}
LABEL_COLUMNS = ["return_fwd", "direction", "depth_bin"]


@dataclass(frozen=True)
class InventoryQuality:
    status: str
    label: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "label": self.label, "message": self.message}


class DataManager:
    """Reusable data-lake inventory and maintenance service."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.cache_path = self.settings.meta_dir / "inventory_cache.json"

    def get_inventory(self, refresh: bool = False) -> list[dict[str, Any]]:
        fingerprint = self._fingerprint()
        if not refresh:
            cached = self._read_cache(fingerprint)
            if cached is not None:
                return cached

        inventory = self._scan_inventory()
        self._write_cache(fingerprint, inventory)
        return inventory

    def delete_symbol_data(self, symbol: str) -> int:
        paths = self._symbol_paths(symbol)
        allowed_roots = [
            self.settings.raw_data_dir,
            self.settings.processed_data_dir,
            self.settings.reports_dir,
        ]

        deleted_count = 0
        for path in paths:
            try:
                resolved = path.resolve()
                if not any(
                    resolved.is_relative_to(root.resolve())
                    for root in allowed_roots
                    if root is not None
                ):
                    continue
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    deleted_count += 1
            except OSError:
                continue

        self._invalidate_cache()
        return deleted_count

    def get_chart_data(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Return joined OHLCV, features, labels, and data gaps for chart inspection."""
        symbol = symbol.upper()
        limit = max(1, min(int(limit), 5000))
        step_ms = self._interval_milliseconds(interval)
        start_dt = self._parse_iso_datetime(start, is_end=False)
        end_dt = self._parse_iso_datetime(end, is_end=True)

        klines = self._collect_klines(symbol, interval, start_dt, end_dt, limit)
        if klines.is_empty():
            return {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "rows": 0,
                "candles": [],
                "gaps": [],
                "columns": {"features": [], "labels": []},
            }

        range_start = klines["open_time"].min()
        range_end = klines["open_time"].max()
        features = self._collect_features(symbol, interval, range_start, range_end)
        labels = self._collect_labels(symbol, interval, range_start, range_end)

        chart_df = klines
        feature_columns: list[str] = []
        label_columns: list[str] = []

        if not features.is_empty():
            feature_columns = [c for c in features.columns if c != "open_time"]
            chart_df = chart_df.join(features, on="open_time", how="left", coalesce=True)
        if not labels.is_empty():
            label_columns = [c for c in labels.columns if c != "open_time"]
            chart_df = chart_df.join(labels, on="open_time", how="left", coalesce=True)

        chart_df = chart_df.sort("open_time")
        gaps = self._detect_time_gaps(chart_df, step_ms)

        return {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "rows": len(chart_df),
            "candles": self._frame_to_chart_rows(chart_df),
            "gaps": gaps,
            "columns": {
                "features": feature_columns,
                "labels": label_columns,
            },
        }

    def _scan_inventory(self) -> list[dict[str, Any]]:
        symbols = self._discover_symbols()
        inventory: list[dict[str, Any]] = []

        def process_symbol(symbol: str) -> dict[str, Any] | None:
            raw_intervals = self._symbol_intervals(symbol)
            size_bytes = self._symbol_size(symbol)
            start_date, end_date, present, gaps = self._symbol_date_range(symbol)

            # Get base 1m partitions for lagging detection
            base_partitions = self._get_base_partitions(symbol)

            # Detailed check for each interval
            detailed_aggs = []
            for interval in raw_intervals:
                p_path = (
                    self.settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
                )
                f_path = self._features_path(symbol, interval)
                l_path = self._labels_path(symbol, interval)

                # Check gap continuity for this interval
                interval_gaps = self._interval_gaps(symbol, interval)

                status = "complete"
                msg = "Data, Features, and Labels OK"

                if not self._has_parquet_data(p_path):
                    status = "incomplete"
                    msg = "Missing or empty Parquet data"
                elif interval_gaps > 0:
                    status = "incomplete"
                    msg = f"{interval_gaps} gap(s) in data continuity"
                elif interval != "1m":
                    # Check if this interval is lagging behind 1m data
                    missing_partitions = self._get_missing_partitions(symbol, interval, base_partitions)
                    if missing_partitions:
                        status = "incomplete"
                        msg = f"Outdated data (missing {len(missing_partitions)} month(s) compared to 1m)"
                elif not self._has_valid_features(f_path):
                    status = "warning"
                    msg = "Missing or Outdated Features"
                elif not self._has_parquet_data(l_path):
                    status = "warning"
                    msg = "Missing Prediction Labels"

                detailed_aggs.append(
                    {"interval": interval, "status": status, "message": msg, "gaps": interval_gaps}
                )

            quality = self._symbol_quality(symbol, detailed_aggs, gaps)

            if size_bytes == 0 and not raw_intervals:
                return None

            return {
                "symbol": symbol,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "aggregations": detailed_aggs,
                "start_date": start_date,
                "end_date": end_date,
                "quality": quality.to_dict(),
                "months_present": present,
                "months_gaps": gaps,
                "has_features": self._features_path(symbol).exists(),
                "has_labels": self._labels_path(symbol).exists(),
            }

        # Use ThreadPoolExecutor for parallel I/O scanning
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_symbol, sorted(symbols)))
            inventory = [r for r in results if r is not None]

        return inventory

    def _discover_symbols(self) -> set[str]:
        symbols: set[str] = set()
        raw_dir = self.settings.raw_binance_spot_klines_1m_dir
        if raw_dir.exists():
            symbols.update(path.name for path in raw_dir.iterdir() if path.is_dir())

        processed_dir = self.settings.processed_binance_spot_klines_dir
        if processed_dir.exists():
            for interval_dir in processed_dir.iterdir():
                if not interval_dir.is_dir():
                    continue
                for symbol_dir in interval_dir.glob("symbol=*"):
                    symbols.add(symbol_dir.name.split("=", 1)[1])

        return symbols

    def _symbol_intervals(self, symbol: str) -> list[str]:
        intervals: list[str] = []
        processed_dir = self.settings.processed_binance_spot_klines_dir
        if not processed_dir.exists():
            return intervals

        for interval_dir in processed_dir.iterdir():
            if not interval_dir.is_dir():
                continue
            symbol_path = interval_dir / f"symbol={symbol}"
            # We include the interval in the list if the folder exists,
            # even if it's empty (so _symbol_quality can flag it as broken)
            if symbol_path.exists():
                intervals.append(interval_dir.name)
        return sorted(intervals, key=self._interval_sort_key)

    def _symbol_size(self, symbol: str) -> int:
        paths = self._symbol_paths(symbol)
        return sum(self._path_size(path) for path in paths)

    def _interval_gaps(self, symbol: str, interval: str) -> int:
        """Count gaps in month (or week for 1w) continuity for a specific interval."""
        base = self.settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
        if not base.exists():
            return 0

        if interval == "1w":
            weeks: list[tuple[int, int]] = []
            for year_dir in base.glob("year=*"):
                if not year_dir.is_dir():
                    continue
                try:
                    year = int(year_dir.name.split("=", 1)[1])
                except (ValueError, IndexError):
                    continue
                for week_dir in year_dir.glob("week=*"):
                    try:
                        week = int(week_dir.name.split("=", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    if self._has_parquet_data(week_dir):
                        weeks.append((year, week))
            if len(weeks) < 2:
                return 0
            weeks.sort()
            from datetime import date as _date

            first = _date.fromisocalendar(weeks[0][0], weeks[0][1], 1)
            last = _date.fromisocalendar(weeks[-1][0], weeks[-1][1], 1)
            expected = ((last - first).days // 7) + 1
            return max(0, expected - len(set(weeks)))
        else:
            partitions: list[tuple[int, int]] = []
            for year_dir in base.glob("year=*"):
                if not year_dir.is_dir():
                    continue
                try:
                    year = int(year_dir.name.split("=", 1)[1])
                except (ValueError, IndexError):
                    continue
                for month_dir in year_dir.glob("month=*"):
                    try:
                        month = int(month_dir.name.split("=", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    if self._has_parquet_data(month_dir):
                        partitions.append((year, month))
            if len(partitions) < 2:
                return 0
            partitions.sort()
            sy, sm = partitions[0]
            ey, em = partitions[-1]
            expected = (ey - sy) * 12 + (em - sm + 1)
            return max(0, expected - len(set(partitions)))

    def _get_base_partitions(self, symbol: str) -> list[tuple[int, int]]:
        """Returns sorted list of (year, month) partitions present in 1m base data."""
        klines_1m_path = self.settings.processed_binance_spot_klines_dir / "1m" / f"symbol={symbol}"
        if not klines_1m_path.exists():
            return []

        partitions: list[tuple[int, int]] = []
        for year_dir in klines_1m_path.glob("year=*"):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name.split("=", 1)[1])
            except (ValueError, IndexError):
                continue
            for month_dir in year_dir.glob("month=*"):
                try:
                    month = int(month_dir.name.split("=", 1)[1])
                except (ValueError, IndexError):
                    continue
                if self._has_parquet_data(month_dir):
                    partitions.append((year, month))
        return sorted(partitions)

    def _get_missing_partitions(self, symbol: str, interval: str, base_partitions: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Check if an interval is missing any partitions that are present in the base 1m data.
        Returns a list of missing (year, month) partitions.
        """
        base_path = self.settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
        if not base_path.exists():
            return base_partitions

        if interval == "1w":
            # Collect all weeks in 1w
            weeks: set[tuple[int, int]] = set()
            for year_dir in base_path.glob("year=*"):
                if not year_dir.is_dir():
                    continue
                try:
                    yr = int(year_dir.name.split("=", 1)[1])
                except (ValueError, IndexError):
                    continue
                for week_dir in year_dir.glob("week=*"):
                    try:
                        wk = int(week_dir.name.split("=", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    if self._has_parquet_data(week_dir):
                        weeks.add((yr, wk))

            # Check which base months are not covered by any 1w weeks
            missing = []
            for yr, mo in base_partitions:
                start_date = date(yr, mo, 1)
                end_date = date(yr, mo, calendar.monthrange(yr, mo)[1])
                month_weeks = set()
                curr = start_date
                while curr <= end_date:
                    iso_yr, iso_wk, _ = curr.isocalendar()
                    month_weeks.add((iso_yr, iso_wk))
                    curr += timedelta(days=1)

                if not month_weeks.intersection(weeks):
                    missing.append((yr, mo))
            return missing
        else:
            # Collect all months in interval
            months: set[tuple[int, int]] = set()
            for year_dir in base_path.glob("year=*"):
                if not year_dir.is_dir():
                    continue
                try:
                    yr = int(year_dir.name.split("=", 1)[1])
                except (ValueError, IndexError):
                    continue
                for month_dir in year_dir.glob("month=*"):
                    try:
                        mo = int(month_dir.name.split("=", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    if self._has_parquet_data(month_dir):
                        months.add((yr, mo))

            return [p for p in base_partitions if p not in months]

    def _symbol_date_range(self, symbol: str) -> tuple[str, str, int, int]:
        """Returns (start_date, end_date, months_present, months_gap_count)."""
        partitions = self._get_base_partitions(symbol)
        if not partitions:
            return ("N/A", "N/A", 0, 0)

        start_year, start_month = partitions[0]
        end_year, end_month = partitions[-1]

        # Calculate expected months
        total_expected = (end_year - start_year) * 12 + (end_month - start_month + 1)
        actual_present = len(set(partitions))
        gaps = total_expected - actual_present

        last_day = calendar.monthrange(end_year, end_month)[1]
        return (
            f"{start_year:04d}-{start_month:02d}-01",
            f"{end_year:04d}-{end_month:02d}-{last_day:02d}",
            actual_present,
            max(0, gaps),
        )

    def _symbol_quality(
        self, symbol: str, aggregations: list[dict[str, Any]], gaps_1m: int
    ) -> InventoryQuality:
        # 1. Check for gaps in any interval (including 1m)
        intervals_with_gaps = [a for a in aggregations if a.get("gaps", 0) > 0]
        if intervals_with_gaps:
            gap_details = ", ".join(f"{a['interval']}({a['gaps']})" for a in intervals_with_gaps)
            return InventoryQuality(
                "incomplete",
                "Gaps Detected",
                f"Data continuity gaps: {gap_details}",
            )

        # 2. Check for lagging intervals (outdated compared to 1m)
        lagging = [a["interval"] for a in aggregations if a["status"] == "incomplete" and "Outdated data" in a.get("message", "")]
        if lagging:
            return InventoryQuality(
                "incomplete",
                "Outdated Data",
                f"Intervals lagging behind 1m: {', '.join(sorted(lagging))}",
            )

        # 3. Check if any aggregation is broken (missing parquet)
        incomplete = [a["interval"] for a in aggregations if a["status"] == "incomplete"]
        if incomplete:
            return InventoryQuality(
                "incomplete",
                "Broken Data",
                f"Intervals with missing parquet data: {', '.join(incomplete)}",
            )

        # 4. Check for warnings (Features/Labels)
        warnings = [a["interval"] for a in aggregations if a["status"] == "warning"]
        if warnings:
            return InventoryQuality(
                "warning",
                "Incomplete Pipeline",
                f"Features/Labels missing/outdated for: {', '.join(sorted(warnings))}",
            )

        qa_warning = self._qa_warning(symbol)
        if qa_warning:
            return InventoryQuality("warning", "QA Warning", qa_warning)

        return InventoryQuality(
            "complete", "Complete", "All found intervals, features, and labels are verified."
        )

    def _qa_warning(self, symbol: str) -> str | None:
        qa_dir = self.settings.reports_qa_dir / symbol
        if not qa_dir.exists():
            return None

        for report_path in qa_dir.rglob("*.json"):
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("overall_passed") is False:
                return "QA warnings detected"
        return None

    def _symbol_paths(self, symbol: str) -> list[Path]:
        """Every path in the lake that belongs to *symbol*.

        Used both to delete a symbol and to measure it, so a missing entry
        leaks twice: deletion strands the files and the reported size
        understates them. Features and labels are enumerated per interval for
        that reason — asking only for the 1d ones left every other interval on
        disk, invisible afterwards because the inventory keys off klines.
        """
        paths = [
            self.settings.raw_binance_spot_klines_1m_dir / symbol,
            self.settings.reports_predictions_dir / symbol,
            self.settings.reports_recommendations_dir / symbol,
            self.settings.reports_eval_dir / symbol,
            self.settings.reports_qa_dir / symbol,
            self.settings.reports_dir / "evaluations" / symbol,
            self.settings.reports_dir / "backtests" / symbol,
        ]

        derived_roots = [
            self.settings.processed_binance_spot_klines_dir,
            self.settings.processed_data_dir / "binance" / "spot" / "features",
            self.settings.processed_binance_spot_labels_dir,
        ]
        for root in derived_roots:
            if not root.exists():
                continue
            for interval_dir in root.iterdir():
                if interval_dir.is_dir():
                    paths.append(interval_dir / f"symbol={symbol}")
                    paths.append(interval_dir / symbol)
        return paths

    def _features_path(self, symbol: str, interval: str = "1d") -> Path:
        return (
            self.settings.processed_data_dir
            / "binance"
            / "spot"
            / "features"
            / interval
            / f"symbol={symbol}"
        )

    def _labels_path(self, symbol: str, interval: str = "1d") -> Path:
        return self.settings.processed_binance_spot_labels_dir / interval / f"symbol={symbol}"

    def _collect_klines(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime | None,
        end_dt: datetime | None,
        limit: int,
    ) -> pl.DataFrame:
        base_path = self.settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
        lazy = self._scan_parquet_tree(base_path, start_dt, end_dt)
        if lazy is None:
            return pl.DataFrame()

        schema = lazy.schema
        missing = [col for col in KLINE_COLUMNS if col not in schema]
        if missing:
            raise ValueError(f"Klines data is missing required columns: {', '.join(missing)}")

        lazy = self._normalize_open_time_lazy(lazy)
        lazy = self._filter_time_range(lazy, start_dt, end_dt)
        lazy = (
            lazy.select(KLINE_COLUMNS)
            .unique(subset=["open_time"], keep="last")
            .sort("open_time")
            .tail(limit)
        )
        return lazy.collect()

    def _collect_features(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pl.DataFrame:
        path = get_features_parquet_path(symbol, interval, settings=self.settings)
        if not path.exists() or path.stat().st_size == 0:
            return pl.DataFrame()

        lazy = pl.scan_parquet(str(path), hive_partitioning=False)
        if "open_time" not in lazy.schema:
            return pl.DataFrame()

        feature_cols = [
            col for col in lazy.schema if col == "open_time" or col not in KLINE_SIDE_COLUMNS
        ]
        if len(feature_cols) <= 1:
            return pl.DataFrame()

        lazy = self._normalize_open_time_lazy(lazy)
        lazy = self._filter_time_range(lazy, start_dt, end_dt)
        return (
            lazy.select(feature_cols)
            .unique(subset=["open_time"], keep="last")
            .sort("open_time")
            .collect()
        )

    def _collect_labels(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pl.DataFrame:
        path = get_labels_parquet_path(symbol, interval, settings=self.settings)
        if not path.exists() or path.stat().st_size == 0:
            return pl.DataFrame()

        lazy = pl.scan_parquet(str(path), hive_partitioning=False)
        if "open_time" not in lazy.schema:
            return pl.DataFrame()

        label_cols = ["open_time", *[col for col in LABEL_COLUMNS if col in lazy.schema]]
        if len(label_cols) <= 1:
            return pl.DataFrame()

        lazy = self._normalize_open_time_lazy(lazy)
        lazy = self._filter_time_range(lazy, start_dt, end_dt)
        return (
            lazy.select(label_cols)
            .unique(subset=["open_time"], keep="last")
            .sort("open_time")
            .collect()
        )

    def _has_valid_features(self, f_path: Path) -> bool:
        # For inventory purposes, treat features as valid if at least one parquet
        # file is present. Detailed schema checks are not required here.
        return self._has_parquet_data(f_path)

    @staticmethod
    def _partition_in_range(
        year_dir: Path,
        leaf_dir: Path,
        start_dt: datetime | None,
        end_dt: datetime | None,
    ) -> bool:
        """Decide from the `year=`/`month=` names alone whether a leaf can match.

        Cheap partition pruning: a 1m symbol holds ~100 monthly files and
        scanning them all to answer a one-month chart request cost seconds.
        Anything unparseable is kept, so a malformed name never hides data.
        """
        if start_dt is None and end_dt is None:
            return True
        try:
            year = int(year_dir.name.split("=", 1)[1])
            key, raw_value = leaf_dir.name.split("=", 1)
            value = int(raw_value)
        except (ValueError, IndexError):
            return True

        if key == "month":
            first = datetime(year, value, 1, tzinfo=UTC)
            last_day = calendar.monthrange(year, value)[1]
            last = datetime(year, value, last_day, 23, 59, 59, tzinfo=UTC)
        elif key == "week":
            try:
                first = datetime.combine(
                    date.fromisocalendar(year, value, 1), datetime.min.time(), tzinfo=UTC
                )
            except ValueError:
                return True
            last = first + timedelta(days=7)
        else:
            return True

        if start_dt is not None and last < start_dt:
            return False
        if end_dt is not None and first > end_dt:
            return False
        return True

    @classmethod
    def _scan_parquet_tree(
        cls,
        path: Path,
        start_dt: datetime | None = None,
        end_dt: datetime | None = None,
    ) -> pl.LazyFrame | None:
        if path.is_file() and path.suffix == ".parquet" and path.stat().st_size > 0:
            return pl.scan_parquet(str(path), hive_partitioning=False)
        if not path.exists():
            return None

        files: list[Path] = []
        year_dirs = sorted(p for p in path.glob("year=*") if p.is_dir())
        if year_dirs:
            for year_dir in year_dirs:
                for leaf_dir in sorted(year_dir.iterdir()):
                    if not leaf_dir.is_dir():
                        continue
                    if not cls._partition_in_range(year_dir, leaf_dir, start_dt, end_dt):
                        continue
                    files.extend(
                        p
                        for p in leaf_dir.glob("*.parquet")
                        if p.is_file() and p.stat().st_size > 0
                    )
        else:
            files = [p for p in path.glob("**/*.parquet") if p.is_file() and p.stat().st_size > 0]

        if not files:
            return None
        return pl.scan_parquet([str(p) for p in files], hive_partitioning=False)

    @staticmethod
    def _normalize_open_time_lazy(lazy: pl.LazyFrame) -> pl.LazyFrame:
        dtype = lazy.schema.get("open_time")
        if dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            expr = pl.from_epoch(pl.col("open_time"), time_unit="ms").dt.replace_time_zone("UTC")
        else:
            expr = pl.col("open_time").cast(pl.Datetime("ms", "UTC"))
        return lazy.with_columns(expr.alias("open_time"))

    @staticmethod
    def _filter_time_range(
        lazy: pl.LazyFrame,
        start_dt: datetime | None,
        end_dt: datetime | None,
    ) -> pl.LazyFrame:
        if start_dt is not None:
            lazy = lazy.filter(pl.col("open_time") >= start_dt)
        if end_dt is not None:
            lazy = lazy.filter(pl.col("open_time") <= end_dt)
        return lazy

    @staticmethod
    def _parse_iso_datetime(value: str | None, *, is_end: bool) -> datetime | None:
        if not value:
            return None

        raw = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}$", raw):
            if is_end:
                year, month = (int(part) for part in raw.split("-"))
                raw = f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
            else:
                raw = f"{raw}-01"

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            parsed = datetime.fromisoformat(raw)
            if is_end:
                parsed = parsed + timedelta(days=1) - timedelta(milliseconds=1)
        else:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _detect_time_gaps(self, df: pl.DataFrame, step_ms: int) -> list[dict[str, int]]:
        timestamps = [self._to_epoch_ms(value) for value in df["open_time"].to_list()]
        gaps: list[dict[str, int]] = []
        for previous, current in zip(timestamps, timestamps[1:]):
            diff = current - previous
            if diff <= step_ms:
                continue
            missing = max(1, int(diff // step_ms) - 1)
            first_missing = previous + step_ms
            last_missing = current - step_ms
            gaps.append(
                {
                    "time": first_missing // 1000,
                    "from": first_missing // 1000,
                    "to": last_missing // 1000,
                    "missing": missing,
                }
            )
        return gaps

    def _frame_to_chart_rows(self, df: pl.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in df.to_dicts():
            try:
                candle = {
                    "time": self._to_epoch_ms(row["open_time"]) // 1000,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            except (KeyError, TypeError, ValueError):
                continue

            if not all(
                math.isfinite(candle[key]) for key in ["open", "high", "low", "close", "volume"]
            ):
                continue

            for key, value in row.items():
                if key in KLINE_COLUMNS:
                    continue
                cleaned = self._json_scalar(value)
                if cleaned is not None or key in LABEL_COLUMNS:
                    candle[key] = cleaned
            rows.append(candle)
        return rows

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, datetime):
            return int(value.timestamp())
        return value

    @staticmethod
    def _to_epoch_ms(value: Any) -> int:
        if isinstance(value, datetime):
            return int(value.timestamp() * 1000)
        if isinstance(value, int | float):
            return int(value)
        raise TypeError(f"Unsupported timestamp value: {value!r}")

    @staticmethod
    def _interval_milliseconds(interval: str) -> int:
        try:
            return INTERVAL_TO_MILLISECONDS[interval]
        except KeyError as e:
            raise ValueError(f"Unsupported interval: {interval}") from e

    @staticmethod
    def _interval_sort_key(interval: str) -> tuple[int, str]:
        if interval in INTERVAL_ORDER:
            return (INTERVAL_ORDER.index(interval), interval)
        return (len(INTERVAL_ORDER), interval)

    # How far below each root the fingerprint walks. The tree is
    # root/<interval>/symbol=X/year=Y/month=Z, and a lost or added month shows
    # up as an mtime change on its `year=` directory — three levels down. The
    # original fingerprint stopped at the first level, so it never noticed:
    # deleting three months of 1d klines left the cached inventory reporting
    # "Complete" indefinitely, and the Refresh button could not clear it.
    _FINGERPRINT_DEPTH = 3

    def _fingerprint(self) -> dict[str, int]:
        roots = [
            self.settings.raw_binance_spot_klines_1m_dir,
            self.settings.processed_binance_spot_klines_dir,
            self.settings.processed_data_dir / "binance" / "spot" / "features",
            self.settings.processed_binance_spot_labels_dir,
            self.settings.reports_dir,
        ]
        entry_count = 0
        latest_mtime = 0

        def walk(directory: Path, depth: int) -> None:
            nonlocal entry_count, latest_mtime
            try:
                entries = list(directory.iterdir())
            except OSError:
                return
            for entry in entries:
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                entry_count += 1
                latest_mtime = max(latest_mtime, stat.st_mtime_ns)
                if entry.is_dir() and depth < self._FINGERPRINT_DEPTH:
                    walk(entry, depth + 1)

        for root in roots:
            if root.exists():
                walk(root, 1)
        return {"file_count": entry_count, "latest_mtime_ns": latest_mtime}

    def _read_cache(self, fingerprint: dict[str, int]) -> list[dict[str, Any]] | None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if payload.get("fingerprint") != fingerprint:
            return None
        inventory = payload.get("inventory")
        return inventory if isinstance(inventory, list) else None

    def _write_cache(self, fingerprint: dict[str, int], inventory: list[dict[str, Any]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "inventory": inventory}, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )

    def _invalidate_cache(self) -> None:
        if self.cache_path.exists():
            self.cache_path.unlink()

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
        return 0

    @staticmethod
    def _has_parquet_data(path: Path) -> bool:
        """Strict check: directory must contain at least one .parquet file with size > 0."""
        if not path.exists():
            return False
        if path.is_file():
            return path.suffix == ".parquet" and path.stat().st_size > 0

        # Check direct children or immediate partitions (avoid deep recursive rglob for performance)
        for p in path.glob("*.parquet"):
            if p.is_file() and p.stat().st_size > 0:
                return True
        for p in path.glob("**/*.parquet"):
            if p.is_file() and p.stat().st_size > 0:
                return True
        return False
