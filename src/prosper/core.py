"""Core data management and inventory logic for Prosper."""

from __future__ import annotations

import calendar
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prosper.config import Settings, get_settings


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

    def _scan_inventory(self) -> list[dict[str, Any]]:
        symbols = self._discover_symbols()
        inventory: list[dict[str, Any]] = []

        def process_symbol(symbol: str) -> dict[str, Any] | None:
            raw_intervals = self._symbol_intervals(symbol)
            size_bytes = self._symbol_size(symbol)
            start_date, end_date, present, gaps = self._symbol_date_range(symbol)

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
                elif not self._has_parquet_data(f_path):
                    status = "warning"
                    msg = "Missing Indicators (Features)"
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
        return intervals

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

    def _symbol_date_range(self, symbol: str) -> tuple[str, str, int, int]:
        """Returns (start_date, end_date, months_present, months_gap_count)."""
        klines_1m_path = self.settings.processed_binance_spot_klines_dir / "1m" / f"symbol={symbol}"
        if not klines_1m_path.exists():
            return ("N/A", "N/A", 0, 0)

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

        if not partitions:
            return ("N/A", "N/A", 0, 0)

        partitions.sort()
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

        # Collect all intervals to verify Features/Labels
        # We MUST check 1m as it is the foundation
        check_intervals = set([a["interval"] for a in aggregations])
        check_intervals.add("1m")

        # 2. Check if any aggregation is broken (missing parquet)
        incomplete = [a["interval"] for a in aggregations if a["status"] == "incomplete"]
        if incomplete:
            return InventoryQuality(
                "incomplete",
                "Broken Data",
                f"Intervals with missing parquet data: {', '.join(incomplete)}",
            )

        # 3. Check for warnings (Features/Labels)
        missing_fl = []
        for interval in check_intervals:
            f_path = self._features_path(symbol, interval)
            l_path = self._labels_path(symbol, interval)
            if not self._has_parquet_data(f_path) or not self._has_parquet_data(l_path):
                missing_fl.append(interval)

        if missing_fl:
            return InventoryQuality(
                "warning",
                "Incomplete Pipeline",
                f"Features/Labels missing for: {', '.join(sorted(missing_fl))}",
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
        paths = [
            self.settings.raw_binance_spot_klines_1m_dir / symbol,
            self.settings.reports_predictions_dir / symbol,
            self.settings.reports_recommendations_dir / symbol,
            self.settings.reports_eval_dir / symbol,
            self.settings.reports_qa_dir / symbol,
            self._features_path(symbol),
            self._labels_path(symbol),
        ]
        processed_dir = self.settings.processed_binance_spot_klines_dir
        if processed_dir.exists():
            for interval_dir in processed_dir.iterdir():
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

    def _fingerprint(self) -> dict[str, int]:
        # Fast fingerprint: only mtime of main directories and file count directly in symbol folders
        roots = [
            self.settings.raw_binance_spot_klines_1m_dir,
            self.settings.processed_binance_spot_klines_dir,
            self.settings.processed_data_dir / "binance" / "spot" / "features",
            self.settings.processed_binance_spot_labels_dir,
            self.settings.reports_dir,
        ]
        file_count = 0
        latest_mtime = 0
        for root in roots:
            if not root.exists():
                continue
            # only direct subfolders (e.g. symbol)
            for sub in root.iterdir():
                try:
                    stat = sub.stat()
                except OSError:
                    continue
                latest_mtime = max(latest_mtime, stat.st_mtime_ns)
                if sub.is_dir():
                    # count only files directly in the symbol folder
                    file_count += sum(1 for f in sub.iterdir() if f.is_file())
                elif sub.is_file():
                    file_count += 1
        return {"file_count": file_count, "latest_mtime_ns": latest_mtime}

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
