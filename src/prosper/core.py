"""Core data management and inventory logic for Prosper."""

from __future__ import annotations

import calendar
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prosper.config import Settings, get_settings


BASE_INTERVAL = "1m"
REQUIRED_INVENTORY_INTERVALS = (BASE_INTERVAL,)
ANALYSIS_READY_INTERVALS = ("1d",)


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

        for symbol in sorted(symbols):
            aggregations = self._symbol_intervals(symbol)
            size_bytes = self._symbol_size(symbol)
            start_date, end_date = self._symbol_date_range(symbol)
            quality = self._symbol_quality(symbol, aggregations)

            if size_bytes == 0 and not aggregations:
                continue

            inventory.append(
                {
                    "symbol": symbol,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "aggregations": sorted(aggregations),
                    "start_date": start_date,
                    "end_date": end_date,
                    "quality": quality.to_dict(),
                    "has_features": self._features_path(symbol).exists(),
                    "has_labels": self._labels_path(symbol).exists(),
                }
            )

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
            symbol_path = interval_dir / f"symbol={symbol}"
            if interval_dir.is_dir() and self._has_parquet_data(symbol_path):
                intervals.append(interval_dir.name)
        return intervals

    def _symbol_size(self, symbol: str) -> int:
        paths = self._symbol_paths(symbol)
        return sum(self._path_size(path) for path in paths)

    def _symbol_date_range(self, symbol: str) -> tuple[str, str]:
        klines_1m_path = self.settings.processed_binance_spot_klines_dir / "1m" / f"symbol={symbol}"
        if not klines_1m_path.exists():
            return ("N/A", "N/A")

        partitions: list[tuple[int, int]] = []
        for year_dir in klines_1m_path.glob("year=*"):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name.split("=", 1)[1])
            except ValueError:
                continue
            for month_dir in year_dir.glob("month=*"):
                try:
                    month = int(month_dir.name.split("=", 1)[1])
                except ValueError:
                    continue
                if self._has_parquet_data(month_dir):
                    partitions.append((year, month))

        if not partitions:
            return ("N/A", "N/A")

        start_year, start_month = min(partitions)
        end_year, end_month = max(partitions)
        last_day = calendar.monthrange(end_year, end_month)[1]
        return (
            f"{start_year:04d}-{start_month:02d}-01",
            f"{end_year:04d}-{end_month:02d}-{last_day:02d}",
        )

    def _symbol_quality(self, symbol: str, intervals: list[str]) -> InventoryQuality:
        missing_required = [item for item in REQUIRED_INVENTORY_INTERVALS if item not in intervals]
        if missing_required:
            return InventoryQuality(
                "incomplete",
                "Incomplete",
                f"Missing base interval: {', '.join(missing_required)}",
            )

        missing_analysis = [item for item in ANALYSIS_READY_INTERVALS if item not in intervals]
        if missing_analysis:
            return InventoryQuality(
                "incomplete",
                "Incomplete",
                f"Missing analysis interval: {', '.join(missing_analysis)}",
            )

        qa_warning = self._qa_warning(symbol)
        if qa_warning:
            return InventoryQuality("warning", "Warning", qa_warning)

        return InventoryQuality("healthy", "Healthy", "No QA issues detected")

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

    def _features_path(self, symbol: str) -> Path:
        return self.settings.processed_data_dir / "binance" / "spot" / "features" / "1d" / f"symbol={symbol}"

    def _labels_path(self, symbol: str) -> Path:
        return self.settings.processed_binance_spot_labels_dir / "1d" / f"symbol={symbol}"

    def _fingerprint(self) -> dict[str, int]:
        # Szybki fingerprint: tylko mtime głównych katalogów i liczba plików bezpośrednio w folderach symboli
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
            # tylko bezpośrednie podfoldery (np. symbol)
            for sub in root.iterdir():
                try:
                    stat = sub.stat()
                except OSError:
                    continue
                latest_mtime = max(latest_mtime, stat.st_mtime_ns)
                if sub.is_dir():
                    # liczymy tylko pliki bezpośrednio w folderze symbolu
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
            json.dumps({"fingerprint": fingerprint, "inventory": inventory}, indent=2, sort_keys=True),
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
        if path.is_file():
            return path.suffix == ".parquet"
        return path.is_dir() and any(file.suffix == ".parquet" for file in path.rglob("*.parquet"))
