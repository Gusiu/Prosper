"""The inventory cache must notice a partition that disappeared.

`_fingerprint` used to stat only the direct children of each root, and the
tree is root/<interval>/symbol=X/year=Y/month=Z — so losing three months of 1d
klines changed nothing it looked at. The cached inventory kept reporting
"Complete" indefinitely, and `⟳ Refresh` (which does not force a rescan) could
not clear it; only `🛡️ Verify Quality` saw reality.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta

import polars as pl
from prosper.config import Settings
from prosper.inventory import DataManager
from prosper.storage.layout import get_parquet_file_path

SYMBOL = "TESTUSDT"


def _month(settings: Settings, year: int, month: int, interval: str = "1d") -> None:
    start = datetime(year, month, 1, tzinfo=UTC)
    save = pl.DataFrame(
        {
            "open_time": [start + timedelta(days=i) for i in range(5)],
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [10.0] * 5,
        }
    )
    from prosper.storage.parquet import save_parquet

    save_parquet(
        save, get_parquet_file_path(SYMBOL, interval, year, month=month, settings=settings)
    )


def test_fingerprint_changes_when_a_month_is_deleted(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    for month in (1, 2, 3):
        _month(settings, 2024, month)

    manager = DataManager(settings)
    before = manager._fingerprint()

    shutil.rmtree(
        get_parquet_file_path(SYMBOL, "1d", 2024, month=2, settings=settings).parent
    )
    after = manager._fingerprint()

    assert before != after, "a lost month must invalidate the cached inventory"


def test_fingerprint_changes_when_a_month_is_added(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    _month(settings, 2024, 1)
    manager = DataManager(settings)
    before = manager._fingerprint()

    _month(settings, 2024, 2)

    assert manager._fingerprint() != before


def test_fingerprint_changes_when_a_whole_year_is_deleted(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    _month(settings, 2023, 12)
    _month(settings, 2024, 1)
    manager = DataManager(settings)
    before = manager._fingerprint()

    year_dir = get_parquet_file_path(SYMBOL, "1d", 2023, month=12, settings=settings).parent.parent
    shutil.rmtree(year_dir)

    assert manager._fingerprint() != before


def test_a_stale_cache_is_not_served_after_data_is_lost(tmp_path) -> None:
    """End to end: the plain (non-forcing) inventory call must see the loss."""
    settings = Settings(data_root=tmp_path)
    for month in (1, 2, 3):
        # months_present counts the 1m base partitions, not the aggregations.
        _month(settings, 2024, month, interval="1m")

    manager = DataManager(settings)
    primed = manager.get_inventory(refresh=True)
    assert primed and primed[0]["months_present"] == 3

    shutil.rmtree(
        get_parquet_file_path(SYMBOL, "1m", 2024, month=3, settings=settings).parent
    )

    # No refresh flag: this is what the Refresh button does.
    after = DataManager(settings).get_inventory()
    assert after[0]["months_present"] == 2


def test_an_untouched_lake_still_hits_the_cache(tmp_path) -> None:
    """The fingerprint has to stay stable, or the cache is pointless."""
    settings = Settings(data_root=tmp_path)
    _month(settings, 2024, 1)
    manager = DataManager(settings)

    assert manager._fingerprint() == manager._fingerprint()


def test_deleting_a_symbol_removes_every_interval(tmp_path) -> None:
    """`_symbol_paths` asked for 1d features and labels only.

    Every other interval survived a symbol deletion, and because the inventory
    keys off klines the leftovers were invisible afterwards. The same list
    measures a symbol, so the reported size understated it too.
    """
    settings = Settings(data_root=tmp_path)
    intervals = ("1m", "1h", "4h", "1d")
    for interval in intervals:
        _month(settings, 2024, 1, interval=interval)
        for kind in ("features", "labels"):
            base = (
                settings.processed_data_dir / "binance" / "spot" / kind / interval /
                f"symbol={SYMBOL}"
            )
            base.mkdir(parents=True, exist_ok=True)
            (base / f"{kind}.parquet").write_bytes(b"x")

    for tree in ("evaluations", "backtests"):
        d = settings.reports_dir / tree / SYMBOL / "ml_1d_20240101000000"
        d.mkdir(parents=True)
        (d / "artifact.json").write_text("{}", encoding="utf-8")

    DataManager(settings).delete_symbol_data(SYMBOL)

    leftovers = [
        p for p in settings.data_root.rglob(f"*{SYMBOL}*") if p.exists()
    ]
    assert not leftovers, f"symbol data survived deletion: {leftovers}"


def test_symbol_size_counts_every_interval(tmp_path) -> None:
    settings = Settings(data_root=tmp_path)
    _month(settings, 2024, 1, interval="1d")
    manager = DataManager(settings)
    base_size = manager._symbol_size(SYMBOL) if hasattr(manager, "_symbol_size") else None

    features = (
        settings.processed_data_dir / "binance" / "spot" / "features" / "1h" /
        f"symbol={SYMBOL}"
    )
    features.mkdir(parents=True)
    (features / "features.parquet").write_bytes(b"y" * 4096)

    paths = manager._symbol_paths(SYMBOL)
    assert features in paths, "1h features are part of the symbol's footprint"
    assert base_size is None or manager._symbol_size(SYMBOL) > base_size
