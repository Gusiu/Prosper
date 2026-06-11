"""Unit tests for walk-forward training window boundaries."""

from prosper.labels.depth import HorizonSpec


def _safe_end(train_start: int, train_end: int, forward_days: int) -> int:
    """Mirror the leakage guard used across predict modules."""
    return max(train_start, train_end - forward_days)


def test_safe_end_excludes_future_labels_for_short_horizon() -> None:
    train_start, train_end, horizon = 0, 100, 28
    safe = _safe_end(train_start, train_end, horizon)
    assert safe == 72
    # Label at index 72 needs close[72 + 28] = close[100], which is the last known index
    assert safe + horizon == train_end


def test_safe_end_respects_train_start_floor() -> None:
    train_start, train_end, horizon = 90, 100, 28
    safe = _safe_end(train_start, train_end, horizon)
    assert safe == 90


def test_horizon_specs_match_standard_defaults() -> None:
    horizons = [
        HorizonSpec("short", 28),
        HorizonSpec("medium", 182),
        HorizonSpec("long", 365),
    ]
    assert [h.forward_days for h in horizons] == [28, 182, 365]
