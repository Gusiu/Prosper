"""Shared depth-bin, direction, and horizon utilities.

Canonical implementations — imported by labels/build, predict/*, eval/*.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Direction constants ──────────────────────────────────────────────────────
DIRECTION_CLASSES: list[str] = ["short", "flat", "long"]
DIR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(DIRECTION_CLASSES)}
IDX_TO_DIR: dict[int, str] = {i: c for i, c in enumerate(DIRECTION_CLASSES)}

# ── Depth-bin constants ──────────────────────────────────────────────────────
DEFAULT_DEPTH_BINS_STR = "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+"
DEPTH_BIN_LABELS: list[str] = [
    "1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-34", "34+",
]
N_DEPTH_BINS: int = len(DEPTH_BIN_LABELS)


# ── Horizon spec ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HorizonSpec:
    name: str
    forward_days: int


DEFAULT_HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec("short", 28),
    HorizonSpec("medium", 182),
    HorizonSpec("long", 365),
)


# ── Direction helper ─────────────────────────────────────────────────────────
def direction_from_return(r: float, flat_threshold: float) -> str | None:
    """Classify a forward return into long / flat / short.

    Returns ``None`` when *r* is ``None`` (convenience for callers that pass
    nullable values).
    """
    if r is None:
        return None
    if abs(r) <= flat_threshold:
        return "flat"
    return "long" if r > flat_threshold else "short"


# ── Depth-bin helpers ────────────────────────────────────────────────────────
def parse_depth_bins(
    bins_str: str,
) -> tuple[list[tuple[float, float | None]], list[str]]:
    """Parse ``'1-2,2-3,...,34+'`` into ranges and stable labels."""
    bins: list[tuple[float, float | None]] = []
    labels: list[str] = []
    parts = [p.strip() for p in bins_str.split(",") if p.strip()]
    for part in parts:
        if part.endswith("+"):
            min_val = float(part[:-1])
            bins.append((min_val, None))
            labels.append(f"{int(min_val)}+")
        else:
            lo, hi = part.split("-", 1)
            min_val = float(lo)
            max_val = float(hi)
            bins.append((min_val, max_val))
            labels.append(f"{int(min_val)}-{int(max_val)}")
    return bins, labels


def assign_depth_bin(
    value_pct: float, bins: list[tuple[float, float | None]]
) -> int | None:
    """Assign a positive percentage value to a depth-bin index."""
    for idx, (min_val, max_val) in enumerate(bins):
        if max_val is None:
            if value_pct >= min_val:
                return idx
        else:
            if min_val <= value_pct < max_val:
                return idx
    return None
