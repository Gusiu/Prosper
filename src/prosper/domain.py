"""Shared domain vocabulary: directions, depth bins, horizons, intervals.

These concepts are common to labelling, prediction, evaluation and planning.
They live at package level rather than under any one of those steps so that,
say, a predictor importing `days_to_steps` does not appear to depend on the
labelling module.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Direction constants ──────────────────────────────────────────────────────
# Two classes, not three. `flat` used to absorb every move under 1%, which was
# a *cost* threshold wearing a label's clothes: it destroyed the difference
# between +0.2% and +0.9%, baked one venue's fees into the training data, and
# degenerated with horizon (6.7% of short-horizon labels but 0.23% of
# long-horizon ones, so the long head was really answering a two-class
# question anyway). The cost hurdle now lives at the decision layer as
# `Settings.planner_round_trip_cost`, and how far a move goes is the depth
# head's job — its `0-1` bin holds exactly the rows `flat` used to.
DIRECTION_CLASSES: list[str] = ["short", "long"]
DIR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(DIRECTION_CLASSES)}
IDX_TO_DIR: dict[int, str] = {i: c for i, c in enumerate(DIRECTION_CLASSES)}

# ── Interval arithmetic ──────────────────────────────────────────────────────
INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
}
MINUTES_PER_DAY = 1440


def interval_minutes(interval: str) -> int:
    """Length of one bar of *interval* in minutes."""
    try:
        return INTERVAL_MINUTES[interval]
    except KeyError as e:
        raise ValueError(
            f"Unsupported interval: {interval!r}. "
            f"Known intervals: {', '.join(INTERVAL_MINUTES)}"
        ) from e


def days_to_steps(days: float, interval: str) -> int:
    """Convert a calendar-day span into a number of *interval* bars (min 1)."""
    return max(1, round(days * MINUTES_PER_DAY / interval_minutes(interval)))


# ── Horizon spec ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HorizonSpec:
    """A forecast horizon defined in *calendar days*, independent of bar size.

    ``forward_days`` is the semantic definition ("one year ahead"); the number
    of bars it spans depends on the interval and is obtained via
    :meth:`steps`. Keeping the definition calendar-based is what makes a 1h
    run and a 1d run of the same horizon comparable.
    """

    name: str
    forward_days: int

    def steps(self, interval: str) -> int:
        """Number of *interval* bars this horizon spans."""
        return days_to_steps(self.forward_days, interval)


DEFAULT_HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec("short", 28),
    HorizonSpec("medium", 182),
    HorizonSpec("long", 365),
)


# ── Direction helper ─────────────────────────────────────────────────────────
def direction_from_return(r: float | None) -> str | None:
    """Classify a forward return into long / short by sign.

    There is no threshold and no flat class: the label states what the market
    did, and whether that is worth trading after costs is a decision, not a
    label. An exactly zero return is not classifiable, so it returns ``None``
    and the row is excluded rather than being silently assigned a side.

    Returns ``None`` when *r* is ``None`` too, for callers passing nullables.
    """
    if r is None or r == 0.0:
        return None
    return "long" if r > 0.0 else "short"


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


def assign_depth_bin(value_pct: float, bins: list[tuple[float, float | None]]) -> int | None:
    """Assign a positive percentage value to a depth-bin index."""
    for idx, (min_val, max_val) in enumerate(bins):
        if max_val is None:
            if value_pct >= min_val:
                return idx
        else:
            if min_val <= value_pct < max_val:
                return idx
    return None


# ── Depth-bin constants ──────────────────────────────────────────────────────
# The bin edges follow the Fibonacci sequence. They run to 144+ because the
# scale of a move grows with the horizon: on BTCUSDT a one-year move has a
# median magnitude near 59%, so the old top bin (34+) swallowed 70% of the
# long-horizon labels and the depth head degenerated into a constant. The
# extended tail restores its resolution — measured on 2020-09..2026-06, the
# effective number of bins used at the long horizon goes from 2.75 to 7.22.
# It starts at 0 because there is no flat class to absorb the sub-1% moves any
# more: the depth head now covers the whole magnitude axis, and `0-1` holds
# exactly the rows that used to be labelled flat.
DEFAULT_DEPTH_BINS_STR = "0-1,1-2,2-3,3-5,5-8,8-13,13-21,21-34,34-55,55-89,89-144,144+"

# Runs produced before 2026-07-29 used this scheme. Their artifacts name their
# own bins, so nothing needs migrating — but a reader that assumes the current
# labels would silently drop their `34+` mass, which is most of the tail.
LEGACY_DEPTH_BINS_STR = "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+"

DEPTH_BIN_RANGES, DEPTH_BIN_LABELS = parse_depth_bins(DEFAULT_DEPTH_BINS_STR)
N_DEPTH_BINS: int = len(DEPTH_BIN_LABELS)

# A move at or above this magnitude (in percent) is what the planner treats as
# "large" when it measures risk. Deriving the set from the edges rather than
# naming bins keeps the metric identical across bin schemes.
LARGE_MOVE_PCT: float = 13.0


def depth_bin_lower_edge(label: str) -> float:
    """Lower edge of a depth-bin label, in percent. ``'21-34'`` -> 21.0."""
    head = label.split("-", 1)[0].rstrip("+")
    return float(head)


def is_large_move_bin(label: str, threshold_pct: float = LARGE_MOVE_PCT) -> bool:
    """Whether *label* denotes a move the planner counts as large."""
    try:
        return depth_bin_lower_edge(label) >= threshold_pct
    except ValueError:
        return False


def depth_bin_midpoints(labels: list[str]) -> dict[str, float]:
    """Representative magnitude of each bin, in percent.

    The open-ended top bin has no midpoint; ``low * 1.25`` is the same
    convention the evaluation layer has always used for it.
    """
    ranges, parsed = parse_depth_bins(",".join(labels))
    return {
        label: (low + high) / 2.0 if high is not None else low * 1.25
        for label, (low, high) in zip(parsed, ranges, strict=True)
    }


def depth_scheme_of(depth_bins: dict[str, float] | None) -> list[str]:
    """The bin labels a stored prediction actually uses, in ascending order.

    Artifacts are self-describing: the keys of ``depth_long_bins`` name the
    scheme that produced them. Reading the scheme from the payload rather than
    from the current constant is what lets an 8-bin run and an 11-bin run be
    evaluated side by side without either being misread.
    """
    if not depth_bins:
        return list(DEPTH_BIN_LABELS)
    try:
        return sorted(depth_bins, key=depth_bin_lower_edge)
    except ValueError:
        return list(DEPTH_BIN_LABELS)
