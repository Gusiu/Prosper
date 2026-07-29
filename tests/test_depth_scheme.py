"""The depth-bin scheme has one definition, and artifacts name their own.

Two failure modes are guarded here. First, the scheme used to be copy-pasted
into five predictors, a Settings default and the SPA, so changing it would have
silently desynchronised the models. Second, runs written before the tail was
extended carry a `34+` bin; a reader that assumes the current labels drops that
bin, and with it most of the long-horizon probability mass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from prosper.config import Settings
from prosper.domain import (
    DEFAULT_DEPTH_BINS_STR,
    DEPTH_BIN_LABELS,
    LEGACY_DEPTH_BINS_STR,
    depth_bin_lower_edge,
    depth_bin_midpoints,
    depth_scheme_of,
    is_large_move_bin,
    parse_depth_bins,
)
from prosper.planner.windows import calculate_risk_metric

SRC = Path(__file__).resolve().parents[1] / "src" / "prosper"
# Three or more comma-separated bins in a row: the shape of a hardcoded scheme.
BIN_LITERAL = re.compile(r'"\d+-\d+,\d+-\d+,\d+-\d+[^"]*"')


def test_only_domain_defines_the_bin_scheme() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "domain.py":
            continue
        for match in BIN_LITERAL.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(SRC)}: {match.group()}")

    assert not offenders, (
        "depth bins must come from prosper.domain, not a local literal:\n  "
        + "\n  ".join(offenders)
    )


def test_the_label_list_matches_the_scheme_string() -> None:
    _, labels = parse_depth_bins(DEFAULT_DEPTH_BINS_STR)
    assert labels == DEPTH_BIN_LABELS


def test_settings_take_their_bins_from_the_domain() -> None:
    assert Settings(data_root=Path(".")).label_depth_bins_default == DEPTH_BIN_LABELS


def test_the_tail_reaches_144() -> None:
    """A one-year move on BTC has a median magnitude near 59%.

    With the old 34+ top bin, 70% of long-horizon labels landed in it and the
    depth head degenerated into a constant.
    """
    assert DEPTH_BIN_LABELS[-1] == "144+"
    assert "34-55" in DEPTH_BIN_LABELS
    assert "89-144" in DEPTH_BIN_LABELS


@pytest.mark.parametrize(
    ("label", "edge"),
    [("1-2", 1.0), ("13-21", 13.0), ("89-144", 89.0), ("144+", 144.0)],
)
def test_lower_edge_is_read_from_the_label(label: str, edge: float) -> None:
    assert depth_bin_lower_edge(label) == edge


@pytest.mark.parametrize("label", ["13-21", "21-34", "34-55", "55-89", "89-144", "144+", "34+"])
def test_large_move_bins_are_recognised_in_both_schemes(label: str) -> None:
    assert is_large_move_bin(label)


@pytest.mark.parametrize("label", ["1-2", "2-3", "3-5", "5-8", "8-13"])
def test_small_bins_are_not_large(label: str) -> None:
    assert not is_large_move_bin(label)


def test_risk_is_unchanged_by_splitting_the_tail() -> None:
    """Extending 34+ into four bins subdivides mass; it must not move it."""
    legacy = {"1-2": 0.1, "8-13": 0.2, "13-21": 0.3, "34+": 0.4}
    current = {"1-2": 0.1, "8-13": 0.2, "13-21": 0.3, "34-55": 0.15, "55-89": 0.15, "144+": 0.1}

    # Bullish bar, so the short side is the adverse one and must carry the tail.
    risk_legacy = calculate_risk_metric(0.7, 0.3, {}, legacy)
    risk_current = calculate_risk_metric(0.7, 0.3, {}, current)

    assert risk_legacy == pytest.approx(0.3 * 0.7)
    assert risk_current == pytest.approx(0.3 * 0.7)

    # Mirrored: bearish, and the long side carries the same tail.
    assert calculate_risk_metric(0.3, 0.7, legacy, {}) == pytest.approx(0.3 * 0.7)
    assert calculate_risk_metric(0.3, 0.7, current, {}) == pytest.approx(0.3 * 0.7)


def test_a_run_scheme_is_read_from_its_own_payload() -> None:
    legacy_bins = dict.fromkeys(parse_depth_bins(LEGACY_DEPTH_BINS_STR)[1], 0.125)

    assert depth_scheme_of(legacy_bins) == parse_depth_bins(LEGACY_DEPTH_BINS_STR)[1]
    assert depth_scheme_of({}) == DEPTH_BIN_LABELS


def test_the_scheme_is_ordered_by_magnitude_not_alphabetically() -> None:
    """`'8-13'` sorts after `'144+'` as a string; the order must be numeric."""
    scrambled = {"144+": 0.5, "8-13": 0.3, "21-34": 0.2}
    assert depth_scheme_of(scrambled) == ["8-13", "21-34", "144+"]


def test_the_open_bin_gets_a_representative_magnitude() -> None:
    mids = depth_bin_midpoints(["1-2", "144+"])
    assert mids["1-2"] == pytest.approx(1.5)
    assert mids["144+"] == pytest.approx(180.0)
