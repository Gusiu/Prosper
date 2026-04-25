"""Tests for label building."""

import pytest

from prosper.labels.build import assign_depth_bin, parse_depth_bins


def test_parse_depth_bins() -> None:
    """Test parsing depth bin string."""
    bins_str = "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+"
    bins = parse_depth_bins(bins_str)

    assert len(bins) == 8
    assert bins[0] == (1.0, 2.0)
    assert bins[-1] == (34.0, None)


def test_assign_depth_bin() -> None:
    """Test assigning value to depth bin."""
    bins = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, None)]

    assert assign_depth_bin(1.5, bins) == 0
    assert assign_depth_bin(2.5, bins) == 1
    assert assign_depth_bin(4.0, bins) == 2
    assert assign_depth_bin(10.0, bins) == 3
    assert assign_depth_bin(0.5, bins) is None


def test_parse_depth_bins_invalid() -> None:
    """Test parsing invalid depth bin string."""
    with pytest.raises(ValueError):
        parse_depth_bins("invalid")
