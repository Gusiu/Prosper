"""A symbol whose history begins inside the requested range must not crash the run.

Asking for 2020-01-01 onwards from a symbol first quoted on 2020-08-11 puts its
very first bar inside the range, so the walk-forward loop starts at row 0. The
causal normalisation window reaches back from there and finds nothing, and
`np.percentile` cannot describe an empty array — it indexes `arr[-1]`. Six runs
of a recompute died that way, after the predictions they would have produced were
already known to be untrainable.

The honest outcome is an untrained bar (invariant 3), not an exception: a bar with
no history behind it has no training rows either, so there was never anything to
lose. Losing the whole run instead is the defect.
"""

from __future__ import annotations

import numpy as np
import pytest
from prosper.predict.gru import _robust_normalise


def test_an_empty_fitting_window_is_refused_not_crashed() -> None:
    """`None` says "cannot normalise", which the caller turns into an untrained
    bar. Returning zeros instead would hand the model a silently fabricated
    scale."""
    features = np.arange(60, dtype=np.float32).reshape(20, 3)

    assert _robust_normalise(features[0:0], features) is None
    assert _robust_normalise(np.empty((0, 3), dtype=np.float32), features) is None


def test_a_window_of_one_bar_still_normalises() -> None:
    """The guard must fire only on genuinely empty input; one row is degenerate
    but well defined, and the IQR floor already handles a zero spread."""
    features = np.arange(60, dtype=np.float32).reshape(20, 3)

    normalised = _robust_normalise(features[0:1], features)

    assert normalised is not None
    assert normalised.shape == features.shape
    assert np.isfinite(normalised).all()


def test_normalisation_reads_only_the_window_it_is_given() -> None:
    """It is fitted on past bars and applied to all of them; if the statistics
    came from the full array the forecast would see its own future."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 4)).astype(np.float32)
    features[100:] += 50.0  # a regime shift strictly after the fitting window

    fitted_on_past = _robust_normalise(features[:100], features)
    assert fitted_on_past is not None
    # The shifted tail must land far from zero: the statistics knew nothing of it.
    assert abs(float(np.median(fitted_on_past[150:]))) > 1.0


@pytest.mark.parametrize("module_name", ["prosper.predict.gru", "prosper.predict.tft"])
def test_both_sequence_predictors_guard_the_empty_window(module_name: str) -> None:
    """Both compute the same statistics, so both had the same crash."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module_name))
    assert "shape[0] == 0" in source or "size == 0" in source, (
        f"{module_name} does not guard an empty normalisation window"
    )
