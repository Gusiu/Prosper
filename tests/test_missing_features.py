"""Both sequence networks see a missing feature as the same number.

The indicators' warm-up leaves the first 89 bars of every symbol with missing
features, and the early training windows of a run reach back that far. GRU put
such a value at the window's median, TFT at a raw zero - for an RSI on its
0-100 scale an extreme reading - so the two networks the study compares were
trained on different inputs for the same bars. Both also counted a missing
value as a zero when they took the median and interquartile range, which moved
the scale of every other bar of the window.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

LONG_WARM_UP, SHORT_WARM_UP = 89, 13


def _features() -> np.ndarray:
    rng = np.random.default_rng(3)
    features = rng.normal(50.0, 10.0, size=(300, 3))
    features[:LONG_WARM_UP, 1] = np.nan
    features[:SHORT_WARM_UP, 2] = np.nan
    return features


def _scaled(network: str) -> np.ndarray:
    from prosper.predict import gru, tft

    forecaster = {"gru": gru.GRUForecaster, "tft": tft.TFTForecaster}[network]
    features = _features()
    scaled = forecaster.normalise(SimpleNamespace(features=features), features[:250])
    assert scaled is not None
    return scaled


def test_both_networks_scale_the_same_window_identically() -> None:
    assert np.array_equal(_scaled("gru"), _scaled("tft"))


@pytest.mark.parametrize("network", ["gru", "tft"])
def test_a_missing_feature_lands_at_the_window_median(network: str) -> None:
    scaled = _scaled(network)

    assert np.all(scaled[:LONG_WARM_UP, 1] == 0.0)
    assert np.all(scaled[:SHORT_WARM_UP, 2] == 0.0)


@pytest.mark.parametrize("network", ["gru", "tft"])
def test_the_scale_is_taken_from_the_values_present(network: str) -> None:
    """Median 0 and interquartile range 1 over the bars that have the feature."""
    present = _scaled(network)[LONG_WARM_UP:250, 1]

    assert float(np.median(present)) == pytest.approx(0.0, abs=1e-12)
    q75, q25 = np.percentile(present, [75, 25])
    assert float(q75 - q25) == pytest.approx(1.0, abs=1e-12)
