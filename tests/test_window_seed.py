"""Each trained model's randomness belongs to it alone.

Both DL predictors seeded once before the walk-forward loop, so every horizon and
every retraining window drew from one continuous stream. The number of training
steps taken anywhere therefore shifted the weight initialisation, dropout masks
and shuffling of everything downstream — which makes a controlled experiment
impossible.

Not hypothetically. Raising the `long` epoch ceiling from 10 to 20 on ETHUSDT tft
moved the two horizons whose ceiling had not changed: `medium` gained 0.070
accuracy and `short` 0.012, while `long` — the one actually changed — gained
0.022. The horizon under test improved *least*. Had the run not carried those two
as an in-run control, it would have been reported as evidence the ceiling helps.
"""

from __future__ import annotations

import inspect

import pytest
from prosper.domain import HORIZON_NAMES
from prosper.predict.defaults import window_seed

MODELS = ("tft", "gru")
HORIZONS = HORIZON_NAMES


def test_every_field_changes_the_seed() -> None:
    reference = window_seed(42, "tft", "long", 2025, 3)

    assert window_seed(43, "tft", "long", 2025, 3) != reference, "base seed"
    assert window_seed(42, "gru", "long", 2025, 3) != reference, "model"
    assert window_seed(42, "tft", "medium", 2025, 3) != reference, "horizon"
    assert window_seed(42, "tft", "long", 2024, 3) != reference, "year"
    assert window_seed(42, "tft", "long", 2025, 4) != reference, "month"


def test_the_model_name_is_hashed_not_summed() -> None:
    """The first version added character codes, and 'tft' and 'gru' both sum to
    334 — so the two models shared every seed in the project."""
    assert sum(ord(c) for c in "tft") == sum(ord(c) for c in "gru") == 334
    assert window_seed(42, "tft", "long", 2025, 3) != window_seed(42, "gru", "long", 2025, 3)


def test_no_collisions_across_a_decade_of_windows() -> None:
    seen: dict[int, tuple] = {}
    for model in MODELS:
        for horizon in HORIZONS:
            for year in range(2017, 2027):
                for month in range(1, 13):
                    seed = window_seed(42, model, horizon, year, month)
                    identity = (model, horizon, year, month)
                    assert seed not in seen, f"{identity} collides with {seen[seed]}"
                    seen[seed] = identity
    assert len(seen) == len(MODELS) * len(HORIZONS) * 10 * 12


def test_seeds_are_valid_for_torch() -> None:
    for model in MODELS:
        for horizon in HORIZONS:
            seed = window_seed(42, model, horizon, 2025, 6)
            assert 0 <= seed < 2**31 - 1


def test_the_seed_is_stable_across_processes() -> None:
    """`hash()` is salted per process, so it cannot be used here: a run would not
    reproduce tomorrow. This pins concrete values."""
    assert window_seed(42, "tft", "long", 2025, 3) == window_seed(42, "tft", "long", 2025, 3)
    # A literal, so a change to the mixing function is a deliberate act and not
    # something that silently invalidates every stored run.
    assert window_seed(42, "tft", "short", 2024, 1) == 1545155248


@pytest.mark.parametrize("module_name", ["prosper.predict.tft", "prosper.predict.gru"])
def test_both_predictors_reseed_per_window(module_name: str) -> None:
    """A helper nobody calls fixes nothing."""
    import importlib

    module = importlib.import_module(module_name)
    predictor = module.predict_tft if module_name.endswith("tft") else module.predict_gru
    source = inspect.getsource(predictor)

    assert "window_seed(" in source, "the per-window seed is never applied"
    assert "manual_seed(window_seed(" in source, "the derived seed must reach torch"
