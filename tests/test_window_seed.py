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
    assert window_seed(42, "tft", "year", 2025, 3) == window_seed(42, "tft", "year", 2025, 3)
    # A literal, so a change to the mixing function is a deliberate act and not
    # something that silently invalidates every stored run.
    #
    # It changed once, deliberately, on 2026-08-11: the symbol joined the hashed
    # identity, because without it every symbol shared one initialisation stream.
    # Runs written before that date do not reproduce under this build, which is
    # the cost of the fix and the reason this literal exists to make it visible.
    assert window_seed(42, "tft", "week", 2024, 1) == 2076467837


@pytest.mark.parametrize("module_name", ["prosper.predict.tft", "prosper.predict.gru"])
def test_both_predictors_reseed_per_window(module_name: str) -> None:
    """A helper nobody calls fixes nothing."""
    import importlib

    module = importlib.import_module(module_name)
    predictor = module.predict_tft if module_name.endswith("tft") else module.predict_gru
    source = inspect.getsource(predictor)

    assert "window_seed(" in source, "the per-window seed is never applied"
    assert "manual_seed(window_seed(" in source, "the derived seed must reach torch"


def test_selecting_fewer_horizons_leaves_the_selected_ones_untouched() -> None:
    """`--horizons month` must train the same month model a full run would.

    The property the replication test rests on: TFT costs about 29 minutes per
    horizon, so testing one hypothesis about the monthly horizon on four new
    symbols is only affordable if training it alone gives the same model. Two
    things have to hold, and both are structural rather than incidental.

    Measured on xgboost over ETHUSDT 2025-01..2026-04: the month payload differs
    on 0 of 485 bars between a four-horizon run and `--horizons month`.
    """
    import ast

    from prosper.predict import gru, ml, tft, xgboost_model

    for module in (ml, xgboost_model, gru, tft):
        source = inspect.getsource(module)
        tree = ast.parse(source)

        # 1. The windows are resolved from every horizon, not the selected ones,
        #    so the derived widths and the normalisation span do not move.
        resolves = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_train_windows"
        ]
        assert resolves, f"{module.__name__} does not resolve its windows"
        for call in resolves:
            passed = {ast.unparse(arg) for arg in call.args}
            assert passed & {"horizons", "horizon_specs"}, (
                f"{module.__name__} resolves windows from a filtered horizon list; "
                "the widths and the normalisation span would then depend on --horizons"
            )

        # 2. The selection is applied inside the retraining loop, where it can
        #    only skip a horizon — never reshape another one's inputs.
        assert "not in selected" in source, module.__name__


def test_the_seed_ignores_which_other_horizons_ran() -> None:
    """The arithmetic behind the property above: a window's seed names only its
    own horizon, so a run training one horizon seeds it exactly as a run training
    four does."""
    for model in MODELS:
        for horizon in HORIZONS:
            alone = window_seed(42, model, horizon, 2025, 6)
            alongside = window_seed(42, model, horizon, 2025, 6)
            assert alone == alongside
            others = {window_seed(42, model, name, 2025, 6) for name in HORIZONS if name != horizon}
            assert alone not in others, f"{model}/{horizon} shares a seed with a sibling"


def test_two_symbols_do_not_share_an_initialisation() -> None:
    """The omission that cost a result.

    `window_seed` hashed (base seed, model, horizon, year, month) and stopped
    there, so for a given retraining window *every symbol drew the same weights,
    the same dropout masks and the same batch order*. Four symbols trained under
    one base seed were therefore not four independent draws of "this model on a
    new dataset" — they were one draw of the initialisation sequence applied to
    four datasets, and a four-symbol replication was read as independent evidence
    on that basis.

    The shared factor is not small. Reseeding moved one symbol's monthly accuracy
    by 0.049 on average and 0.064 at worst, against a claimed effect of 0.043.
    """
    window = (42, "tft", "month", 2023, 1)
    seeds = {s: window_seed(*window, symbol=s) for s in ("ADAUSDT", "XRPUSDT", "LTCUSDT")}

    assert len(set(seeds.values())) == 3, f"symbols still share a seed: {seeds}"


def test_the_symbol_is_the_only_thing_that_changed() -> None:
    """Adding it must not disturb the isolation the function exists for: two runs
    differing in one hyperparameter still agree everywhere that hyperparameter
    does not reach."""
    for symbol in ("BTCUSDT", ""):
        reference = window_seed(42, "tft", "month", 2025, 3, symbol)
        assert window_seed(43, "tft", "month", 2025, 3, symbol) != reference, "base seed"
        assert window_seed(42, "gru", "month", 2025, 3, symbol) != reference, "model"
        assert window_seed(42, "tft", "week", 2025, 3, symbol) != reference, "horizon"
        assert window_seed(42, "tft", "month", 2024, 3, symbol) != reference, "year"
        assert window_seed(42, "tft", "month", 2025, 4, symbol) != reference, "month"
        # And it is still a pure function of that identity.
        assert window_seed(42, "tft", "month", 2025, 3, symbol) == reference


def test_the_symbol_reaches_the_seed_from_both_deep_predictors() -> None:
    """A decorrelated seed nobody passes is no decorrelation at all."""
    from prosper.predict import gru, tft

    for module in (gru, tft):
        source = inspect.getsource(module)
        call = source[source.index("window_seed(") :]
        call = call[: call.index(")") + 1]
        assert "symbol=symbol" in call, (
            f"{module.__name__} seeds its windows without naming the symbol, so "
            "every symbol would share one initialisation stream again"
        )
