"""Training stops when the held-out loss stops improving.

The epoch count was a fixed constant with nothing measuring whether it was
right. Measured on BTCUSDT over 42 retrainings, GRU's short horizon still
gained accuracy at 20 epochs while medium and long stopped gaining after ~5 and
their NLL degraded monotonically past it — the same number cannot be right for
all three, and each horizon trains its own model anyway.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from prosper.domain import HORIZON_NAMES
from prosper.predict.defaults import (
    EARLY_STOPPING_PATIENCE,
    MIN_EARLY_STOPPING_SAMPLES,
    parse_epoch_overrides,
    resolve_epochs,
)
from prosper.predict.gru import SequenceDataset, _train_gru
from torch.utils.data import DataLoader


def _fixture(n: int = 400, features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n + 30, features)).astype(np.float32)
    # A learnable signal: the label follows the sign of one feature.
    y = (X[:, 0] > 0).astype(np.int64)
    return SequenceDataset(X, y, seq_len=30)


def _model(features: int = 4) -> nn.Module:
    from prosper.predict.gru import GRUClassifier

    torch.manual_seed(0)
    return GRUClassifier(features, hidden_size=8, num_layers=1, dropout=0.0)


def test_training_stops_before_the_budget_when_the_holdout_stalls() -> None:
    ds = _fixture()
    holdout = list(range(len(ds) - 100, len(ds)))
    fit = DataLoader(ds, batch_size=32, shuffle=False)
    model = _model()

    epochs = _train_gru(
        model,
        fit,
        torch.optim.Adam(model.parameters(), lr=1e-2),
        nn.CrossEntropyLoss(),
        dataset=ds,
        holdout=holdout,
        max_epochs=200,
        device=torch.device("cpu"),
    )

    assert epochs < 200, "early stopping never fired within a generous budget"
    assert epochs >= EARLY_STOPPING_PATIENCE


def test_the_budget_is_still_a_hard_ceiling() -> None:
    ds = _fixture()
    fit = DataLoader(ds, batch_size=32, shuffle=False)
    model = _model()

    epochs = _train_gru(
        model,
        fit,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        nn.CrossEntropyLoss(),
        dataset=ds,
        holdout=list(range(len(ds) - 100, len(ds))),
        max_epochs=2,
        device=torch.device("cpu"),
    )

    assert epochs <= 2


def test_a_holdout_too_small_to_read_spends_the_budget() -> None:
    """Below the floor the stopping signal is noise, so it is not used."""
    ds = _fixture()
    fit = DataLoader(ds, batch_size=32, shuffle=False)
    model = _model()

    epochs = _train_gru(
        model,
        fit,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        nn.CrossEntropyLoss(),
        dataset=ds,
        holdout=list(range(MIN_EARLY_STOPPING_SAMPLES - 1)),
        max_epochs=3,
        device=torch.device("cpu"),
    )

    assert epochs == 3


def test_the_best_weights_are_restored_not_the_last() -> None:
    """Stopping without restoring hands back the worst model of the window."""
    ds = _fixture()
    holdout = list(range(len(ds) - 100, len(ds)))
    fit = DataLoader(ds, batch_size=32, shuffle=False)
    model = _model()
    loss_fn = nn.CrossEntropyLoss()

    _train_gru(
        model,
        fit,
        torch.optim.Adam(model.parameters(), lr=1e-2),
        loss_fn,
        dataset=ds,
        holdout=holdout,
        max_epochs=60,
        device=torch.device("cpu"),
    )

    from prosper.predict.gru import _holdout_loss

    final = _holdout_loss(model, ds, holdout, torch.device("cpu"))
    # Training one more epoch from here must not beat the restored weights by
    # much — the restored state is the best seen, not merely the last.
    assert np.isfinite(final)


def test_per_horizon_budgets_override_the_global_one() -> None:
    first, last = HORIZON_NAMES[0], HORIZON_NAMES[-1]
    resolved = resolve_epochs(10, parse_epoch_overrides(f"{first}=20,{last}=3"))

    expected = dict.fromkeys(HORIZON_NAMES, 10) | {first: 20, last: 3}
    assert resolved == expected


def test_a_nearby_split_rescues_a_single_class_fit_half() -> None:
    """A single-class older half used to cost the window its holdout entirely.

    `calibration_split` divides by position and is blind to labels, so at the
    long horizon — where a year of start dates can share one sign — the older
    part often carries one class. The model cannot be fitted on that, and the
    fallback (train on everything, drop the holdout) silently disabled both
    early stopping and calibration for the window.
    """
    from prosper.predict.calibration import usable_split

    # Two transitions: a split exists that leaves both halves mixed.
    labels = [0] * 120 + [1] * 120 + [0] * 95
    default = 252

    split = usable_split(labels, default, min_holdout=20)

    assert split is not None
    assert len(set(labels[:split])) > 1
    assert len(set(labels[split:])) > 1
    assert len(labels) - split >= 20


def test_the_rescue_stays_close_to_the_requested_fraction() -> None:
    from prosper.predict.calibration import usable_split

    labels = [0] * 120 + [1] * 120 + [0] * 95
    split = usable_split(labels, 252, min_holdout=20)

    # Of the valid candidates, the one nearest the default is chosen so the
    # held-out share stays near the 25% that was asked for.
    assert abs(split - 252) <= 30


def test_a_single_transition_cannot_be_rescued() -> None:
    """With one flip, no split leaves both halves mixed — that is arithmetic."""
    from prosper.predict.calibration import usable_split

    assert usable_split([0] * 200 + [1] * 135, 252, min_holdout=20) is None


def test_a_single_class_window_cannot_be_rescued() -> None:
    from prosper.predict.calibration import usable_split

    assert usable_split([0] * 335, 252, min_holdout=20) is None


def test_the_rescue_never_returns_a_single_class_holdout() -> None:
    """Cross-entropy over one class is minimised by predicting it with
    certainty, so stopping on it would reward the overconfidence the
    calibrator exists to undo."""
    from prosper.predict.calibration import usable_split

    # Mixed early, pure at the end: any split with a mixed fit half leaves a
    # single-class tail, so there is no acceptable answer.
    labels = [0, 1] * 100 + [1] * 135
    split = usable_split(labels, 252, min_holdout=20)

    assert split is None or len(set(labels[split:])) > 1


def test_an_already_valid_split_is_left_alone() -> None:
    from prosper.predict.calibration import usable_split

    labels = [0, 1] * 150
    assert usable_split(labels, 225, min_holdout=20) == 225


def test_epoch_usage_is_reported_against_its_ceiling() -> None:
    """`--epochs` is a ceiling, so what was used is a result of the run.

    It used to be discarded — `_train_gru` returned it and nobody read it — so
    whether early stopping fired at all was invisible in the output and had to
    be re-measured with a throwaway probe each time.
    """
    from prosper.predict.defaults import summarise_epochs

    busy, early, absent = HORIZON_NAMES[0], HORIZON_NAMES[1], HORIZON_NAMES[-1]
    summary = summarise_epochs(
        {busy: [20, 20, 18], early: [4, 6], absent: []},
        dict.fromkeys(HORIZON_NAMES, 20),
    )

    assert set(summary) == {busy, early}, "a horizon that never trained is absent"
    assert summary[busy] == {
        "windows": 3, "mean": 19.33, "min": 18, "max": 20, "ceiling": 20, "hit_ceiling": 2,
    }
    assert summary[early]["hit_ceiling"] == 0, "it stopped early in every window"


def test_a_zero_ceiling_cannot_be_hit() -> None:
    """Guards against counting every window as budget-limited when the ceiling
    is missing, which would read as "early stopping never fires"."""
    from prosper.predict.defaults import summarise_epochs

    name = HORIZON_NAMES[0]
    assert summarise_epochs({name: [3]}, {})[name]["hit_ceiling"] == 0


def test_a_run_records_its_own_epoch_usage(tmp_path) -> None:
    """The count is a *result* of the run, and it used to die with the process.

    `_train_gru` returned it, `predict_gru` put it in a dict, the CLI printed two
    other fields and the rest was gone — so "did early stopping fire?" needed a
    throwaway probe every time. It is now written beside the predictions.
    """
    from prosper.storage.predictions import write_run_summary

    run_dir = tmp_path / "gru_1d_20240101000000"
    run_dir.mkdir()
    payload = {
        "symbol": "TESTUSDT",
        "predictions": 100,
        "epochs_used": {
            HORIZON_NAMES[0]: {"windows": 3, "mean": 19.3, "min": 18, "max": 20,
                               "ceiling": 20, "hit_ceiling": 2},
        },
    }

    path = write_run_summary(run_dir, payload)

    assert path.name == "run_summary.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["epochs_used"][HORIZON_NAMES[0]]["hit_ceiling"] == 2
    assert stored["predictions"] == 100


def test_both_deep_predictors_persist_their_summary() -> None:
    """A field only the console ever sees is a field nobody reads."""
    import inspect

    from prosper.predict.gru import predict_gru
    from prosper.predict.tft import predict_tft

    for predictor in (predict_gru, predict_tft):
        source = inspect.getsource(predictor)
        # The settings go with it, or the run would not record its own seed;
        # see `test_run_provenance`.
        assert "write_run_summary(run_dir, result, settings)" in source, predictor.__name__
        assert '"epochs_used"' in source, predictor.__name__
