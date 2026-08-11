"""A run must say what produced it.

`run_summary.json` recorded the training windows and the epochs each horizon
actually used, but not the **seed** or whether deterministic mode was on. That
gap surfaced while comparing two seeds of one configuration: the two runs were
indistinguishable on disk, and which seed made which had to be recovered from a
shell log that could as easily have been lost.

It matters more than bookkeeping. The project's reproducibility claim is that
re-running a configuration reproduces it byte for byte — a claim that cannot even
be stated about a run whose configuration is unrecorded.

The stamp is applied inside `write_run_summary` rather than by the callers,
because five separately-maintained result dicts had all forgotten it.
"""

from __future__ import annotations

import json
from pathlib import Path

from prosper.config import get_settings
from prosper.storage.predictions import write_run_summary


def _summary(tmp_path: Path, slug: str, **settings_kwargs) -> dict:
    run_dir = tmp_path / slug
    run_dir.mkdir(parents=True)
    settings = get_settings(data_root=tmp_path, **settings_kwargs)
    path = write_run_summary(run_dir, {"symbol": "ETHUSDT", "predictions": 10}, settings)
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_seed_is_recorded(tmp_path: Path) -> None:
    payload = _summary(tmp_path, "tft_1d_20260811041918", seed=7, deterministic=True)

    assert payload["seed"] == 7
    assert payload["deterministic"] is True


def test_an_unseeded_run_says_so_rather_than_inventing_a_number(tmp_path: Path) -> None:
    """`None` is the honest record for "the caller chose nothing"; writing 42
    would claim a configuration that was never requested."""
    payload = _summary(tmp_path, "gru_1d_20260811041918", seed=None, deterministic=False)

    assert payload["seed"] is None
    assert payload["deterministic"] is False


def test_the_model_and_interval_come_from_the_run_slug(tmp_path: Path) -> None:
    """Invariant 5 already encodes them in the directory name, so re-deriving
    them here would be a second definition that could disagree."""
    payload = _summary(tmp_path, "xgboost_1h_20260811041918", seed=1)

    assert payload["model_type"] == "xgboost"
    assert payload["interval"] == "1h"


def test_the_caller_keeps_the_last_word(tmp_path: Path) -> None:
    """A predictor that knows better than the slug must be able to say so."""
    run_dir = tmp_path / "tft_1d_20260811041918"
    run_dir.mkdir(parents=True)
    settings = get_settings(data_root=tmp_path, seed=3)
    path = write_run_summary(run_dir, {"model_type": "tft-ensemble"}, settings)

    assert json.loads(path.read_text(encoding="utf-8"))["model_type"] == "tft-ensemble"


def test_every_predictor_stamps_its_run(tmp_path: Path) -> None:
    """The stamp is only worth anything if no predictor bypasses it."""
    import inspect

    from prosper.predict import baseline, gru, ml, tft, xgboost_model

    for module in (baseline, ml, xgboost_model, gru, tft):
        source = inspect.getsource(module)
        assert "write_run_summary(" in source, module.__name__
        call = source[source.index("write_run_summary(", source.index("def predict")) :]
        call = call[: call.index(")") + 1]
        assert "settings" in call, (
            f"{module.__name__} writes a summary without its settings, so the run "
            "would not record the seed that produced it"
        )
