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
    """The stamp is only worth anything if no predictor bypasses it.

    Predictions reach disk in one place, `walk_forward.finish_run`, which writes
    the summary with the settings beside them; every predictor ends in `run`.
    """
    import inspect

    from prosper.predict import baseline, gru, ml, tft, walk_forward, xgboost_model

    finish = inspect.getsource(walk_forward.finish_run)
    assert "write_run_summary(run_dir, result, settings)" in finish
    for module in (baseline, ml, xgboost_model, gru, tft):
        source = inspect.getsource(module)
        assert "write_versioned_predictions" not in source, (
            f"{module.__name__} writes predictions itself, around the summary"
        )
        entry = next(name for name in dir(module) if name.startswith("predict_"))
        assert "return run(" in inspect.getsource(getattr(module, entry)), module.__name__


def test_the_environment_that_produced_a_run_is_recorded(tmp_path: Path) -> None:
    """`--save-metadata` promised this and wrote nothing; every run records it now.

    A retrained model is only comparable with an old one when the libraries
    that compute it match, and after the fact that is unanswerable unless it
    was written down.
    """
    environment = _summary(tmp_path, "xgboost_1d_20260811041918")["environment"]

    assert environment["python"]
    assert environment["packages"]["xgboost"] == "1.7.6"
    assert set(environment["packages"]) >= {"numpy", "polars", "torch", "scikit-learn"}
    assert "git_commit" in environment and "git_dirty" in environment
    assert set(environment["threads"]) == {"torch", "OMP_NUM_THREADS", "POLARS_MAX_THREADS"}


def test_no_command_offers_a_flag_that_does_nothing() -> None:
    """--strict and --save-metadata were accepted by fifteen commands and read by
    none. The dashboard offered both as switches."""
    import os
    import subprocess
    import sys

    # The help is drawn with box characters; decoding them in the locale's
    # code page fails on Windows whenever the child writes UTF-8.
    utf8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for command in (["predict", "xgboost"], ["eval", "predictions"], ["backfill"]):
        help_text = subprocess.run(
            [sys.executable, "-m", "prosper.cli", *command, "--help"],
            capture_output=True, text=True, encoding="utf-8", env=utf8, check=True,
        ).stdout
        assert "--strict" not in help_text and "--save-metadata" not in help_text



def test_reading_the_configuration_does_not_touch_the_disk(tmp_path: Path) -> None:
    """Building Settings used to create raw/, processed/, reports/ and meta/ under
    whatever root it was given; four empty folders sat in the repository root."""
    from prosper.config import Settings

    root = tmp_path / "somewhere"
    Settings(data_root=root)
    assert not root.exists()
