"""Every flag the API sends must exist on the CLI command it targets.

The browser never runs Python; `api/commands.py` builds an argv list and the
API spawns `python -m prosper.cli ...`. Nothing checked that the two agreed, so
`predict baseline` could lose `--interval` while the API kept sending it —
which it did. Training the baseline from the UI failed with "No such option:
--interval" and the only symptom was a task that exited non-zero.
"""

from __future__ import annotations

import typer
from prosper.api.commands import build_train_command
from prosper.api.models import ResearchFlags, TrainRequest
from prosper.cli import app

MODEL_TYPES = ("baseline", "ml", "xgboost", "gru", "tft")


def _cli_options(*path: str) -> set[str]:
    """Every option string a CLI command accepts, e.g. {'--symbol', ...}."""
    click_app = typer.main.get_command(app)
    command = click_app
    for part in path:
        command = command.commands[part]  # type: ignore[attr-defined]
    names: set[str] = set()
    for param in command.params:
        names.update(getattr(param, "opts", []))
        names.update(getattr(param, "secondary_opts", []))
    return names


def _flags_in(argv: list[str]) -> set[str]:
    return {token for token in argv if token.startswith("--")}


def test_every_predictor_accepts_what_the_api_sends() -> None:
    offenders: list[str] = []
    for model_type in MODEL_TYPES:
        argv = build_train_command(
            TrainRequest(
                symbol="BTCUSDT",
                model_type=model_type,
                interval="1d",
                start="2024-01-01",
                end="2024-06-30",
            )
        )[0]
        accepted = _cli_options("predict", model_type)
        for flag in _flags_in(argv):
            if flag not in accepted:
                offenders.append(f"predict {model_type}: {flag}")

    assert not offenders, "API sends flags the CLI does not accept:\n  " + "\n  ".join(offenders)


def test_research_flags_reach_every_predictor() -> None:
    """--strict/--deterministic/--seed/--save-metadata are advertised on all of them."""
    offenders: list[str] = []
    for model_type in MODEL_TYPES:
        argv = build_train_command(
            TrainRequest(
                symbol="BTCUSDT",
                model_type=model_type,
                start="2024-01-01",
                end="2024-06-30",
                flags=ResearchFlags(strict=True, deterministic=True, seed=7, save_metadata=True),
            )
        )[0]
        accepted = _cli_options("predict", model_type)
        for flag in _flags_in(argv):
            if flag not in accepted:
                offenders.append(f"predict {model_type}: {flag}")

    assert not offenders, "research flags rejected:\n  " + "\n  ".join(offenders)


def test_every_predictor_takes_an_interval() -> None:
    """Horizons are calendar spans, so the interval is never implicit."""
    for model_type in MODEL_TYPES:
        assert "--interval" in _cli_options("predict", model_type), model_type


def test_the_api_never_invents_a_model_type() -> None:
    """The CLI is the authority on which predictors exist."""
    cli_models = set(typer.main.get_command(app).commands["predict"].commands)  # type: ignore[attr-defined]
    assert set(MODEL_TYPES) <= cli_models


def test_the_ui_offers_every_model_the_api_accepts() -> None:
    """The baseline was missing from the dropdown, so the benchmark model
    could not be trained from the dashboard at all — even though the API
    accepts it, the CLI implements it, and every comparison depends on it.
    """
    import re
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )
    select = re.search(r'id="model-select".*?</select>', html, re.S)
    assert select, "model-select not found in index.html"
    offered = set(re.findall(r'<option value="([^"]+)"', select.group()))

    assert set(MODEL_TYPES) == offered, (
        f"dropdown offers {sorted(offered)}, API accepts {sorted(MODEL_TYPES)}"
    )
