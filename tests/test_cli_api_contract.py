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
from prosper.domain import HORIZON_NAMES

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


def _typer_default(command: str, option: str):
    click_app = typer.main.get_command(app)
    cmd = click_app.commands["predict"].commands[command]  # type: ignore[attr-defined]
    for param in cmd.params:
        if option in getattr(param, "opts", []):
            return param.default
    raise AssertionError(f"{command} has no {option}")


def test_the_api_does_not_impose_an_epoch_count_the_caller_never_chose() -> None:
    """This is why the dashboard trained 5 epochs where the CLI trained 20.

    Typer's default applies only when the flag is absent, and the API used to
    append `--epochs` on every call, so its own default always won. Two
    definitions of one value; the API's silently overrode the other.
    """
    for model_type in ("gru", "tft"):
        argv = build_train_command(
            TrainRequest(
                symbol="BTCUSDT", model_type=model_type,
                start="2024-01-01", end="2024-06-30",
            )
        )[0]
        assert "--epochs" not in argv
        assert "--max-epochs" not in argv


def test_an_explicit_epoch_count_is_passed_through() -> None:
    argv = build_train_command(
        TrainRequest(
            symbol="BTCUSDT", model_type="gru", epochs=8,
            start="2024-01-01", end="2024-06-30",
        )
    )[0]
    assert argv[argv.index("--epochs") + 1] == "8"


def test_cli_epoch_defaults_come_from_the_shared_module() -> None:
    from prosper.predict.defaults import DEFAULT_EPOCH_BUDGET

    assert _typer_default("gru", "--epochs") == DEFAULT_EPOCH_BUDGET["gru"]
    assert _typer_default("tft", "--max-epochs") == DEFAULT_EPOCH_BUDGET["tft"]


def test_the_full_horizon_set_is_not_spelled_out() -> None:
    """Sending every horizon on each call would make the API's argv differ
    from a plain CLI invocation for no reason."""
    argv = build_train_command(
        TrainRequest(symbol="BTCUSDT", model_type="ml", start="2024-01-01", end="2024-06-30")
    )[0]
    assert "--horizons" not in argv


def test_a_horizon_subset_reaches_the_command() -> None:
    argv = build_train_command(
        TrainRequest(
            symbol="BTCUSDT", model_type="ml",
            # Deliberately out of order, to prove the argv is canonicalised.
            horizons=[HORIZON_NAMES[-1], HORIZON_NAMES[0]],
            start="2024-01-01", end="2024-06-30",
        )
    )[0]
    expected = f"{HORIZON_NAMES[0]},{HORIZON_NAMES[-1]}"
    assert argv[argv.index("--horizons") + 1] == expected


def test_every_predictor_accepts_a_horizon_selection() -> None:
    for model_type in MODEL_TYPES:
        assert "--horizons" in _cli_options("predict", model_type), model_type


def test_the_training_form_has_no_hardcoded_epoch_value() -> None:
    """It carried `value="5"` while the CLI defaulted to 20.

    The number must come from /api/meta/training-defaults, so the form and the
    command line cannot drift apart again.
    """
    import re
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )
    field = re.search(r'id="train-epochs"[^>]*>', html)
    assert field, "train-epochs input not found"
    assert 'value="' not in field.group(), (
        "the epoch field must not carry a hardcoded default: " + field.group()
    )


def test_the_form_exposes_a_horizon_control() -> None:
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )
    assert 'id="train-horizons"' in html
    assert 'id="train-epoch-overrides"' in html


def test_the_stability_command_takes_full_iso_dates() -> None:
    """Its predecessor took YYYY-MM here and YYYY-MM-DD for its twin backtest.

    Two sibling commands over the same run disagreeing about date format is the
    kind of wart that makes a CLI untrustworthy for a thesis appendix.
    """
    for command in ("stability", "backtest"):
        assert "--start" in _cli_options("eval", command)
        assert "--end" in _cli_options("eval", command)

    click_app = typer.main.get_command(app)
    stability = click_app.commands["eval"].commands["stability"]  # type: ignore[attr-defined]
    helps = " ".join(str(getattr(p, "help", "") or "") for p in stability.params)
    assert "YYYY-MM-DD" in helps
    assert "YYYY-MM " not in helps, "the month-only format is gone"


def test_the_dead_walkforward_knobs_are_gone() -> None:
    """`train_months` was accepted and ignored; nothing retrains here."""
    accepted = _cli_options("eval", "stability")
    for dead in ("--train-months", "--train_months", "--step-months", "--step_months"):
        assert dead not in accepted, dead


def test_the_ui_exposes_the_stability_panel() -> None:
    """A CLI-only report is a report nobody reads."""
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )
    assert 'id="stability-metric"' in html
    assert 'data-action="stability-load"' in html
    assert 'id="stabilityChart"' in html


def test_the_stability_action_is_wired_to_a_handler() -> None:
    """`data-action` is dispatched by one delegated listener; an attribute with
    no entry in the registry is a dead button."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "frontend"
    main_js = (root / "js" / "main.js").read_text(encoding="utf-8")
    assert '"stability-load": () => loadStability()' in main_js
    assert "loadStability" in main_js
    assert "onStabilityMetricChange" in main_js

    evaluation_js = (root / "js" / "tabs" / "evaluation.js").read_text(encoding="utf-8")
    assert "export async function loadStability" in evaluation_js


def test_the_sharpness_bar_does_not_hardcode_three_classes() -> None:
    """Direction has two classes; ln(3) credited a coin flip with 0.37 of the
    sharpness it should earn none of. Same defect as the Python side had."""
    from pathlib import Path

    evaluation_js = (
        Path(__file__).resolve().parents[1] / "frontend" / "js" / "tabs" / "evaluation.js"
    ).read_text(encoding="utf-8")
    assert "Math.log(3)" not in evaluation_js
    assert "n_classes" in evaluation_js
