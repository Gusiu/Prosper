"""The API must not send a flag the CLI does not have, or omit one the caller chose.

`api/commands.py` is the only place argv is built, and the rule it exists to
enforce is that a parameter the caller did not choose is *left out* — Typer's
default only applies to an absent flag, which is how the dashboard once trained 5
epochs where the CLI trained 20.

Pooling adds a second trap: `--symbols` exists on `predict ml` and `predict
xgboost` only. Sending it to `predict gru` would not be ignored, it would fail
the run.
"""

from __future__ import annotations

import pytest
from prosper.api.commands import build_train_command
from prosper.api.models import TrainRequest


def _argv(**kwargs) -> list[str]:
    request = TrainRequest(symbol="ETHUSDT", start="2024-01-01", end="2026-04-30", **kwargs)
    commands = build_train_command(request)
    assert len(commands) == 1
    return commands[0]


def test_no_pool_flag_when_the_caller_named_no_symbols() -> None:
    assert "--symbols" not in _argv(model_type="xgboost")
    assert "--symbols" not in _argv(model_type="ml")


def test_the_pool_reaches_the_tabular_predictors() -> None:
    argv = _argv(model_type="xgboost", pool_symbols=["BTCUSDT", "SOLUSDT"])
    assert argv[argv.index("--symbols") + 1] == "BTCUSDT,SOLUSDT"

    argv = _argv(model_type="ml", pool_symbols=["btcusdt"])
    assert argv[argv.index("--symbols") + 1] == "BTCUSDT"


@pytest.mark.parametrize("model_type", ["gru", "tft", "baseline"])
def test_the_pool_never_reaches_a_predictor_without_the_flag(model_type: str) -> None:
    """These would fail the run rather than ignore it."""
    argv = _argv(model_type=model_type, pool_symbols=["BTCUSDT"])
    assert "--symbols" not in argv


def test_the_predicted_symbol_is_not_repeated_into_its_own_pool() -> None:
    argv = _argv(model_type="xgboost", pool_symbols=["ETHUSDT", "BTCUSDT", "BTCUSDT"])
    assert argv[argv.index("--symbols") + 1] == "BTCUSDT"


def test_a_malformed_symbol_never_reaches_argv() -> None:
    """This string becomes a command-line argument, so it is validated the same
    way `--symbol` is rather than trusted."""
    argv = _argv(model_type="xgboost", pool_symbols=["BTC;rm -rf /", "BTCUSDT"])
    assert argv[argv.index("--symbols") + 1] == "BTCUSDT"

    argv = _argv(model_type="xgboost", pool_symbols=["../../etc/passwd"])
    assert "--symbols" not in argv


def test_the_window_override_is_omitted_unless_chosen() -> None:
    """The width now follows from the horizon and the history; sending an
    unasked-for value would override a decision the data made."""
    assert "--train-window-per-horizon" not in _argv(model_type="xgboost")

    argv = _argv(model_type="xgboost", train_window_by_horizon={"year": 1456})
    assert argv[argv.index("--train-window-per-horizon") + 1] == "year=1456"


def test_an_unknown_horizon_name_is_dropped_from_the_override() -> None:
    argv = _argv(model_type="xgboost", train_window_by_horizon={"decade": 3650})
    assert "--train-window-per-horizon" not in argv
