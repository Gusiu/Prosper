"""Every queued job works on the data lake the server was started with.

The in-process endpoints read `get_settings().data_root`, which
`PROSPER_DATA_ROOT` can point anywhere, while `api/commands.py` wrote a literal
`--root ./data` into every argv. A server started against a scratch copy of the
lake therefore showed the copy and trained into the real one: two training runs
landed in the study's lake that way. The CLI takes its root from `--root` alone,
so the argv is the only place the server's root can reach a queued job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prosper.api.commands import (
    build_aggregate_command,
    build_batch_eval_command,
    build_eval_command,
    build_pipeline_command,
    build_repair_commands,
    build_train_command,
)
from prosper.api.models import (
    AggregateRequest,
    BatchEvalRequest,
    EvalRequest,
    PipelineRequest,
    RepairRequest,
    TrainRequest,
)
from prosper.config import get_settings

MODEL_TYPES = ("baseline", "ml", "xgboost", "gru", "tft")


@pytest.fixture
def scratch_root(tmp_path, monkeypatch) -> Path:
    """Configure the server the way the scratch copy was: through the environment.

    The cached settings are dropped so the next `get_settings()` reads the
    variable, exactly as a freshly started server does; monkeypatch restores
    the cache afterwards.
    """
    root = tmp_path / "scratch" / "data"
    monkeypatch.setenv("PROSPER_DATA_ROOT", str(root))
    monkeypatch.setattr("prosper.config._settings", None)
    return root


def _every_command() -> list[list[str]]:
    """One argv from each branch of every builder."""
    commands: list[list[str]] = []
    commands += build_pipeline_command(
        PipelineRequest(symbol="BTCUSDT", start="2024-01-01", end="2024-06-30", intervals=["1h"])
    )
    for model_type in MODEL_TYPES:
        commands += build_train_command(
            TrainRequest(
                symbol="BTCUSDT", model_type=model_type, start="2024-01-01", end="2024-06-30"
            )
        )
    commands += build_eval_command(
        EvalRequest(symbol="BTCUSDT", model_type="xgboost", timestamp="20240501123000")
    )
    commands += build_batch_eval_command(BatchEvalRequest())
    commands += build_aggregate_command(
        AggregateRequest(symbol="BTCUSDT", start="2024-01-01", end="2024-06-30", intervals=["1h"])
    )
    inventory = [
        {
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            # A gapped base, a broken aggregate and a missing feature set: every
            # step the repair chain can contain.
            "aggregations": [
                {"interval": "1m", "status": "incomplete", "gaps": 3},
                {"interval": "1h", "status": "incomplete", "gaps": 2},
                {"interval": "1d", "status": "warning", "gaps": 0},
            ],
        }
    ]
    repair, _ = build_repair_commands(RepairRequest(symbol="BTCUSDT"), inventory)
    commands += repair
    return commands


def _roots(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, token in enumerate(argv) if token == "--root"]


def test_every_command_carries_the_configured_root(scratch_root: Path) -> None:
    commands = _every_command()
    subcommands = {tuple(argv[3:5]) for argv in commands}
    assert len(subcommands) >= 7, f"not every builder was exercised: {sorted(subcommands)}"

    offenders = [
        " ".join(argv[3:5]) + f" -> {_roots(argv)}"
        for argv in commands
        if _roots(argv) != [str(scratch_root)]
    ]
    assert not offenders, "commands that would write elsewhere:\n  " + "\n  ".join(offenders)


def test_the_queued_jobs_and_the_endpoints_read_the_same_lake(scratch_root: Path) -> None:
    """What the routes read and what the CLI is told must be one value."""
    argv = build_train_command(
        TrainRequest(symbol="BTCUSDT", start="2024-01-01", end="2024-06-30")
    )[0]

    assert Path(_roots(argv)[0]) == get_settings().data_root == scratch_root
