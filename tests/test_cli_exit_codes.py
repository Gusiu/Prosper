"""CLI exit-code behavior for pipeline commands."""

from prosper.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_backfill_partial_success_exits_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        "prosper.cli.backfill",
        lambda **kwargs: {"processed": 2, "failed": 1, "total_months": 3},
    )

    result = runner.invoke(
        app,
        [
            "backfill",
            "--symbol",
            "BTCUSDT",
            "--start",
            "2024-01",
            "--end",
            "2024-03",
            "--root",
            "./data",
        ],
    )

    assert result.exit_code == 0, result.stdout


def test_backfill_zero_processed_exits_one(monkeypatch) -> None:
    monkeypatch.setattr(
        "prosper.cli.backfill",
        lambda **kwargs: {"processed": 0, "failed": 3, "total_months": 3},
    )

    result = runner.invoke(
        app,
        [
            "backfill",
            "--symbol",
            "BTCUSDT",
            "--start",
            "2024-01",
            "--end",
            "2024-03",
            "--root",
            "./data",
        ],
    )

    assert result.exit_code == 1


def test_aggregate_partial_errors_exits_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        "prosper.cli.aggregate",
        lambda **kwargs: {
            "processed_months": ["2024-01"],
            "errors": ["2024-02: missing source"],
        },
    )

    result = runner.invoke(
        app,
        [
            "aggregate",
            "--symbol",
            "BTCUSDT",
            "--from",
            "1m",
            "--to",
            "1d",
            "--start",
            "2024-01",
            "--end",
            "2024-02",
            "--root",
            "./data",
        ],
    )

    assert result.exit_code == 0, result.stdout


def test_aggregate_all_failed_exits_one(monkeypatch) -> None:
    monkeypatch.setattr(
        "prosper.cli.aggregate",
        lambda **kwargs: {
            "processed_months": [],
            "errors": ["2024-01: failed", "2024-02: failed"],
        },
    )

    result = runner.invoke(
        app,
        [
            "aggregate",
            "--symbol",
            "BTCUSDT",
            "--from",
            "1m",
            "--to",
            "1d",
            "--start",
            "2024-01",
            "--end",
            "2024-02",
            "--root",
            "./data",
        ],
    )

    assert result.exit_code == 1


def test_aggregate_no_work_no_errors_exits_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        "prosper.cli.aggregate",
        lambda **kwargs: {"processed_months": [], "errors": []},
    )

    result = runner.invoke(
        app,
        [
            "aggregate",
            "--symbol",
            "BTCUSDT",
            "--from",
            "1m",
            "--to",
            "1d",
            "--start",
            "2024-01",
            "--end",
            "2024-02",
            "--root",
            "./data",
        ],
    )

    assert result.exit_code == 0, result.stdout
