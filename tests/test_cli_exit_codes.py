"""CLI exit-code behavior for pipeline commands."""

from typer.testing import CliRunner

from prosper.cli import app

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


def test_a_failure_keeps_its_traceback(monkeypatch, tmp_path) -> None:
    """Commands printed only the exception message, so an unexpected failure was
    one line with no location - and the dashboard's task log showed only that."""

    def explode(**kwargs):
        raise KeyError("missing_column")

    monkeypatch.setattr("prosper.cli.build_labels", explode)
    result = runner.invoke(
        app, ["labels", "build", "--symbol", "BTCUSDT", "--root", str(tmp_path)]
    )

    assert result.exit_code == 1
    # CliRunner mixes stderr into the output, where the traceback lands.
    assert "Traceback" in result.output and "missing_column" in result.output


def test_a_deliberate_exit_is_not_reported_twice(tmp_path) -> None:
    """The broad handler caught `typer.Exit(1)` raised inside the try and printed
    a second line, "Error: 1", after the real message."""
    result = runner.invoke(
        app, ["labels", "build", "--symbol", "NOPEUSDT", "--root", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "No data found" in result.stdout
    assert "Error: 1" not in result.stdout
