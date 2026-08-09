"""CLI entrypoint using Typer."""

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from prosper.binance.rest import BinanceRESTClient
from prosper.config import get_settings
from prosper.domain import DEFAULT_DEPTH_BINS_STR, DEFAULT_HORIZONS
from prosper.eval.backtest import run_backtest
from prosper.eval.predictions import (
    collect_evaluation_summaries,
    evaluate_available_model_runs,
    evaluate_predictions,
)
from prosper.eval.stability import eval_stability
from prosper.features.build import build_features
from prosper.labels.build import build_labels
from prosper.pipeline.aggregate import aggregate, normalize_target_intervals
from prosper.pipeline.backfill import backfill
from prosper.planner.windows import plan_windows
from prosper.predict.baseline import predict_baseline as predict_baseline_3horizons
from prosper.predict.defaults import (
    DEFAULT_EPOCH_BUDGET,
    DEFAULT_TRAIN_WINDOW_DAYS,
    HORIZON_NAMES,
)
from prosper.predict.gru import predict_gru
from prosper.predict.ml import predict_ml
from prosper.predict.xgboost_model import predict_xgboost
from prosper.qa.checks import run_qa_checks

# Read the settings rather than repeat their values: a `log_level` field that
# nothing consults is worse than none, because it looks configurable.
_log_settings = get_settings()
logging.basicConfig(
    level=getattr(logging, _log_settings.log_level.upper(), logging.INFO),
    format=_log_settings.log_format,
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Repeated in three --help strings before this; the horizons are the authority.
LONGEST_HORIZON_DAYS = max(h.forward_days for h in DEFAULT_HORIZONS)

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Prosper - Crypto market analysis tool for Binance SPOT data")

symbols_app = typer.Typer(help="Symbol management")
qa_app = typer.Typer(help="Quality assurance")
labels_app = typer.Typer(help="Label building")
predict_app = typer.Typer(help="Prediction generation")
planner_app = typer.Typer(help="Action window planning")
features_app = typer.Typer(help="Feature engineering")
eval_app = typer.Typer(help="Forecast quality, stability over time, and trading evaluation")

app.add_typer(symbols_app, name="symbols")
app.add_typer(qa_app, name="qa")
app.add_typer(labels_app, name="labels")
app.add_typer(predict_app, name="predict")
app.add_typer(planner_app, name="planner")
app.add_typer(features_app, name="features")
app.add_typer(eval_app, name="eval")


def _print_epoch_usage(results: dict[str, Any]) -> None:
    """Show what early stopping actually decided.

    `--epochs` is a ceiling, so the epochs a horizon used are a result of the
    run. They were computed, returned, and then never printed — which is how
    "did early stopping fire?" stayed a question needing a bespoke probe.
    `hit_ceiling` is the one to read: a horizon reaching its budget in most
    windows is limited by the budget, not by the signal.
    """
    usage = results.get("epochs_used") or {}
    if not usage:
        return
    table = Table(title="Epochs used (--epochs is a ceiling, not a target)")
    table.add_column("horizon")
    table.add_column("windows", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("min-max", justify="right")
    table.add_column("ceiling", justify="right")
    table.add_column("hit it", justify="right")
    for name, stats in usage.items():
        windows = int(stats["windows"])
        hit = int(stats["hit_ceiling"])
        table.add_row(
            name,
            str(windows),
            f"{stats['mean']:.1f}",
            f"{int(stats['min'])}-{int(stats['max'])}",
            str(int(stats["ceiling"])),
            f"{hit} ({hit / windows:.0%})" if windows else "-",
        )
    console.print(table)


@symbols_app.command("list")
def symbols_list(
    quote_asset: str = typer.Option(
        None, "--quote-asset", help="Filter by quote asset (e.g., USDT)"
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON file path"),
) -> None:
    """
    Fetch SPOT symbols list from Binance exchangeInfo and save to data/meta/binance_spot_symbols.json.
    """
    settings = get_settings()

    try:
        with BinanceRESTClient() as client:
            symbols_list_data = client.get_symbols_spot(quote_asset=quote_asset)

        console.print(f"[green]Found {len(symbols_list_data)} symbols[/green]")

        if quote_asset:
            console.print(f"Filtered by quote asset: {quote_asset}")

        output_path = output or settings.meta_dir / "binance_spot_symbols.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"symbols": symbols_list_data, "count": len(symbols_list_data)},
                f,
                indent=2,
                sort_keys=True,
            )

        console.print(f"[green]Saved to: {output_path}[/green]")

        console.print("\n[bold]First 20 symbols:[/bold]")
        for sym in symbols_list_data[:20]:
            console.print(f"  - {sym}")

        if len(symbols_list_data) > 20:
            console.print(f"  ... and {len(symbols_list_data) - 20} more")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("backfill")
def backfill_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol (e.g., BTCUSDT)"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM format"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM format"),
    workers: int = typer.Option(
        get_settings().download_workers,
        "--workers",
        "-w",
        help="Number of parallel workers",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run mode (simulate only)"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Reprocess months that already have Parquet (needed after a schema change)",
    ),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """
    Backfill monthly data from Binance Public Data.
    """
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        summary = backfill(
            symbol=symbol,
            start=start,
            end=end,
            workers=workers,
            dry_run=dry_run,
            # Without this there is no way to pick up a widened kline schema:
            # every month already has a Parquet, so every month is skipped and
            # the new columns never appear.
            skip_existing=not force,
            settings=settings,
        )

        processed = summary.get("processed", 0)
        failed = summary.get("failed", 0)
        total = summary.get("total_months", 0)
        if failed > 0:
            console.print(f"[yellow]Warning: {failed}/{total} months had errors[/yellow]")
        if processed == 0:
            # Only fail when zero months succeeded â€” partial errors should continue the chain
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@qa_app.command("check")
def qa_check(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    interval: str = typer.Option("1m", "--interval", help="Time interval (1m, 1h, 1d)"),
    year: int = typer.Option(..., "--year", help="Year"),
    month: int = typer.Option(..., "--month", help="Month (1-12)"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """
    Run QA checks on data (sorting, duplicates, gaps, OHLC invariants).
    """
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        results = run_qa_checks(symbol, interval, year, month, settings=settings)

        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]QA Results for {symbol} {interval} {year}-{month:02d}[/bold]")
        console.print(f"Row count: {results['row_count']}")

        checks = results["checks"]
        for check_name, check_result in checks.items():
            status = "[OK]" if check_result.get("passed", False) else "[FAIL]"
            color = "green" if check_result.get("passed", False) else "red"
            console.print(f"[{color}]{status}[/{color}] {check_name}: {check_result}")

        if results.get("overall_passed", False):
            console.print("\n[green]All checks passed![/green]")
        else:
            console.print("\n[red]Some checks failed![/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("aggregate")
def aggregate_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    from_interval: str = typer.Option("1m", "--from", help="Source interval"),
    to_intervals: str = typer.Option(
        ...,
        "--to",
        help="Target intervals, comma-separated (e.g. 1h,1d,1w)",
    ),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM format"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM format"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
    extra_intervals: list[str] | None = typer.Argument(
        None,
        help="Additional target intervals (used only for syntax: --to 1h 1d 1w).",
    ),
) -> None:
    """
    Aggregate klines from 1m to 1h, 1d, 1w.
    """
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)
    to_list = normalize_target_intervals([to_intervals, *(extra_intervals or [])])

    try:
        # Robust date handling: truncate YYYY-MM-DD to YYYY-MM if needed
        clean_start = start[:7] if len(start) >= 7 else start
        clean_end = end[:7] if len(end) >= 7 else end

        results = aggregate(
            symbol=symbol,
            from_interval=from_interval,
            to_intervals=to_list,
            start=clean_start,
            end=clean_end,
            settings=settings,
        )

        console.print("\n[bold]Aggregation Results[/bold]")
        console.print(f"Processed months: {len(results.get('processed_months', []))}")
        console.print(f"Errors: {len(results.get('errors', []))}")

        errors = results.get("errors", [])
        processed = results.get("processed_months", [])
        if errors:
            console.print("[yellow]Warnings:[/yellow]")
            for error in errors:
                console.print(f"  - {error}")

        if errors and not processed:
            # Only fail if there were ACTUAL errors and ZERO success.
            # If 0 processed but 0 errors, it just means data was already there - CONTINUE.
            raise typer.Exit(1)

        if not errors and not processed:
            console.print(
                "[blue]Info: No new months to aggregate (data already exists or range empty). Continuing...[/blue]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@labels_app.command("build")
def labels_build(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    base_interval: str = typer.Option(
        "1d",
        "--base-interval",
        "--base_interval",
        help="Base interval for labels",
    ),
    forward_days: int = typer.Option(
        1,
        "--forward-days",
        "--forward_days",
        help="Forward horizon in days (N) for r_fwd",
    ),
    depth_bins: str = typer.Option(
        DEFAULT_DEPTH_BINS_STR,
        "--depth-bins",
        help="Depth bin definitions",
    ),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """
    Build direction (long/short) and depth labels from daily klines.
    """
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        stats = build_labels(
            symbol=symbol,
            base_interval=base_interval,
            forward_days=forward_days,
            depth_bins_str=depth_bins,
            settings=settings,
        )

        if "error" in stats:
            console.print(f"[red]Error: {stats['error']}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]Label Statistics for {symbol}[/bold]")
        console.print(f"Total labels: {stats['total_labels']}")
        console.print("\nDirection distribution:")
        for direction, count in stats["direction_distribution"].items():
            pct = stats["direction_percentages"][direction]
            console.print(f"  {direction}: {count} ({pct:.2f}%)")

        console.print(f"\nImbalance ratio: {stats['imbalance_ratio']:.2f}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@features_app.command("build")
def features_build(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    base_interval: str = typer.Option(
        "1d",
        "--base-interval",
        "--base_interval",
        help="Base interval for features (MVP: 1d)",
    ),
    start: str = typer.Option(None, "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(None, "--end", help="End date in YYYY-MM-DD format"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Build engineered features parquet."""
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        results = build_features(
            symbol=symbol,
            base_interval=base_interval,
            start=start,
            end=end,
            settings=settings,
        )
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)

        console.print(f"[green][OK][/green] Feature parquet saved: {results['path']}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@predict_app.command("baseline")
def predict_baseline_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD format"),
    interval: str = typer.Option("1d", "--interval", help="Bar interval (1d, 1h, ...)"),
    horizons: str = typer.Option(
        None,
        "--horizons",
        help="Comma-separated horizons to train (default: all). Unlisted ones are written as untrained and skipped downstream.",
    ),
    window_days: int = typer.Option(
        DEFAULT_TRAIN_WINDOW_DAYS,
        "--window-days",
        "--window_days",
        help=f"Rolling window in calendar days; must exceed the longest horizon ({LONGEST_HORIZON_DAYS}d)",
    ),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    deterministic: bool = typer.Option(
        False, "--deterministic", help="Enable deterministic training"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Generate baseline probabilistic predictions for short/medium/long horizons."""
    settings = get_settings(
        data_root=root,
        strict=strict,
        deterministic=deterministic,
        seed=seed,
        save_metadata=save_metadata,
    )

    try:
        results = predict_baseline_3horizons(
            symbol=symbol,
            start=start,
            end=end,
            settings=settings,
            interval=interval,
            horizons_selected=horizons,
            rolling_window_days=window_days,
        )
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)

        console.print(f"[green][OK][/green] Predictions generated: {results['predictions']}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@predict_app.command("ml")
def predict_ml_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD format"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Feature interval (1m, 1h, 1d)"),
    horizons: str = typer.Option(
        None,
        "--horizons",
        help="Comma-separated horizons to train (default: all). Unlisted ones are written as untrained and skipped downstream.",
    ),
    train_window_days: int = typer.Option(
        DEFAULT_TRAIN_WINDOW_DAYS,
        "--train-window-days",
        "--train_window_days",
        help=f"Training window in calendar days; must exceed the longest horizon ({LONGEST_HORIZON_DAYS}d)",
    ),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    deterministic: bool = typer.Option(
        False, "--deterministic", help="Enable deterministic training"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Generate ML probabilistic predictions for short/medium/long horizons."""
    settings = get_settings(
        data_root=root,
        strict=strict,
        deterministic=deterministic,
        seed=seed,
        save_metadata=save_metadata,
    )

    try:
        results = predict_ml(
            symbol=symbol,
            start=start,
            end=end,
            settings=settings,
            interval=interval,
            horizons_selected=horizons,
            train_window_days=train_window_days,
        )
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)

        console.print(f"[green][OK][/green] ML Predictions generated: {results['predictions']}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@predict_app.command("xgboost")
def predict_xgboost_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD format"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Feature interval (1m, 1h, 1d)"),
    horizons: str = typer.Option(
        None,
        "--horizons",
        help="Comma-separated horizons to train (default: all). Unlisted ones are written as untrained and skipped downstream.",
    ),
    train_window_days: int = typer.Option(
        DEFAULT_TRAIN_WINDOW_DAYS,
        "--train-window-days",
        "--train_window_days",
        help=f"Training window in calendar days; must exceed the longest horizon ({LONGEST_HORIZON_DAYS}d)",
    ),
    n_estimators: int = typer.Option(100, "--n-estimators", help="XGBoost n_estimators"),
    max_depth: int = typer.Option(6, "--max-depth", help="XGBoost max_depth"),
    learning_rate: float = typer.Option(0.1, "--learning-rate", help="XGBoost learning rate"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    deterministic: bool = typer.Option(
        False, "--deterministic", help="Enable deterministic training"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Generate XGBoost probabilistic predictions for short/medium/long horizons."""
    settings = get_settings(
        data_root=root,
        strict=strict,
        deterministic=deterministic,
        seed=seed,
        save_metadata=save_metadata,
    )

    try:
        results = predict_xgboost(
            symbol=symbol,
            start=start,
            end=end,
            settings=settings,
            interval=interval,
            horizons_selected=horizons,
            train_window_days=train_window_days,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)

        console.print(
            f"[green][OK][/green] XGBoost Predictions generated: {results['predictions']}"
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@predict_app.command("gru")
def predict_gru_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD format"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Feature interval (1m, 1h, 1d)"),
    horizons: str = typer.Option(
        None,
        "--horizons",
        help="Comma-separated horizons to train (default: all). Unlisted ones are written as untrained and skipped downstream.",
    ),
    train_window_days: int = typer.Option(
        DEFAULT_TRAIN_WINDOW_DAYS,
        "--train-window-days",
        help="Training window in calendar days; must exceed the longest horizon",
    ),
    seq_len: int = typer.Option(30, "--seq-len", help="Sequence length for GRU input"),
    epochs: int = typer.Option(
        DEFAULT_EPOCH_BUDGET["gru"],
        "--epochs",
        help="Epoch budget per window; early stopping usually ends training sooner",
    ),
    epochs_by_horizon: str = typer.Option(
        None,
        "--epochs-per-horizon",
        help="Per-horizon epoch ceilings, e.g. 'short=20,medium=5'. Overrides the budget.",
    ),
    hidden_size: int = typer.Option(64, "--hidden-size", help="GRU hidden layer size"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    deterministic: bool = typer.Option(
        False, "--deterministic", help="Enable deterministic training"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Generate GRU (Deep Learning) probabilistic predictions for short/medium/long horizons."""
    settings = get_settings(
        data_root=root,
        strict=strict,
        deterministic=deterministic,
        seed=seed,
        save_metadata=save_metadata,
    )
    try:
        results = predict_gru(
            symbol=symbol,
            start=start,
            end=end,
            settings=settings,
            interval=interval,
            seq_len=seq_len,
            horizons_selected=horizons,
            train_window_days=train_window_days,
            epochs=epochs,
            epochs_by_horizon=epochs_by_horizon,
            hidden_size=hidden_size,
        )
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green][OK][/green] GRU Predictions generated: {results['predictions']} rows (device: {results.get('device', 'cpu')})"
        )
        _print_epoch_usage(results)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@predict_app.command("tft")
def predict_tft_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD format"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Feature interval (1m, 1h, 1d)"),
    horizons: str = typer.Option(
        None,
        "--horizons",
        help="Comma-separated horizons to train (default: all). Unlisted ones are written as untrained and skipped downstream.",
    ),
    train_window_days: int = typer.Option(
        DEFAULT_TRAIN_WINDOW_DAYS,
        "--train-window-days",
        help="Training window in calendar days; must exceed the longest horizon",
    ),
    seq_len: int = typer.Option(60, "--seq-len", help="Encoder sequence length"),
    max_epochs: int = typer.Option(
        DEFAULT_EPOCH_BUDGET["tft"],
        "--max-epochs",
        help="Epoch budget per window; early stopping usually ends training sooner",
    ),
    epochs_by_horizon: str = typer.Option(
        None,
        "--epochs-per-horizon",
        help="Per-horizon epoch ceilings, e.g. 'short=20,medium=5'. Overrides the budget.",
    ),
    hidden_size: int = typer.Option(32, "--hidden-size", help="TFT hidden layer size"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    deterministic: bool = typer.Option(
        False, "--deterministic", help="Enable deterministic training"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Generate TFT (Temporal Fusion Transformer) probabilistic predictions."""
    from prosper.predict.tft import predict_tft

    settings = get_settings(
        data_root=root,
        strict=strict,
        deterministic=deterministic,
        seed=seed,
        save_metadata=save_metadata,
    )
    try:
        results = predict_tft(
            symbol=symbol,
            start=start,
            end=end,
            settings=settings,
            interval=interval,
            seq_len=seq_len,
            horizons_selected=horizons,
            train_window_days=train_window_days,
            max_epochs=max_epochs,
            epochs_by_horizon=epochs_by_horizon,
            hidden_size=hidden_size,
        )
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green][OK][/green] TFT Predictions generated: {results['predictions']} rows"
        )
        _print_epoch_usage(results)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@planner_app.command("windows")
def planner_windows(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(None, "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(None, "--end", help="End date in YYYY-MM-DD format"),
    model_type: str = typer.Option(
        None, "--model-type", "--model_type", help="Source model; latest run if omitted"
    ),
    timestamp: str = typer.Option(
        None, "--timestamp", help="Source run timestamp; latest run if omitted"
    ),
    interval: str = typer.Option(None, "--interval", help="Source run interval"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """
    Plan action windows and generate recommendations (Strong Buy ... Strong Sell).
    """
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:

        results = plan_windows(
            symbol=symbol,
            start=start,
            end=end,
            settings=settings,
            model_type=model_type,
            timestamp=timestamp,
            interval=interval,
        )

        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)

        run = results.get("run", {})
        console.print(f"\n[bold]Action Windows for {symbol}[/bold]")
        console.print(f"Source run: {run.get('slug', 'unknown')}")
        windows_all = results.get("windows", []) or []
        console.print(f"Total windows: {len(windows_all)}")
        skipped = results.get("skipped_untrained_horizons", 0)
        if skipped:
            console.print(f"[yellow]Skipped {skipped} untrained horizon rows[/yellow]")

        console.print("\n[bold]Sample windows:[/bold]")
        for window in windows_all[:5]:
            diag = window.get("diagnostics", {})
            edge_mean = diag.get("edge_mean", 0.0)
            risk_mean = diag.get("risk_mean", 0.0)
            console.print(
                f"  {window['start_date']} to {window['end_date']} ({window.get('horizon')}): "
                f"{window['recommendation']} (edge_mean={edge_mean:.3f}, risk_mean={risk_mean:.3f})"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command("stability")
def eval_stability_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD"),
    model_type: str = typer.Option(
        None, "--model-type", "--model_type", help="Source model; latest run if omitted"
    ),
    timestamp: str = typer.Option(
        None, "--timestamp", help="Source run timestamp; latest run if omitted"
    ),
    interval: str = typer.Option(None, "--interval", help="Source run interval"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Track one run's forecast quality month by month, to see whether it holds up."""
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        report = eval_stability(
            symbol=symbol,
            start=start,
            end=end,
            settings=settings,
            model_type=model_type,
            timestamp=timestamp,
            interval=interval,
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green][OK][/green] Stability report saved to: {report['report_path']}")
    console.print(f"Source run: {report['run']['slug']}  months: {len(report['months'])}")

    table = Table(title="Quality over time (mean logloss; lower is better)")
    table.add_column("horizon")
    table.add_column("samples", justify="right")
    table.add_column("overall", justify="right")
    table.add_column("best month", justify="right")
    table.add_column("worst month", justify="right")
    for name, overall in report["overall"].items():
        if not overall["samples"]:
            table.add_row(name, "0", "-", "-", "-")
            continue
        scored = [
            (m["month"], m["horizons"][name]["logloss"])
            for m in report["months"]
            if m["horizons"][name]["samples"] and not m["horizons"][name]["sparse"]
        ]
        best = min(scored, key=lambda r: r[1]) if scored else None
        worst = max(scored, key=lambda r: r[1]) if scored else None
        table.add_row(
            name,
            str(overall["samples"]),
            f"{overall['logloss']:.3f}",
            f"{best[1]:.3f} ({best[0]})" if best else "-",
            f"{worst[1]:.3f} ({worst[0]})" if worst else "-",
        )
    console.print(table)


@eval_app.command("backtest")
def eval_backtest_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD"),
    capital: float = typer.Option(10000.0, "--capital", help="Initial Capital in USDT"),
    model_type: str = typer.Option(
        None, "--model-type", "--model_type", help="Source model; latest run if omitted"
    ),
    timestamp: str = typer.Option(
        None, "--timestamp", help="Source run timestamp; latest run if omitted"
    ),
    interval: str = typer.Option(None, "--interval", help="Source run interval"),
    horizon: str = typer.Option("short", "--horizon", help="Horizon driving the signal"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Run Trading Simulator (Backtest) over one versioned prediction run."""
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        res = run_backtest(
            symbol=symbol,
            start=start,
            end=end,
            initial_capital=capital,
            settings=settings,
            model_type=model_type,
            timestamp=timestamp,
            interval=interval,
            horizon=horizon,
            save_report=True,
        )

        if "error" in res:
            console.print(f"[red]Error: {res['error']}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]Backtest Results for {symbol}[/bold]")
        console.print(f"Source run:      {res['run']['slug']} (horizon: {res['horizon']})")
        console.print(f"Initial Capital: ${res['initial_capital']:.2f}")
        console.print(f"Final Value:     ${res['final_value']:.2f}")

        color = "green" if res["roi_pct"] >= 0 else "red"
        console.print(f"ROI:             [{color}]{res['roi_pct']:.2f}%[/{color}]")
        console.print(f"Max Drawdown:    [red]-{res['max_drawdown_pct']:.2f}%[/red]")
        sharpe = res.get("sharpe")
        console.print(
            f"Sharpe:          {sharpe:.2f}" if sharpe is not None else "Sharpe:          n/a"
        )
        console.print(f"Total Trades:    {res['total_trades']}")
        console.print(f"Turnover:        {res['turnover_ratio']:.2f}x initial capital")
        console.print(f"Time in market:  {res['time_in_market_pct']:.1f}%")
        console.print(f"Days Tested:     {res['days_tested']}")

        bench = res["benchmark"]
        bench_color = "green" if bench["roi_pct"] >= 0 else "red"
        excess_color = "green" if bench["excess_roi_pct"] >= 0 else "red"
        console.print(
            f"\nBuy & hold ROI:  [{bench_color}]{bench['roi_pct']:.2f}%[/{bench_color}]"
        )
        console.print(
            f"Excess vs hold:  [{excess_color}]{bench['excess_roi_pct']:+.2f}%[/{excess_color}]"
        )

        # Beating buy & hold is the weakest of four claims; printing only that
        # one is how a reduced-beta position gets mistaken for a forecast.
        nulls = res.get("nulls") or {}
        if nulls:
            table = Table(title="Did the model add anything?")
            table.add_column("comparison")
            table.add_column("result", justify="right")
            table.add_column("what it rules out")

            constant = nulls["constant_exposure"]
            colour = "green" if constant["excess_roi_pct"] >= 0 else "red"
            table.add_row(
                f"constant {100 * constant['mean_exposure']:.0f}% exposure",
                f"[{colour}]{constant['excess_roi_pct']:+.2f} pp[/{colour}]",
                "carrying less beta",
            )

            timing = nulls["mistimed_replay"]
            if timing.get("percentile") is not None:
                # 50 is chance; the >=95 convention, and with many runs examined
                # even that is generous.
                colour = "green" if timing["percentile"] >= 95 else "yellow"
                table.add_row(
                    "vs its own mistimed copies",
                    f"[{colour}]{timing['percentile']:.1f} pctile[/{colour}]",
                    f"volatility timing ({timing['samples']} shifts)",
                )

            momentum = nulls["naive_momentum"]
            if momentum.get("roi_pct") is not None:
                replay = timing.get("replay_roi_pct")
                delta = None if replay is None else replay - momentum["roi_pct"]
                colour = "green" if (delta or 0) >= 0 else "red"
                table.add_row(
                    f"{momentum['lookback_bars']}-bar momentum rule",
                    "n/a" if delta is None else f"[{colour}]{delta:+.2f} pp[/{colour}]",
                    "a trend rule with no model in it",
                )
            console.print()
            console.print(table)

        console.print(f"\nReport:          {res.get('report_path', 'n/a')}")
        console.print("\n[yellow]Execution assumptions:[/yellow]")
        for note in res["assumptions"]:
            console.print(f"  - {note}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _run_predictions_evaluation(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str,
    root: Path,
    strict: bool,
    save_metadata: bool,
) -> None:
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        result = evaluate_predictions(
            symbol=symbol,
            model_type=model_type,
            timestamp=timestamp,
            interval=interval,
            settings=settings,
        )
        metrics = result["metrics"]
        artifacts = result["artifacts"]
        score = metrics.get("overall", {}).get("model_score")
        score_text = f"{score:.2f}" if isinstance(score, int | float) else "N/A"

        console.print(f"[green][OK][/green] Prediction evaluation saved to: {metrics['artifacts_dir']}")
        console.print(f"Model score: {score_text}")
        console.print(f"Scored rows: {metrics.get('scored_rows', 0)}")
        console.print(f"Metrics: {artifacts['metrics']}")
        console.print(f"Quality rows: {artifacts['predictions_quality']}")
        console.print(f"Recommendations: {artifacts['recommendations']}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command("predictions")
def eval_predictions_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    model_type: str = typer.Option(..., "--model-type", "--model_type", help="Model type"),
    timestamp: str = typer.Option(..., "--timestamp", help="Versioned prediction timestamp"),
    interval: str = typer.Option("1d", "--interval", help="Prediction/data interval"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Evaluate a versioned prediction run and generate quality/recommendation artifacts."""
    _run_predictions_evaluation(symbol, model_type, timestamp, interval, root, strict, save_metadata)


@eval_app.command("compare")
def eval_compare_cmd(
    symbol: str | None = typer.Option(None, "--symbol", help="Optional symbol filter"),
    interval: str | None = typer.Option(None, "--interval", help="Optional interval filter"),
    output: Path = typer.Option(None, "--output", "-o", help="Write the table as CSV"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
) -> None:
    """Rank every evaluated run side by side."""
    settings = get_settings(data_root=root)
    rows = collect_evaluation_summaries(symbol=symbol, interval=interval, settings=settings)

    if not rows:
        console.print("[yellow]No evaluations found. Run `prosper eval predictions` first.[/yellow]")
        raise typer.Exit(1)

    table = Table(title="Model comparison", header_style="bold")
    for column in ("#", "Symbol", "Model", "Int.", "Run", "Score", "Acc", "Brier", "ECE"):
        table.add_column(column)
    table.add_column("Untrained", justify="right")
    for column in HORIZON_NAMES:
        table.add_column(f"acc {column}", justify="right")

    def fmt(value: Any, digits: int = 3) -> str:
        return f"{value:.{digits}f}" if isinstance(value, int | float) else "--"

    for rank, row in enumerate(rows, start=1):
        untrained = int(row.get("untrained_rows") or 0)
        table.add_row(
            str(rank),
            str(row["symbol"]),
            str(row["model_type"]),
            str(row["interval"]),
            str(row["timestamp"]),
            fmt(row["model_score"], 1),
            fmt(row["accuracy"]),
            fmt(row["brier"]),
            fmt(row["ece"]),
            f"[red]{untrained}[/red]" if untrained else "0",
            fmt(row.get("acc_short")),
            fmt(row.get("acc_medium")),
            fmt(row.get("acc_long")),
        )

    console.print(table)
    if any(int(row.get("untrained_rows") or 0) for row in rows):
        console.print(
            "[yellow]Runs with untrained rows are not comparable: those horizons "
            "produced no forecast and contribute no samples.[/yellow]"
        )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_csv(output)
        console.print(f"[green][OK][/green] Table written to: {output}")


@eval_app.command("batch")
def eval_batch_cmd(
    symbol: str | None = typer.Option(None, "--symbol", help="Optional symbol filter"),
    model_type: str | None = typer.Option(None, "--model-type", "--model_type", help="Optional model filter"),
    interval: str | None = typer.Option(None, "--interval", help="Optional interval filter"),
    limit: int | None = typer.Option(None, "--limit", help="Maximum runs to evaluate"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Evaluate available model runs in bulk; useful for cron/scheduler jobs."""
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)
    try:
        result = evaluate_available_model_runs(
            symbol=symbol,
            model_type=model_type,
            interval=interval,
            limit=limit,
            settings=settings,
        )
        console.print(
            f"[green][OK][/green] Batch evaluation complete: "
            f"{result.get('evaluated', 0)} ok, {result.get('failed', 0)} failed"
        )
        if result.get("status_path"):
            console.print(f"Status: {result['status_path']}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("ui")
def start_ui(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the UI server on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host IP to bind to"),
    open_browser: bool = typer.Option(
        True, "--open-browser/--no-open-browser", help="Open the dashboard on startup"
    ),
) -> None:
    """
    Start the Prosper Data Manager Web UI (Dashboard).
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Uvicorn is not installed. Please install it with: poetry add uvicorn[/red]"
        )
        raise typer.Exit(1)

    url = f"http://{host}:{port}"
    if open_browser:
        # Launching a browser is a property of this command, not of the ASGI
        # app, which also runs under tests and process managers.
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    console.print(f"[green]Starting Prosper UI on {url}[/green]")
    uvicorn.run("prosper.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()



