"""CLI entrypoint using Typer."""

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from prosper.binance.rest import BinanceRESTClient
from prosper.config import get_settings
from prosper.eval.backtest import run_backtest
from prosper.eval.walkforward import eval_walkforward
from prosper.features.build import build_features
from prosper.labels.build import build_labels
from prosper.pipeline.aggregate import aggregate, normalize_target_intervals
from prosper.pipeline.backfill import backfill
from prosper.planner.windows import plan_windows
from prosper.predict.baseline import predict_baseline as predict_baseline_3horizons
from prosper.predict.gru import predict_gru
from prosper.predict.ml import predict_ml
from prosper.qa.checks import run_qa_checks
from prosper.storage.layout import get_eval_walkforward_summary_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Prosper - Crypto market analysis tool for Binance SPOT data")

symbols_app = typer.Typer(help="Symbol management")
qa_app = typer.Typer(help="Quality assurance")
labels_app = typer.Typer(help="Label building")
predict_app = typer.Typer(help="Prediction generation")
planner_app = typer.Typer(help="Action window planning")
features_app = typer.Typer(help="Feature engineering")
eval_app = typer.Typer(help="Walk-forward evaluation")

app.add_typer(symbols_app, name="symbols")
app.add_typer(qa_app, name="qa")
app.add_typer(labels_app, name="labels")
app.add_typer(predict_app, name="predict")
app.add_typer(planner_app, name="planner")
app.add_typer(features_app, name="features")
app.add_typer(eval_app, name="eval")


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
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry run mode (simulate only)"),
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
            skip_existing=True,
            settings=settings,
        )

        total = summary.get("total", 0)
        failed = summary.get("failed", 0)
        if failed > 0:
            console.print(f"[yellow]Warning: {failed}/{total} months had errors[/yellow]")
        if total > 0 and failed >= total:
            # Only fail if ALL months failed — partial success should continue the chain
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
    flat_threshold: float = typer.Option(
        0.01, "--flat-threshold", help="Flat threshold (e.g., 0.01 = 1%)"
    ),
    depth_bins: str = typer.Option(
        "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+",
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
    Build direction (long/flat/short) and depth labels from daily klines.
    """
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        stats = build_labels(
            symbol=symbol,
            base_interval=base_interval,
            forward_days=forward_days,
            flat_threshold=flat_threshold,
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
    window_days: int = typer.Option(
        180,
        "--window-days",
        "--window_days",
        help="Rolling window size in days (used for empirical frequencies)",
    ),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Generate baseline probabilistic predictions for short/medium/long horizons."""
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        results = predict_baseline_3horizons(
            symbol=symbol,
            start=start,
            end=end,
            settings=settings,
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
    train_window_days: int = typer.Option(
        150,
        "--train-window-days",
        "--train_window_days",
        help="Training window size in days",
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
            train_window_days=train_window_days,
        )
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)

        console.print(f"[green][OK][/green] ML Predictions generated: {results['predictions']}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@predict_app.command("gru")
def predict_gru_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD format"),
    train_window_days: int = typer.Option(
        365, "--train-window-days", help="Training window size in days"
    ),
    seq_len: int = typer.Option(30, "--seq-len", help="Sequence length for GRU input"),
    epochs: int = typer.Option(20, "--epochs", help="Training epochs per window"),
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
            seq_len=seq_len,
            train_window_days=train_window_days,
            epochs=epochs,
            hidden_size=hidden_size,
        )
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green][OK][/green] GRU Predictions generated: {results['predictions']} rows (device: {results.get('device', 'cpu')})"
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@predict_app.command("tft")
def predict_tft_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD format"),
    train_window_days: int = typer.Option(
        365, "--train-window-days", help="Training window size in days"
    ),
    seq_len: int = typer.Option(60, "--seq-len", help="Encoder sequence length"),
    max_epochs: int = typer.Option(10, "--max-epochs", help="Training epochs per window"),
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
            seq_len=seq_len,
            train_window_days=train_window_days,
            max_epochs=max_epochs,
            hidden_size=hidden_size,
        )
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green][OK][/green] TFT Predictions generated: {results['predictions']} rows"
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@planner_app.command("windows")
def planner_windows(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    short_weeks: str = typer.Option("1-26", "--short-weeks", help="Short horizon weeks (min-max)"),
    medium_weeks: str = typer.Option(
        "13-52", "--medium-weeks", help="Medium horizon weeks (min-max)"
    ),
    long_weeks: str = typer.Option("26-104", "--long-weeks", help="Long horizon weeks (min-max)"),
    start: str = typer.Option(None, "--start", help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(None, "--end", help="End date in YYYY-MM-DD format"),
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

        def parse_weeks(weeks_str: str) -> tuple[int, int]:
            parts = weeks_str.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid week range format: {weeks_str}")
            return (int(parts[0]), int(parts[1]))

        short_range = parse_weeks(short_weeks)
        medium_range = parse_weeks(medium_weeks)
        long_range = parse_weeks(long_weeks)

        results = plan_windows(
            symbol=symbol,
            short_weeks=short_range,
            medium_weeks=medium_range,
            long_weeks=long_range,
            start=start,
            end=end,
            settings=settings,
        )

        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]Action Windows for {symbol}[/bold]")
        windows_all = results.get("windows", []) or []
        console.print(f"Total windows: {len(windows_all)}")

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


@eval_app.command("walkforward")
def eval_walkforward_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    train_months: int = typer.Option(
        24, "--train-months", "--train_months", help="Train months for walk-forward"
    ),
    step_months: int = typer.Option(
        1, "--step-months", "--step_months", help="Step months for walk-forward"
    ),
    start: str = typer.Option(..., "--start", help="Start month in YYYY-MM"),
    end: str = typer.Option(..., "--end", help="End month in YYYY-MM"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Evaluate predictions with walk-forward probabilistic + trading metrics (MVP)."""
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        report = eval_walkforward(
            symbol=symbol,
            start=start,
            end=end,
            train_months=train_months,
            step_months=step_months,
            settings=settings,
        )

        out_path = get_eval_walkforward_summary_path(symbol, settings=settings)
        console.print(f"[green][OK][/green] Eval report saved to: {out_path}")
        console.print(f"Horizons: {', '.join(report['horizons'].keys())}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command("backtest")
def eval_backtest_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    start: str = typer.Option(..., "--start", help="Start date in YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date in YYYY-MM-DD"),
    capital: float = typer.Option(10000.0, "--capital", help="Initial Capital in USDT"),
    root: Path = typer.Option(Path("./data"), "--root", help="Local data lake root directory"),
    strict: bool = typer.Option(False, "--strict", help="Fail fast on data quality issues"),
    save_metadata: bool = typer.Option(
        False, "--save-metadata", help="Save meta.json alongside artifacts"
    ),
) -> None:
    """Run Trading Simulator (Backtest) using generated predictions."""
    settings = get_settings(data_root=root, strict=strict, save_metadata=save_metadata)

    try:
        res = run_backtest(
            symbol=symbol,
            start=start,
            end=end,
            initial_capital=capital,
            settings=settings,
        )

        if "error" in res:
            console.print(f"[red]Error: {res['error']}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]Backtest Results for {symbol}[/bold]")
        console.print(f"Initial Capital: ${res['initial_capital']:.2f}")
        console.print(f"Final Value:     ${res['final_value']:.2f}")

        color = "green" if res["roi_pct"] >= 0 else "red"
        console.print(f"ROI:             [{color}]{res['roi_pct']:.2f}%[/{color}]")
        console.print(f"Max Drawdown:    [red]-{res['max_drawdown_pct']:.2f}%[/red]")
        console.print(f"Total Trades:    {res['total_trades']}")
        console.print(f"Days Tested:     {res['days_tested']}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("ui")
def start_ui(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the UI server on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host IP to bind to"),
) -> None:
    """
    Start the Prosper Data Manager Web UI (Dashboard).
    """
    try:
        import uvicorn
        console.print(f"[green]Starting Prosper UI on http://{host}:{port}[/green]")
        uvicorn.run("prosper.api.server:app", host=host, port=port, reload=False)
    except ImportError:
        console.print("[red]Uvicorn is not installed. Please install it with: poetry add uvicorn[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
