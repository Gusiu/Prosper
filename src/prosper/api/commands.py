"""Structured request -> argv list.

Subprocesses run with ``shell=False``, so nothing here ever reaches a shell.
Keeping the translation on the server means the browser never has to know the
CLI's surface, and decisions about the data lake (such as what a repair
involves) stay where the data is.
"""

from __future__ import annotations

from typing import Any

from prosper.api.models import (
    AggregateRequest,
    BatchEvalRequest,
    EvalRequest,
    PipelineRequest,
    RepairRequest,
    ResearchFlags,
    TrainRequest,
)
from prosper.api.validation import (
    is_valid_interval,
    is_valid_model_type,
    is_valid_run_timestamp,
    is_valid_symbol,
)
from prosper.config import get_settings
from prosper.predict.defaults import HORIZON_NAMES, POOLING_MODELS

# Kept under its old name: `prosper.api.server` re-exports it.
_is_valid_symbol = is_valid_symbol


def _root() -> str:
    """The data lake the server was configured with, as the CLI's `--root`.

    The CLI takes its root from that flag alone, so this is the only way a
    queued job learns which lake the endpoints are reading; a literal would send
    it to `./data` whatever `PROSPER_DATA_ROOT` says. Read per request, as the
    routes do. A relative root is passed as configured: the subprocess inherits
    the server's working directory, so it names the same directory in both.
    """
    return str(get_settings().data_root)


def _interval(value: str) -> str:
    """An interval that becomes argv, and then a directory the CLI writes into."""
    if not is_valid_interval(value):
        raise ValueError(f"Invalid interval: {value}")
    return value


def _intervals(values: list[str]) -> list[str]:
    return [_interval(value) for value in values]


def _model_type(value: str) -> str:
    normalised = value.lower()
    if not is_valid_model_type(normalised):
        raise ValueError(f"Invalid model type: {value}")
    return normalised


def _run_timestamp(value: str) -> str:
    if not is_valid_run_timestamp(value):
        raise ValueError(f"Invalid run timestamp: {value}")
    return value


def _number(params: dict[str, Any], key: str, kind: type) -> str:
    """A numeric model parameter, rendered by us rather than by the caller.

    The value lands in argv after a fixed flag, so it could not smuggle in a
    flag of its own - but it could still arrive as any JSON type, and a string
    that happens to parse later is not a number that was checked now.
    """
    try:
        return str(kind(params[key]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key}: {params[key]!r}") from exc


def _build_research_flags(flags: ResearchFlags, command_type: str = "generic") -> list[str]:
    """Build CLI research flags from structured ResearchFlags object."""
    result: list[str] = []
    if command_type == "predict":
        if flags.deterministic:
            result.append("--deterministic")
        if flags.seed is not None:
            result.extend(["--seed", str(flags.seed)])
    return result


def build_pipeline_command(req: PipelineRequest) -> list[list[str]]:
    """
    Build a safe pipeline command chain from structured request.
    Returns list of argv lists (one per chained command).
    """
    symbol = req.symbol.upper()
    if not _is_valid_symbol(symbol):
        raise ValueError(f"Invalid symbol: {symbol}")

    s_month = req.start[:7] if len(req.start) >= 7 else req.start
    e_month = req.end[:7] if len(req.end) >= 7 else req.end
    r_flags = _build_research_flags(req.flags, "pipeline")

    commands: list[list[str]] = []

    # 1. Backfill
    backfill_cmd = [
        "python", "-m", "prosper.cli", "backfill",
        "--symbol", symbol,
        "--start", s_month,
        "--end", e_month,
        "--root", _root(),
        *r_flags,
    ]
    commands.append(backfill_cmd)

    # 2. Aggregate (if intervals selected)
    intervals = _intervals(req.intervals)
    if intervals:
        commands.append(
            [
                "python", "-m", "prosper.cli", "aggregate",
                "--symbol", symbol,
                "--from", "1m",
                "--start", s_month,
                "--end", e_month,
                "--to", ",".join(intervals),
                "--root", _root(),
                *r_flags,
            ]
        )

    # 3. Features and labels for the base interval and everything aggregated.
    commands.extend(
        _features_and_labels_commands(symbol, ["1m", *intervals], req.start, req.end, r_flags)
    )

    return commands


def build_train_command(req: TrainRequest) -> list[list[str]]:
    """
    Build a safe training command from structured request.
    """
    symbol = req.symbol.upper()
    if not _is_valid_symbol(symbol):
        raise ValueError(f"Invalid symbol: {symbol}")

    r_flags = _build_research_flags(req.flags, "predict")
    model_type = _model_type(req.model_type)

    cmd = [
        "python", "-m", "prosper.cli", "predict", model_type,
        "--symbol", symbol,
        "--start", req.start,
        "--end", req.end,
        "--interval", _interval(req.interval),
        "--root", _root(),
        *r_flags,
    ]

    # Add model-specific params
    # Only pass a horizon list when it is actually a subset; sending the full
    # set on every call would make every argv differ from the CLI's own default.
    selected = [h for h in HORIZON_NAMES if h in set(req.horizons or HORIZON_NAMES)]
    if selected and len(selected) < len(HORIZON_NAMES):
        cmd.extend(["--horizons", ",".join(selected)])

    if req.train_window_by_horizon:
        pairs = ",".join(
            f"{name}={days}"
            for name, days in req.train_window_by_horizon.items()
            if name in HORIZON_NAMES
        )
        if pairs:
            cmd.extend(["--train-window-per-horizon", pairs])

    if model_type in ("gru", "tft"):
        # Silence, not a default, when the caller did not choose one.
        if req.epochs is not None:
            cmd.extend(["--epochs" if model_type == "gru" else "--max-epochs", str(req.epochs)])
        if req.epochs_by_horizon:
            pairs = ",".join(
                f"{name}={count}"
                for name, count in req.epochs_by_horizon.items()
                if name in HORIZON_NAMES
            )
            if pairs:
                cmd.extend(["--epochs-per-horizon", pairs])
        if "seq_len" in req.params:
            cmd.extend(["--seq-len", _number(req.params, "seq_len", int)])
        if "hidden_size" in req.params:
            cmd.extend(["--hidden-size", _number(req.params, "hidden_size", int)])
    elif model_type in POOLING_MODELS:
        if "train_window_days" in req.params:
            cmd.extend(["--train-window-days", _number(req.params, "train_window_days", int)])
        # Only the tabular predictors pool; the sequence models do not offer the
        # flag, so passing it there would fail the whole run rather than be
        # ignored. Each name is validated because this string becomes argv.
        pooled = [
            name.upper()
            for name in req.pool_symbols
            if _is_valid_symbol(name.upper()) and name.upper() != symbol
        ]
        if pooled:
            cmd.extend(["--symbols", ",".join(dict.fromkeys(pooled))])
        if model_type == "xgboost":
            if "n_estimators" in req.params:
                cmd.extend(["--n-estimators", _number(req.params, "n_estimators", int)])
            if "max_depth" in req.params:
                cmd.extend(["--max-depth", _number(req.params, "max_depth", int)])
            if "learning_rate" in req.params:
                cmd.extend(["--learning-rate", _number(req.params, "learning_rate", float)])

    return [cmd]


def build_eval_command(req: EvalRequest) -> list[list[str]]:
    """
    Build a safe evaluation command from structured request.
    """
    symbol = req.symbol.upper()
    if not _is_valid_symbol(symbol):
        raise ValueError(f"Invalid symbol: {symbol}")

    r_flags = _build_research_flags(req.flags, "generic")

    cmd = [
        "python", "-m", "prosper.cli", "eval", "predictions",
        "--symbol", symbol,
        "--model-type", _model_type(req.model_type),
        "--timestamp", _run_timestamp(req.timestamp),
        "--interval", _interval(req.interval),
        "--root", _root(),
        *r_flags,
    ]
    return [cmd]


def build_batch_eval_command(req: BatchEvalRequest) -> list[list[str]]:
    """
    Build a safe batch evaluation command from structured request.
    """
    r_flags = _build_research_flags(req.flags, "generic")

    cmd = [
        "python", "-m", "prosper.cli", "eval", "batch",
        "--root", _root(),
        *r_flags,
    ]
    if req.symbol:
        symbol = req.symbol.upper()
        if not _is_valid_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        cmd.extend(["--symbol", symbol])
    if req.model_type:
        cmd.extend(["--model-type", _model_type(req.model_type)])
    if req.interval:
        cmd.extend(["--interval", _interval(req.interval)])
    if req.limit is not None:
        cmd.extend(["--limit", str(req.limit)])

    return [cmd]


def _features_and_labels_commands(
    symbol: str, intervals: list[str], start: str, end: str, r_flags: list[str]
) -> list[list[str]]:
    commands: list[list[str]] = []
    for interval in intervals:
        commands.append(
            [
                "python", "-m", "prosper.cli", "features", "build",
                "--symbol", symbol,
                "--base-interval", interval,
                "--start", start,
                "--end", end,
                "--root", _root(),
                *r_flags,
            ]
        )
        commands.append(
            [
                "python", "-m", "prosper.cli", "labels", "build",
                "--symbol", symbol,
                "--base-interval", interval,
                "--root", _root(),
                *r_flags,
            ]
        )
    return commands


def build_aggregate_command(req: AggregateRequest) -> list[list[str]]:
    """Aggregate the requested intervals, then rebuild features and labels."""
    symbol = req.symbol.upper()
    if not _is_valid_symbol(symbol):
        raise ValueError(f"Invalid symbol: {symbol}")
    if not req.intervals:
        raise ValueError("Select at least one interval to aggregate")

    r_flags = _build_research_flags(req.flags, "pipeline")
    commands = [
        [
            "python", "-m", "prosper.cli", "aggregate",
            "--symbol", symbol,
            "--from", "1m",
            "--start", req.start[:7],
            "--end", req.end[:7],
            "--to", ",".join(_intervals(req.intervals)),
            "--root", _root(),
            *r_flags,
        ]
    ]
    commands.extend(
        _features_and_labels_commands(
            symbol, ["1m", *req.intervals], req.start, req.end, r_flags
        )
    )
    return commands


def build_repair_commands(
    req: RepairRequest, inventory: list[dict[str, Any]]
) -> tuple[list[list[str]], list[str]]:
    """Derive the repair chain for a symbol from its inventory entry.

    Lives here rather than in the browser because it is a decision about the
    data lake. Reconstructed in the frontend by regex-matching the
    human-readable quality message, it would break the moment that text changed.

    Returns the commands plus a human-readable explanation of each step.
    """
    symbol = req.symbol.upper()
    if not _is_valid_symbol(symbol):
        raise ValueError(f"Invalid symbol: {symbol}")

    entry = next((item for item in inventory if item.get("symbol") == symbol), None)
    if entry is None:
        raise ValueError(f"{symbol} is not present in the data inventory")

    start_date = str(entry.get("start_date", ""))
    end_date = str(entry.get("end_date", ""))
    if start_date == "N/A" or end_date == "N/A" or not start_date or not end_date:
        raise ValueError(f"{symbol} has no usable date range to repair")

    r_flags = _build_research_flags(req.flags, "pipeline")
    aggregations = entry.get("aggregations", []) or []
    commands: list[list[str]] = []
    steps: list[str] = []

    # 1. Gaps in the 1m base require re-downloading; everything else derives from it.
    base = next((a for a in aggregations if a.get("interval") == "1m"), None)
    if base and base.get("gaps", 0) > 0:
        commands.append(
            [
                "python", "-m", "prosper.cli", "backfill",
                "--symbol", symbol,
                "--start", start_date[:7],
                "--end", end_date[:7],
                "--root", _root(),
                *r_flags,
            ]
        )
        steps.append(f"re-download 1m ({base['gaps']} gap(s))")

    # 2. Any other interval that is gapped or missing parquet is re-aggregated.
    broken = sorted(
        {
            a["interval"]
            for a in aggregations
            if a.get("interval") != "1m"
            and (a.get("gaps", 0) > 0 or a.get("status") == "incomplete")
        }
    )
    if broken:
        commands.append(
            [
                "python", "-m", "prosper.cli", "aggregate",
                "--symbol", symbol,
                "--from", "1m",
                "--start", start_date[:7],
                "--end", end_date[:7],
                "--to", ",".join(broken),
                "--root", _root(),
                *r_flags,
            ]
        )
        steps.append(f"re-aggregate {', '.join(broken)}")

    # 3. Intervals flagged only for missing features/labels, plus anything rebuilt above.
    needs_features = sorted(
        {a["interval"] for a in aggregations if a.get("status") == "warning"} | set(broken)
    )
    if needs_features:
        commands.extend(
            _features_and_labels_commands(
                symbol, needs_features, start_date, end_date, r_flags
            )
        )
        steps.append(f"rebuild features and labels for {', '.join(needs_features)}")

    return commands, steps


