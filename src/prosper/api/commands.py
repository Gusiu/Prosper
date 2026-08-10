"""Structured request -> argv list.

Subprocesses run with ``shell=False``, so nothing here ever reaches a shell.
Keeping the translation on the server means the browser never has to know the
CLI's surface, and decisions about the data lake (such as what a repair
involves) stay where the data is.
"""

from __future__ import annotations

import re
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
from prosper.predict.defaults import HORIZON_NAMES

SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,30}$")


def _is_valid_symbol(symbol: str) -> bool:
    return bool(SYMBOL_RE.fullmatch(symbol))


def _build_research_flags(flags: ResearchFlags, command_type: str = "generic") -> list[str]:
    """Build CLI research flags from structured ResearchFlags object."""
    result: list[str] = []
    if flags.strict:
        result.append("--strict")
    if command_type == "predict":
        if flags.deterministic:
            result.append("--deterministic")
        if flags.seed is not None:
            result.extend(["--seed", str(flags.seed)])
    if flags.save_metadata:
        result.append("--save-metadata")
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
        "--root", "./data",
        *r_flags,
    ]
    commands.append(backfill_cmd)

    # 2. Aggregate (if intervals selected)
    if req.intervals:
        to_arg = ",".join(req.intervals)
        agg_cmd = [
            "python", "-m", "prosper.cli", "aggregate",
            "--symbol", symbol,
            "--from", "1m",
            "--start", s_month,
            "--end", e_month,
            "--to", to_arg,
            "--root", "./data",
            *r_flags,
        ]
        commands.append(agg_cmd)

        # 3. Features + Labels for all intervals
        all_intervals = ["1m", *req.intervals]
        for interval in all_intervals:
            features_cmd = [
                "python", "-m", "prosper.cli", "features", "build",
                "--symbol", symbol,
                "--base-interval", interval,
                "--start", req.start,
                "--end", req.end,
                "--root", "./data",
                *r_flags,
            ]
            commands.append(features_cmd)

            labels_cmd = [
                "python", "-m", "prosper.cli", "labels", "build",
                "--symbol", symbol,
                "--base-interval", interval,
                "--root", "./data",
                *r_flags,
            ]
            commands.append(labels_cmd)
    else:
        # Only 1m features + labels
        features_cmd = [
            "python", "-m", "prosper.cli", "features", "build",
            "--symbol", symbol,
            "--base-interval", "1m",
            "--start", req.start,
            "--end", req.end,
            "--root", "./data",
            *r_flags,
        ]
        commands.append(features_cmd)

        labels_cmd = [
            "python", "-m", "prosper.cli", "labels", "build",
            "--symbol", symbol,
            "--base-interval", "1m",
            "--root", "./data",
            *r_flags,
        ]
        commands.append(labels_cmd)

    return commands


def build_train_command(req: TrainRequest) -> list[list[str]]:
    """
    Build a safe training command from structured request.
    """
    symbol = req.symbol.upper()
    if not _is_valid_symbol(symbol):
        raise ValueError(f"Invalid symbol: {symbol}")

    r_flags = _build_research_flags(req.flags, "predict")
    model_type = req.model_type.lower()

    cmd = [
        "python", "-m", "prosper.cli", "predict", model_type,
        "--symbol", symbol,
        "--start", req.start,
        "--end", req.end,
        "--interval", req.interval,
        "--root", "./data",
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
            cmd.extend(["--seq-len", str(req.params["seq_len"])])
        if "hidden_size" in req.params:
            cmd.extend(["--hidden-size", str(req.params["hidden_size"])])
    elif model_type in ("xgboost", "ml"):
        if "train_window_days" in req.params:
            cmd.extend(["--train-window-days", str(req.params["train_window_days"])])
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
                cmd.extend(["--n-estimators", str(req.params["n_estimators"])])
            if "max_depth" in req.params:
                cmd.extend(["--max-depth", str(req.params["max_depth"])])
            if "learning_rate" in req.params:
                cmd.extend(["--learning-rate", str(req.params["learning_rate"])])

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
        "--model-type", req.model_type.lower(),
        "--timestamp", req.timestamp,
        "--interval", req.interval,
        "--root", "./data",
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
        "--root", "./data",
        *r_flags,
    ]
    if req.symbol:
        cmd.extend(["--symbol", req.symbol.upper()])
    if req.model_type:
        cmd.extend(["--model-type", req.model_type.lower()])
    if req.interval:
        cmd.extend(["--interval", req.interval])
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
                "--root", "./data",
                *r_flags,
            ]
        )
        commands.append(
            [
                "python", "-m", "prosper.cli", "labels", "build",
                "--symbol", symbol,
                "--base-interval", interval,
                "--root", "./data",
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
            "--to", ",".join(req.intervals),
            "--root", "./data",
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
    data lake: the frontend used to reconstruct it by regex-matching the
    human-readable quality message, which broke the moment that text changed.

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
                "--root", "./data",
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
                "--root", "./data",
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


