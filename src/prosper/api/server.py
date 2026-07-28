import datetime
import json
import re
import shlex
import shutil
import subprocess
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prosper.config import get_settings
from prosper.eval.predictions import flatten_quality_row
from prosper.inventory import DataManager
from prosper.storage.layout import (
    get_versioned_prediction_file,
    parse_evaluation_folder,
    parse_versioned_prediction_folder,
    resolve_evaluation_run_dir,
    resolve_versioned_prediction_dir,
)
from prosper.storage.runs import list_runs

# ============================================================
# NEW SAFE PYDANTIC MODELS (Phase 1.1 - REST API Refactor)
# Frontend sends structured JSON instead of CLI command strings.
# ============================================================

class ResearchFlags(BaseModel):
    """Research mode flags for reproducibility."""
    strict: bool = False
    deterministic: bool = False
    seed: int | None = 42
    save_metadata: bool = False


class PipelineRequest(BaseModel):
    """Safe structured request for data pipeline operations."""
    symbol: str
    start: str
    end: str
    intervals: list[str] = Field(default_factory=list)
    flags: ResearchFlags = Field(default_factory=ResearchFlags)


class TrainRequest(BaseModel):
    """Safe structured request for model training."""
    symbol: str
    model_type: Literal["ml", "xgboost", "gru", "tft", "baseline"] = "xgboost"
    interval: str = "1d"
    start: str
    end: str
    epochs: int = 5
    params: dict[str, Any] = Field(default_factory=dict)
    flags: ResearchFlags = Field(default_factory=ResearchFlags)


class EvalRequest(BaseModel):
    """Safe structured request for prediction evaluation."""
    symbol: str
    model_type: str
    timestamp: str
    interval: str = "1d"
    flags: ResearchFlags = Field(default_factory=ResearchFlags)


class BatchEvalRequest(BaseModel):
    """Safe structured request for batch evaluation."""
    symbol: str | None = None
    model_type: str | None = None
    interval: str | None = None
    limit: int | None = None
    flags: ResearchFlags = Field(default_factory=ResearchFlags)


# ============================================================
# SAFE COMMAND BUILDERS
# These functions build CLI command lists from structured data,
# eliminating the need for the frontend to construct command strings.
# ============================================================

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
    if model_type in ("gru", "tft"):
        cmd.extend(["--epochs" if model_type == "gru" else "--max-epochs", str(req.epochs)])
        if "seq_len" in req.params:
            cmd.extend(["--seq-len", str(req.params["seq_len"])])
        if "hidden_size" in req.params:
            cmd.extend(["--hidden-size", str(req.params["hidden_size"])])
    elif model_type == "xgboost":
        if "train_window_days" in req.params:
            cmd.extend(["--train-window-days", str(req.params["train_window_days"])])
        if "n_estimators" in req.params:
            cmd.extend(["--n-estimators", str(req.params["n_estimators"])])
        if "max_depth" in req.params:
            cmd.extend(["--max-depth", str(req.params["max_depth"])])
        if "learning_rate" in req.params:
            cmd.extend(["--learning-rate", str(req.params["learning_rate"])])
    elif model_type == "ml":
        if "train_window_days" in req.params:
            cmd.extend(["--train-window-days", str(req.params["train_window_days"])])

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


# ============================================================
# END NEW SAFE MODELS
# ============================================================

# Resolve the absolute path dynamically based on this file's location.
# This makes the project portable to any computer or directory.
frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,30}$")
FORBIDDEN_COMMAND_CHARS = set("|;<>\n\r`")
ALLOWED_PROSPER_COMMANDS = {
    "backfill",
    "aggregate",
    "features",
    "labels",
    "predict",
    "planner",
    "eval",
    "qa",
    "symbols",
}


def _is_valid_symbol(symbol: str) -> bool:
    return bool(SYMBOL_RE.fullmatch(symbol))


def parse_safe_command_chain(command: str) -> list[list[str]]:
    """
    Parse a UI command chain into argv segments without allowing shell execution.

    The frontend may chain Prosper CLI commands with `&&`. Each segment must be
    `poetry run prosper ...`; shell metacharacters are rejected and subprocesses
    are later started with shell=False.
    """
    if any(ch in command for ch in FORBIDDEN_COMMAND_CHARS):
        raise ValueError("Command contains unsupported shell metacharacters")

    segments: list[list[str]] = []
    for raw_segment in command.split("&&"):
        segment = raw_segment.strip()
        if not segment:
            continue
        try:
            args = shlex.split(segment)
        except ValueError as e:
            raise ValueError(f"Invalid command quoting: {e}") from e

        if len(args) < 4 or args[:3] != ["python", "-m", "prosper.cli"]:
            raise ValueError("Each segment must start with 'python -m prosper.cli'")
        if args[3] not in ALLOWED_PROSPER_COMMANDS:
            raise ValueError(f"Unsupported Prosper command: {args[3]}")
        if any(any(ch in token for ch in FORBIDDEN_COMMAND_CHARS) for token in args):
            raise ValueError("Command token contains unsupported characters")
        segments.append(args)

    if not segments:
        raise ValueError("Command is empty")
    return segments


@asynccontextmanager
async def lifespan(app: FastAPI):
    frontend_dir.mkdir(parents=True, exist_ok=True)

    # Auto-open browser via Python
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    yield


app = FastAPI(title="Prosper Analysis Platform UI", lifespan=lifespan)


# A long training run emits hundreds of thousands of lines. The UI only ever
# renders the tail, so keep a bounded window in memory instead of the lot.
MAX_TASK_LOG_LINES = 5000


class TaskRunner:
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.logs: list[str] = []
        self.dropped_log_lines: int = 0
        self.is_running: bool = False
        self.progress: dict[str, Any] = {
            "visible": False,
            "percent": 0.0,
            "label": "Idle",
            "stage": "idle",
        }

    def start(self, cmd_str: str, commands: list[list[str]]):
        if self.is_running:
            raise HTTPException(400, "A task is already running!")
        self.logs = [f"$ {cmd_str}"]
        self.dropped_log_lines = 0
        self.is_running = True
        self.progress = {
            "visible": True,
            "percent": 0.0,
            "label": "Starting",
            "stage": "queued",
        }

        def run_thread():
            try:
                import sys

                total_commands = max(1, len(commands))
                for command_index, original_args in enumerate(commands):
                    self._append_log(f"$ {' '.join(original_args)}")
                    self._set_command_progress(command_index, total_commands, original_args)

                    # Ensure we use the current virtualenv's Python
                    run_args = list(original_args)
                    if run_args[0] == "python":
                        run_args[0] = sys.executable

                    self.process = subprocess.Popen(
                        run_args,
                        shell=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    if self.process.stdout:
                        for line in iter(self.process.stdout.readline, ""):
                            if line:
                                clean_line = line.rstrip("\n")
                                self._append_log(clean_line)
                                self._update_progress_from_log(
                                    clean_line,
                                    command_index,
                                    total_commands,
                                )
                    self.process.wait()
                    self._append_log(
                        f"Task finished with exit code {self.process.returncode}"
                    )
                    # PĹ‚ynny skok postÄ™pu po zakoĹ„czeniu sub-komendy
                    self._set_command_progress(
                        command_index, total_commands, original_args, finished=True
                    )
                    if self.process.returncode != 0:
                        self.progress.update({"label": "Failed", "stage": "failed"})
                        break
            except Exception as e:
                self._append_log(f"Error executing task: {str(e)}")
                self.progress.update({"label": "Failed", "stage": "failed"})
            finally:
                self.is_running = False
                self.process = None
                if self.progress.get("stage") != "failed":
                    self.progress.update(
                        {"percent": 100.0, "label": "Complete", "stage": "complete"}
                    )

        threading.Thread(target=run_thread, daemon=True).start()

    def _append_log(self, line: str) -> None:
        """Append a line, discarding the oldest once the window is full.

        `dropped_log_lines` keeps client cursors absolute: readers page by a
        line number that survives trimming rather than by list position.
        """
        self.logs.append(line)
        overflow = len(self.logs) - MAX_TASK_LOG_LINES
        if overflow > 0:
            del self.logs[:overflow]
            self.dropped_log_lines += overflow

    def read_logs(self, start_idx: int) -> tuple[list[str], int]:
        """Return log lines from absolute *start_idx* and the next cursor."""
        offset = self.dropped_log_lines
        local_start = max(0, start_idx - offset)
        lines = self.logs[local_start:]
        return lines, offset + len(self.logs)

    def _set_command_progress(
        self, command_index: int, total_commands: int, args: list[str], finished: bool = False
    ) -> None:
        # Extract a human-readable command name + symbol
        cli_cmd = args[3] if len(args) > 3 else "task"

        # Try to extract --symbol value
        symbol = ""
        for i, arg in enumerate(args):
            if arg == "--symbol" and i + 1 < len(args):
                symbol = args[i + 1]
                break

        label = f"{cli_cmd.title()} {symbol}" if symbol else cli_cmd.title()

        if finished:
            percent = ((command_index + 1) / total_commands) * 100.0
        else:
            percent = (command_index / total_commands) * 100.0
        self.progress.update(
            {
                "visible": True,
                "percent": round(percent, 1),
                "label": label,
                "stage": cli_cmd,
            }
        )

    def _update_progress_from_log(self, line: str, command_index: int, total_commands: int) -> None:
        match = re.search(r"\[\s*(\d+(?:\.\d+)?)%\s*\]", line)
        if not match:
            return
        local_percent = min(100.0, max(0.0, float(match.group(1))))
        overall_percent = ((command_index + local_percent / 100.0) / total_commands) * 100.0
        self._set_percent(overall_percent)

    def _set_percent(self, percent: float) -> None:
        self.progress["visible"] = True
        self.progress["percent"] = round(min(100.0, max(0.0, percent)), 1)


runner = TaskRunner()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_status() -> dict[str, str]:
    return {"status": "online", "message": "Prosper Engine is operational"}


class RunRequest(BaseModel):
    command: str


@app.post("/api/task/run")
def run_task(req: RunRequest):
    try:
        commands = parse_safe_command_chain(req.command)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    runner.start(req.command, commands)
    return {"status": "started", "command": req.command}


@app.get("/api/task/logs")
def get_task_logs(start_idx: int = 0):
    lines, next_idx = runner.read_logs(start_idx)
    return {
        "logs": lines,
        "next_idx": next_idx,
        "dropped": runner.dropped_log_lines,
        "is_running": runner.is_running,
        "progress": runner.progress,
    }


@app.get("/api/task/status")
def get_task_status():
    return {"is_running": runner.is_running, "progress": runner.progress}


@app.get("/api/data/dates")
def get_dates():
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)

    return {
        "today": today.strftime("%Y-%m-%d"),
        "yesterday": yesterday.strftime("%Y-%m-%d"),
        "default_start_month": "2017-08",
        "yesterday_month": yesterday.strftime("%Y-%m"),
    }


@app.get("/api/data/inventory")
def get_inventory(refresh: bool = False):
    return DataManager().get_inventory(refresh=refresh)


class AIEvaluateRequest(BaseModel):
    symbol: str
    model_type: str
    timestamp: str
    interval: str = "1d"
    params: dict | None = None


@app.get("/api/ai/models")
def get_ai_models() -> dict[str, list[dict[str, Any]]]:
    settings = get_settings()
    base = settings.reports_predictions_dir
    if not base.exists():
        return {"models": []}

    models: list[dict[str, Any]] = []
    for symbol_dir in sorted(base.iterdir() if base.exists() else []):
        if not symbol_dir.is_dir():
            continue
        for entry in sorted(symbol_dir.iterdir()):
            if not entry.is_dir() or "_" not in entry.name:
                continue
            parsed = parse_versioned_prediction_folder(entry.name)
            if not parsed:
                continue
            model_type, interval, timestamp = parsed
            has_fi = (entry / "feature_importances.json").exists()
            jsonl_count = sum(1 for _ in entry.rglob("*.jsonl"))
            models.append(
                {
                    "symbol": symbol_dir.name,
                    "model_type": model_type,
                    "interval": interval,
                    "timestamp": timestamp,
                    "path": str(entry),
                    "has_feature_importances": has_fi,
                    "predictions_files": jsonl_count,
                }
            )
    return {"models": models}


@app.get("/api/runs")
def list_prediction_runs(
    symbol: str | None = None,
    model_type: str | None = None,
    interval: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """List versioned prediction runs, newest first.

    Every downstream artifact (windows, walk-forward, backtest) is keyed by one
    of these runs, so this is the canonical picker for the UI.
    """
    if symbol and not _is_valid_symbol(symbol.upper()):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    runs = list_runs(
        symbol=symbol, model_type=model_type, interval=interval, settings=get_settings()
    )
    return {"runs": [run.to_dict() for run in runs]}


@app.get("/api/ai/predictions")
def get_ai_predictions(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str = "1d",
    year: int | None = None,
    month: int | None = None,
):
    """Return a run's predictions, optionally narrowed to one calendar month.

    A full sub-daily run is tens of megabytes, so callers that only render one
    month should pass ``year``/``month`` rather than filtering client-side.
    """
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    path = get_versioned_prediction_file(
        symbol, model_type, timestamp, settings=settings, interval=interval
    )
    if not path.exists():
        raise HTTPException(status_code=404, detail="Versioned prediction file not found")

    month_prefix: str | None = None
    if year is not None and month is not None:
        month_prefix = f"{year:04d}-{month:02d}"

    try:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if month_prefix and str(row.get("date", ""))[:7] != month_prefix:
                    continue
                rows.append(row)
        return {"predictions": rows, "month": month_prefix, "count": len(rows)}
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/ai/predictions/months")
def get_ai_prediction_months(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str = "1d",
) -> dict[str, Any]:
    """Return the sorted list of ``YYYY-MM`` months present in a run."""
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    path = get_versioned_prediction_file(
        symbol, model_type, timestamp, settings=settings, interval=interval
    )
    if not path.exists():
        raise HTTPException(status_code=404, detail="Versioned prediction file not found")

    months: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                date_value = str(row.get("date", ""))
                if len(date_value) >= 7:
                    months.add(date_value[:7])
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"months": sorted(months)}


@app.get("/api/ai/feature_importances")
def get_feature_importances(
    symbol: str, model_type: str, timestamp: str, interval: str = "1d"
):
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    p = (
        resolve_versioned_prediction_dir(
            symbol, model_type, timestamp, settings=settings, interval=interval
        )
        / "feature_importances.json"
    )
    if not p.exists():
        raise HTTPException(status_code=404, detail="Feature importances not found for this run")
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/api/ai/models/{symbol}/{model_type}/{timestamp}")
def delete_ai_model(
    symbol: str, model_type: str, timestamp: str, interval: str = "1d"
) -> dict[str, Any]:
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    run_dir = resolve_versioned_prediction_dir(
        symbol, model_type.lower(), timestamp, settings=settings, interval=interval
    )
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Model run not found")

    try:
        shutil.rmtree(run_dir)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"status": "deleted", "path": str(run_dir)}


# ============================================================
# NEW SAFE ENDPOINTS (Phase 1.1)
# Frontend sends structured JSON - no command string construction.
# ============================================================

@app.post("/api/pipeline/run")
def run_pipeline(req: PipelineRequest):
    """
    Start a data pipeline (backfill -> aggregate -> features -> labels).
    Accepts structured JSON - SAFE: no command string parsing needed.
    """
    try:
        commands = build_pipeline_command(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cmd_display = " && ".join(" ".join(cmd) for cmd in commands)
    runner.start(cmd_display, commands)
    return {
        "status": "started",
        "command": cmd_display,
        "steps": len(commands),
    }


@app.post("/api/train/run")
def run_training(req: TrainRequest):
    """
    Start model training.
    Accepts structured JSON - SAFE: no command string parsing needed.
    """
    try:
        commands = build_train_command(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cmd_display = " && ".join(" ".join(cmd) for cmd in commands)
    runner.start(cmd_display, commands)
    return {
        "status": "started",
        "command": cmd_display,
    }


@app.post("/api/eval/run")
def run_evaluation(req: EvalRequest):
    """
    Start prediction evaluation.
    Accepts structured JSON - SAFE: no command string parsing needed.
    """
    try:
        commands = build_eval_command(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cmd_display = " && ".join(" ".join(cmd) for cmd in commands)
    runner.start(cmd_display, commands)
    return {
        "status": "started",
        "command": cmd_display,
    }


@app.post("/api/eval/batch")
def run_batch_evaluation(req: BatchEvalRequest):
    """
    Start batch evaluation of model runs.
    Accepts structured JSON - SAFE: no command string parsing needed.
    """
    try:
        commands = build_batch_eval_command(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cmd_display = " && ".join(" ".join(cmd) for cmd in commands)
    runner.start(cmd_display, commands)
    return {
        "status": "started",
        "command": cmd_display,
    }


# ============================================================
# LEGACY ENDPOINTS (still used by the current frontend)
# Removed after the frontend migrates to the structured API above.
# ============================================================

def _read_json_file(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl_file(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


def _evaluation_summary_from_metrics(metrics: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    return {
        "symbol": metrics.get("symbol"),
        "model_type": metrics.get("model_type"),
        "timestamp": metrics.get("timestamp"),
        "interval": metrics.get("interval", "1d"),
        "created_at": metrics.get("created_at"),
        "path": str(run_dir),
        "samples": overall.get("samples", metrics.get("scored_rows", 0)),
        "missing_targets": overall.get("missing_targets"),
        "untrained_rows": overall.get("untrained_rows", 0),
        "model_score": overall.get("model_score"),
        "accuracy": overall.get("accuracy"),
        "brier": overall.get("brier"),
        "nll": overall.get("nll"),
        "ece": overall.get("ece"),
        "artifacts": metrics.get("artifacts", {}),
    }


@app.post("/api/ai/evaluate")
def evaluate_model_predictions(req: AIEvaluateRequest):
    symbol = req.symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    model_type = req.model_type.lower()
    interval = req.interval or "1d"
    params = req.params or {}
    flags = []
    if params.get("strict"):
        flags.append("--strict")
    if params.get("save_metadata"):
        flags.append("--save-metadata")
    cmd = (
        f"python -m prosper.cli eval predictions --symbol {symbol} "
        f"--model-type {model_type} --timestamp {req.timestamp} "
        f"--interval {interval} --root ./data {' '.join(flags)}"
    )

    try:
        commands = parse_safe_command_chain(cmd)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    runner.start(cmd, commands)
    return {"status": "started", "command": cmd}


@app.get("/api/ai/evaluations")
def list_ai_evaluations(
    symbol: str | None = None,
    model_type: str | None = None,
    sort_by: str = "model_score",
) -> dict[str, list[dict[str, Any]]]:
    settings = get_settings()
    base = settings.reports_dir / "evaluations"
    if not base.exists():
        return {"evaluations": []}

    symbol_filter = symbol.upper() if symbol else None
    model_filter = model_type.lower() if model_type else None
    evaluations: list[dict[str, Any]] = []
    for symbol_dir in sorted(base.iterdir()):
        if not symbol_dir.is_dir() or symbol_dir.name.startswith("."):
            continue
        if symbol_filter and symbol_dir.name.upper() != symbol_filter:
            continue
        for run_dir in sorted(symbol_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            parsed = parse_evaluation_folder(run_dir.name)
            if not parsed:
                continue
            parsed_model_type, parsed_interval, parsed_timestamp = parsed
            if model_filter and parsed_model_type.lower() != model_filter:
                continue
            metrics = _read_json_file(run_dir / "metrics.json", {})
            if not metrics:
                metrics = {
                    "symbol": symbol_dir.name,
                    "model_type": parsed_model_type,
                    "interval": parsed_interval,
                    "timestamp": parsed_timestamp,
                }
            evaluations.append(_evaluation_summary_from_metrics(metrics, run_dir))

    sort_key = sort_by if sort_by in {"model_score", "accuracy", "brier", "ece", "nll"} else "model_score"
    reverse = sort_key != "brier" and sort_key != "ece" and sort_key != "nll"
    evaluations.sort(
        key=lambda item: (
            item.get(sort_key) is not None,
            item.get(sort_key) if reverse else -(item.get(sort_key) or 0),
            item.get("created_at") or "",
        ),
        reverse=True,
    )
    return {"evaluations": evaluations}


@app.get("/api/ai/evaluations/status")
def get_evaluation_batch_status() -> dict[str, Any]:
    settings = get_settings()
    status_path = settings.reports_dir / "evaluations" / "batch_status.json"
    payload = _read_json_file(status_path, {})
    if not payload:
        return {"status": "idle", "evaluated": 0, "failed": 0, "results": [], "alerts": []}

    alerts: list[str] = []
    failed = int(payload.get("failed", 0) or 0)
    evaluated = int(payload.get("evaluated", 0) or 0)
    if failed > 0:
        alerts.append(f"{failed} evaluation run(s) failed in the latest batch.")
    for item in payload.get("results", []):
        if item.get("status") != "ok":
            continue
        score = item.get("model_score")
        if isinstance(score, int | float) and score < 50:
            alerts.append(
                f"Low model score ({score:.1f}) for {item.get('symbol')} "
                f"{item.get('model_type')} {item.get('timestamp')}."
            )
    payload["alerts"] = alerts
    payload["status"] = "failed" if failed and not evaluated else ("partial" if failed else "ok")
    payload["status_path"] = str(status_path)
    return payload


def _load_worst_predictions(
    run_dir: Path, limit: int, flag: str | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Return the worst-scoring rows of a run, plus the total scored count.

    Reads `predictions_quality.parquet` lazily so a run with hundreds of
    thousands of rows costs a bounded amount of memory. Falls back to the
    legacy JSONL for evaluations produced before the Parquet switch.
    """
    parquet_path = run_dir / "predictions_quality.parquet"
    if parquet_path.exists():
        lazy = pl.scan_parquet(str(parquet_path)).filter(pl.col("status") == "scored")
        if flag:
            lazy = lazy.filter(pl.col("flags").str.contains(flag, literal=True))
        total = int(lazy.select(pl.len()).collect().item())
        frame = lazy.sort("error_score", descending=True).head(limit).collect()
        return frame.to_dicts(), total

    legacy = run_dir / "predictions_quality.jsonl"
    rows = [
        row
        for row in _read_jsonl_file(legacy, limit=20000)
        if row.get("status") == "scored"
        and (not flag or flag in (row.get("flags") or []))
    ]
    rows.sort(key=lambda row: row.get("error_score", 0.0), reverse=True)
    return [flatten_quality_row(row) for row in rows[:limit]], len(rows)


@app.get("/api/ai/evaluations/{symbol}/{model_type}/{timestamp}")
def get_ai_evaluation_detail(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str = "1d",
    limit: int = 100,
    flag: str | None = None,
) -> dict[str, Any]:
    """Summary, calibration and the worst rows of one evaluation run.

    Deliberately bounded: an unfiltered response for a 1h run used to be ~71 MB
    because every recommendation was serialised so the UI could render fifty.
    """
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 5000")

    run_dir = resolve_evaluation_run_dir(
        symbol, model_type.lower(), timestamp, settings=settings, interval=interval
    )
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Evaluation not found")

    metrics = _read_json_file(run_dir / "metrics.json", {})
    calibration = _read_json_file(run_dir / "calibration.json", {})
    worst_rows, scored_total = _load_worst_predictions(run_dir, limit=limit, flag=flag)
    recommendations = _read_json_file(run_dir / "recommendations.json", {"items": []})
    return {
        "metrics": metrics,
        "summary": _evaluation_summary_from_metrics(metrics, run_dir),
        "calibration": calibration,
        "worst_predictions": worst_rows,
        "worst_predictions_total": scored_total,
        "recommendations": recommendations.get("items", [])[:limit],
        "recommendations_truncated": bool(recommendations.get("truncated")),
        "artifacts": metrics.get("artifacts", {}),
    }


@app.get("/api/ai/evaluations/{symbol}/{model_type}/{timestamp}/flags")
def get_ai_evaluation_flags(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str = "1d",
) -> dict[str, Any]:
    """Distinct quality flags in a run, with counts, for the UI filter."""
    settings = get_settings()
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    run_dir = resolve_evaluation_run_dir(
        symbol, model_type.lower(), timestamp, settings=settings, interval=interval
    )
    parquet_path = run_dir / "predictions_quality.parquet"
    if not parquet_path.exists():
        return {"flags": []}

    frame = (
        pl.scan_parquet(str(parquet_path))
        .filter((pl.col("status") == "scored") & (pl.col("flags") != ""))
        .select(pl.col("flags").str.split("|").alias("flag"))
        .explode("flag")
        .group_by("flag")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect()
    )
    return {"flags": frame.to_dicts()}


@app.get("/api/meta/symbols")
def get_symbol_metadata() -> dict[str, Any]:
    settings = get_settings()
    symbols_path = settings.meta_dir / "binance_spot_symbols.json"
    if not symbols_path.exists():
        return {"symbols": [], "base_assets": [], "quote_assets": []}

    try:
        payload = json.loads(symbols_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Cannot read symbols metadata: {e}") from e

    symbols_raw = payload.get("symbols", []) if isinstance(payload, dict) else payload
    symbols: set[str] = set()
    base_assets: set[str] = set()
    quote_assets: set[str] = set()
    known_quotes = [
        "USDT",
        "FDUSD",
        "USDC",
        "TUSD",
        "BUSD",
        "BTC",
        "ETH",
        "BNB",
        "EUR",
        "TRY",
        "BRL",
        "DAI",
    ]

    for item in symbols_raw if isinstance(symbols_raw, list) else []:
        if isinstance(item, str):
            symbol = item.upper()
            symbols.add(symbol)
            for quote in sorted(known_quotes, key=len, reverse=True):
                if symbol.endswith(quote) and len(symbol) > len(quote):
                    base_assets.add(symbol[: -len(quote)])
                    quote_assets.add(quote)
                    break
        elif isinstance(item, dict):
            symbol = str(item.get("symbol", "")).upper()
            base = str(item.get("baseAsset", item.get("base_asset", ""))).upper()
            quote = str(item.get("quoteAsset", item.get("quote_asset", ""))).upper()
            if symbol:
                symbols.add(symbol)
            if base:
                base_assets.add(base)
            if quote:
                quote_assets.add(quote)

    return {
        "symbols": sorted(symbols),
        "base_assets": sorted(base_assets),
        "quote_assets": sorted(quote_assets),
    }


@app.get("/api/data/klines/{symbol}/{interval}")
def get_kline_chart_data(
    symbol: str,
    interval: str,
    limit: int = 1000,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 5000")

    try:
        payload = DataManager().get_chart_data(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start=start,
            end=end,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if payload["rows"] == 0:
        raise HTTPException(status_code=404, detail="No chart data found")
    return payload


@app.delete("/api/data/{symbol}")
def delete_symbol_data(symbol: str):
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")
    deleted_count = DataManager().delete_symbol_data(symbol)
    return {"status": "deleted", "paths_removed": deleted_count}


@app.get("/api/backtest/{symbol}")
def run_backtest_api(
    symbol: str,
    start: str = "2021-01-01",
    end: str = "2024-06-30",
    model_type: str | None = None,
    timestamp: str | None = None,
    interval: str | None = None,
    horizon: str = "short",
    capital: float = 10000.0,
) -> dict[str, Any]:
    """Simulate one prediction run and return its capital curve and stats.

    The run identity is part of the response: a backtest number is meaningless
    unless you know which model produced the signals behind it.
    """
    symbol = symbol.upper()
    if not _is_valid_symbol(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    settings = get_settings()
    # Import lazily to avoid pulling heavy ML/runtime deps during module import
    from prosper.eval.backtest import run_backtest

    try:
        results = run_backtest(
            symbol,
            start,
            end,
            initial_capital=capital,
            settings=settings,
            model_type=model_type,
            timestamp=timestamp,
            interval=interval,
            horizon=horizon,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])

    history = results.get("chart_data", [])

    return {
        "run": results["run"],
        "horizon": results["horizon"],
        "roi": round(results["roi_pct"], 2),
        "max_drawdown": round(results["max_drawdown_pct"], 2),
        "final_capital": round(results["final_value"], 2),
        "initial_capital": results["initial_capital"],
        "total_trades": results["total_trades"],
        "days_tested": results["days_tested"],
        "skipped_untrained": results["skipped_untrained"],
        "limitations": results["limitations"],
        "chart_data": {
            "dates": [h["date"] for h in history],
            "capital": [h["value"] for h in history],
            "close": [h["price"] for h in history],
        },
    }


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Expires"] = "0"
        return resp


# Mount Frontend App
app.mount("/", NoCacheStaticFiles(directory=str(frontend_dir), html=True), name="frontend")

