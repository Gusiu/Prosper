import datetime
import json
import re
import shlex
import subprocess
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from prosper.config import get_settings
from prosper.core import DataManager

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


class TaskRunner:
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.logs: list[str] = []
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
                    self.logs.append(f"$ {' '.join(original_args)}")
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
                                self.logs.append(clean_line)
                                self._update_progress_from_log(
                                    clean_line,
                                    command_index,
                                    total_commands,
                                )
                    self.process.wait()
                    self.logs.append(f"Task finished with exit code {self.process.returncode}")
                    # Płynny skok postępu po zakończeniu sub-komendy
                    self._set_command_progress(
                        command_index, total_commands, original_args, finished=True
                    )
                    if self.process.returncode != 0:
                        self.progress.update({"label": "Failed", "stage": "failed"})
                        break
            except Exception as e:
                self.logs.append(f"Error executing task: {str(e)}")
                self.progress.update({"label": "Failed", "stage": "failed"})
            finally:
                self.is_running = False
                self.process = None
                if self.progress.get("stage") != "failed":
                    self.progress.update(
                        {"percent": 100.0, "label": "Complete", "stage": "complete"}
                    )

        threading.Thread(target=run_thread, daemon=True).start()

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
    return {
        "logs": runner.logs[start_idx:],
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
    symbol: str, start: str = "2021-01-01", end: str = "2024-06-30"
) -> dict[str, Any]:
    """Dynamically run backtest and return capital curve & stats."""
    settings = get_settings()
    try:
        # Import lazily to avoid pulling heavy ML/runtime deps during module import
        from prosper.eval.backtest import run_backtest

        results = run_backtest(symbol, start, end, settings=settings)
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        history = results.get("chart_data", [])

        return {
            "roi": round(results["roi_pct"], 2),
            "max_drawdown": round(results["max_drawdown_pct"], 2),
            "final_capital": round(results["final_value"], 2),
            "total_trades": results["total_trades"],
            "chart_data": {
                "dates": [h["date"] for h in history],
                "capital": [h["value"] for h in history],
                "close": [h["price"] for h in history],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Expires"] = "0"
        return resp


# Mount Frontend App
app.mount("/", NoCacheStaticFiles(directory=str(frontend_dir), html=True), name="frontend")
