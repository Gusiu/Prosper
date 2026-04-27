import os
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

import subprocess
import shlex

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import polars as pl
import shutil
import datetime

from prosper.config import get_settings
from prosper.storage.layout import get_prediction_report_path, get_features_parquet_path
from prosper.eval.backtest import run_backtest

import webbrowser
import threading
import time

# Resolve the absolute path dynamically based on this file's location.
# This makes the project portable to any computer or directory.
frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"

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
        
    def start(self, cmd_str: str):
        if self.is_running:
            raise HTTPException(400, "A task is already running!")
        self.logs = [f"$ {cmd_str}"]
        self.is_running = True
        
        def run_thread():
            try:
                # Use shell=True for windows poetry env execution if needed,
                # but shlex split is safer. We'll use shell=True for simplicity on Windows.
                self.process = subprocess.Popen(
                    cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                if self.process.stdout:
                    for line in iter(self.process.stdout.readline, ''):
                        if line:
                            self.logs.append(line.rstrip('\n'))
                self.process.wait()
                self.logs.append(f"Task finished with exit code {self.process.returncode}")
            except Exception as e:
                self.logs.append(f"Error executing task: {str(e)}")
            finally:
                self.is_running = False
                self.process = None

        threading.Thread(target=run_thread, daemon=True).start()

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
    if not req.command.startswith("poetry run prosper"):
        raise HTTPException(400, "Invalid command. Must start with 'poetry run prosper'")
    runner.start(req.command)
    return {"status": "started", "command": req.command}

@app.get("/api/task/logs")
def get_task_logs(start_idx: int = 0):
    return {"logs": runner.logs[start_idx:], "is_running": runner.is_running}

@app.get("/api/task/status")
def get_task_status():
    return {"is_running": runner.is_running}


@app.get("/api/data/dates")
def get_dates():
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    
    return {
        "today": today.strftime("%Y-%m-%d"),
        "yesterday": yesterday.strftime("%Y-%m-%d"),
        "default_start_month": "2017-08",
        "yesterday_month": yesterday.strftime("%Y-%m")
    }

@app.get("/api/data/inventory")
def get_inventory():
    settings = get_settings()
    inventory = []
    symbols = set()
    
    raw_dir = settings.raw_binance_spot_klines_1m_dir
    if raw_dir.exists():
        for d in raw_dir.iterdir():
            if d.is_dir():
                symbols.add(d.name)
                
    processed_dir = settings.processed_binance_spot_klines_dir
    if processed_dir.exists():
        for interval_dir in processed_dir.iterdir():
            if interval_dir.is_dir():
                for sym_dir in interval_dir.glob("symbol=*"):
                    symbols.add(sym_dir.name.split("=")[1])
                    
    for symbol in sorted(list(symbols)):
        size_bytes = 0
        
        raw_sym = raw_dir / symbol
        if raw_sym.exists():
            size_bytes += sum(f.stat().st_size for f in raw_sym.rglob("*") if f.is_file())
            
        aggregations = []
        if processed_dir.exists():
            for interval_dir in processed_dir.iterdir():
                if interval_dir.is_dir():
                    sym_path = interval_dir / f"symbol={symbol}"
                    if sym_path.exists():
                        size_bytes += sum(f.stat().st_size for f in sym_path.rglob("*") if f.is_file())
                        aggregations.append(interval_dir.name)
                    
        feat_path = settings.processed_data_dir / "binance" / "spot" / "features" / "1d" / f"symbol={symbol}"
        if feat_path.exists():
            size_bytes += sum(f.stat().st_size for f in feat_path.rglob("*") if f.is_file())
            
        pred_path = settings.reports_predictions_dir / symbol
        if pred_path.exists():
            size_bytes += sum(f.stat().st_size for f in pred_path.rglob("*") if f.is_file())
            
        start_date = "N/A"
        end_date = "N/A"
        
        try:
            klines_1m_path = processed_dir / "1m" / f"symbol={symbol}"
            if klines_1m_path.exists():
                years = [d.name for d in klines_1m_path.iterdir() if d.is_dir() and d.name.startswith("year=")]
                if years:
                    years.sort()
                    min_year = years[0].split("=")[1]
                    max_year = years[-1].split("=")[1]
                    
                    min_months = [d.name for d in (klines_1m_path / years[0]).iterdir() if d.is_dir() and d.name.startswith("month=")]
                    max_months = [d.name for d in (klines_1m_path / years[-1]).iterdir() if d.is_dir() and d.name.startswith("month=")]
                    
                    if min_months and max_months:
                        min_months.sort()
                        max_months.sort()
                        min_month = min_months[0].split("=")[1]
                        max_month = max_months[-1].split("=")[1]
                        
                        start_date = f"{min_year}-{min_month}-01"
                        end_date = f"{max_year}-{max_month}-28"
        except Exception:
            pass

        if size_bytes == 0 and not aggregations:
            continue

        inventory.append({
            "symbol": symbol,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "aggregations": sorted(list(aggregations)),
            "start_date": start_date,
            "end_date": end_date
        })
        
    return inventory

@app.delete("/api/data/{symbol}")
def delete_symbol_data(symbol: str):
    settings = get_settings()
    
    # Broad search for anything related to this symbol
    # 1. Standard locations
    paths = [
        settings.raw_binance_spot_klines_1m_dir / symbol,
        settings.reports_predictions_dir / symbol,
        settings.processed_data_dir / "binance" / "spot" / "features" / f"symbol={symbol}",
        settings.processed_binance_spot_labels_dir / f"symbol={symbol}"
    ]
    
    # 2. All interval folders
    for parent_dir in [
        settings.processed_binance_spot_klines_dir,
    ]:
        if parent_dir.exists():
            for interval_dir in parent_dir.iterdir():
                if interval_dir.is_dir():
                    paths.append(interval_dir / f"symbol={symbol}")
                    # Also check for just symbol name if it's not partitioned
                    paths.append(interval_dir / symbol)
        
    deleted_count = 0
    for p in paths:
        try:
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                deleted_count += 1
        except Exception as e:
            print(f"Error deleting {p}: {e}")
            
    return {"status": "deleted", "paths_removed": deleted_count}


@app.get("/api/backtest/{symbol}")
def run_backtest_api(symbol: str, start: str = "2021-01-01", end: str = "2024-06-30") -> dict[str, Any]:
    """Dynamically run backtest and return capital curve & stats."""
    settings = get_settings()
    try:
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
                "close": [h["price"] for h in history]
            }
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
