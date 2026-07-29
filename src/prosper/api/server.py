"""ASGI application: wiring only.

Request models live in :mod:`prosper.api.models`, the CLI translation in
:mod:`prosper.api.commands`, the background runner in :mod:`prosper.api.tasks`
and the endpoints in :mod:`prosper.api.routes`. This module composes them and
serves the SPA.

Names are re-exported below so `from prosper.api.server import ...` keeps
working for tests and any existing caller.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from prosper.api.commands import (
    _is_valid_symbol,
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
    ResearchFlags,
    TrainRequest,
)
from prosper.api.routes import ALL_ROUTERS
from prosper.api.routes.backtest import run_backtest_api
from prosper.api.routes.data import (
    delete_symbol_data,
    get_dates,
    get_inventory,
    get_kline_chart_data,
)
from prosper.api.routes.evaluations import (
    get_ai_evaluation_detail,
    get_ai_evaluation_flags,
    get_evaluation_batch_status,
    list_ai_evaluations,
)
from prosper.api.routes.meta import get_symbol_metadata
from prosper.api.routes.pipeline import (
    run_aggregate,
    run_batch_evaluation,
    run_evaluation,
    run_pipeline,
    run_repair,
    run_training,
)
from prosper.api.routes.runs import (
    delete_ai_model,
    get_ai_models,
    get_ai_prediction_months,
    get_ai_predictions,
    get_feature_importances,
    list_prediction_runs,
)
from prosper.api.routes.tasks import (
    cancel_task,
    get_status,
    get_task_logs,
    get_task_queue,
    get_task_status,
    move_task,
)
from prosper.api.tasks import MAX_TASK_LOG_LINES, TaskRunner, runner

# Resolve the absolute path dynamically based on this file's location.
# This makes the project portable to any computer or directory.
frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Importing this module must not have side effects beyond serving: opening
    # a browser belongs to the `prosper ui` command, not to the ASGI app, which
    # also runs under tests, uvicorn --reload and any external process manager.
    frontend_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Prosper Analysis Platform UI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for _router in ALL_ROUTERS:
    app.include_router(_router)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Expires"] = "0"
        return resp


# Mount the SPA last: it claims "/" and would otherwise shadow the API routes.
app.mount("/", NoCacheStaticFiles(directory=str(frontend_dir), html=True), name="frontend")

__all__ = [
    "MAX_TASK_LOG_LINES",
    "AggregateRequest",
    "BatchEvalRequest",
    "EvalRequest",
    "NoCacheStaticFiles",
    "PipelineRequest",
    "RepairRequest",
    "ResearchFlags",
    "TaskRunner",
    "TrainRequest",
    "_is_valid_symbol",
    "cancel_task",
    "app",
    "build_aggregate_command",
    "build_batch_eval_command",
    "build_eval_command",
    "build_pipeline_command",
    "build_repair_commands",
    "build_train_command",
    "delete_ai_model",
    "delete_symbol_data",
    "frontend_dir",
    "get_ai_evaluation_detail",
    "get_ai_evaluation_flags",
    "get_ai_models",
    "get_ai_prediction_months",
    "get_ai_predictions",
    "get_dates",
    "get_evaluation_batch_status",
    "get_feature_importances",
    "get_inventory",
    "get_kline_chart_data",
    "get_status",
    "get_symbol_metadata",
    "get_task_logs",
    "get_task_queue",
    "get_task_status",
    "list_ai_evaluations",
    "list_prediction_runs",
    "move_task",
    "run_aggregate",
    "run_backtest_api",
    "run_batch_evaluation",
    "run_evaluation",
    "run_pipeline",
    "run_repair",
    "run_training",
    "runner",
]
