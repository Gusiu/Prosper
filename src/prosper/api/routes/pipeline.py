"""Endpoints that submit work to the background runner."""

from fastapi import APIRouter, HTTPException

from prosper.api.commands import (
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
    TrainRequest,
)
from prosper.api.tasks import runner
from prosper.inventory import DataManager

router = APIRouter()


@router.post("/api/pipeline/run")
def run_pipeline(req: PipelineRequest):
    """
    Start a data pipeline (backfill -> aggregate -> features -> labels).
    """
    try:
        commands = build_pipeline_command(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task = runner.submit(f"Pipeline [{req.symbol.upper()}]", commands)
    return {"status": "queued", **task.to_dict(), **runner.snapshot()}


@router.post("/api/train/run")
def run_training(req: TrainRequest):
    """
    Start model training.
    """
    try:
        commands = build_train_command(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task = runner.submit(
        f"Train {req.model_type.upper()} [{req.symbol.upper()}]", commands
    )
    return {"status": "queued", **task.to_dict(), **runner.snapshot()}


@router.post("/api/eval/run")
def run_evaluation(req: EvalRequest):
    """
    Start prediction evaluation.
    """
    try:
        commands = build_eval_command(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task = runner.submit(
        f"Evaluate {req.model_type} [{req.symbol.upper()}] {req.timestamp}", commands
    )
    return {"status": "queued", **task.to_dict(), **runner.snapshot()}


@router.post("/api/pipeline/aggregate")
def run_aggregate(req: AggregateRequest):
    """Aggregate 1m into the chosen intervals and rebuild features and labels."""
    try:
        commands = build_aggregate_command(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    task = runner.submit(f"Aggregate [{req.symbol.upper()}]", commands)
    return {"status": "queued", **task.to_dict(), **runner.snapshot()}


@router.post("/api/pipeline/repair")
def run_repair(req: RepairRequest):
    """Repair a symbol; the server derives the steps from the current inventory."""
    inventory = DataManager().get_inventory()
    try:
        commands, steps = build_repair_commands(req, inventory)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not commands:
        raise HTTPException(
            status_code=400, detail=f"Nothing to repair for {req.symbol.upper()}"
        )

    task = runner.submit(f"Fix Quality [{req.symbol.upper()}]", commands)
    return {"status": "queued", "steps_explained": steps, **task.to_dict(), **runner.snapshot()}


@router.post("/api/eval/batch")
def run_batch_evaluation(req: BatchEvalRequest):
    """
    Start batch evaluation of model runs.
    """
    try:
        commands = build_batch_eval_command(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task = runner.submit("Batch evaluation", commands)
    return {"status": "queued", **task.to_dict(), **runner.snapshot()}
