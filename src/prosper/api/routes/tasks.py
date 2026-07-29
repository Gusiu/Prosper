"""Runner status, queue management and log streaming."""

from typing import Any

from fastapi import APIRouter, HTTPException

from prosper.api.tasks import runner

router = APIRouter()


@router.get("/api/status")
def get_status() -> dict[str, str]:
    return {"status": "online", "message": "Prosper Engine is operational"}


@router.get("/api/task/logs")
def get_task_logs(start_idx: int = 0) -> dict[str, Any]:
    lines, next_idx = runner.read_logs(start_idx)
    return {
        "logs": lines,
        "next_idx": next_idx,
        "dropped": runner.dropped_log_lines,
        **runner.snapshot(),
    }


@router.get("/api/task/status")
def get_task_status() -> dict[str, Any]:
    return runner.snapshot()


@router.get("/api/task/queue")
def get_task_queue() -> dict[str, Any]:
    """The authoritative queue. The browser renders this rather than its own copy."""
    return runner.snapshot()


@router.delete("/api/task/queue/{task_id}")
def cancel_task(task_id: int) -> dict[str, Any]:
    """Drop a pending task. The one already running is not interrupted."""
    if not runner.cancel(task_id):
        raise HTTPException(
            status_code=404, detail=f"Task {task_id} is not pending (finished, or running)"
        )
    return {"status": "cancelled", **runner.snapshot()}


@router.post("/api/task/queue/{task_id}/move")
def move_task(task_id: int, direction: int) -> dict[str, Any]:
    """Shift a pending task one slot earlier (-1) or later (+1)."""
    if direction not in (-1, 1):
        raise HTTPException(status_code=400, detail="direction must be -1 or 1")
    if not runner.move(task_id, direction):
        raise HTTPException(status_code=404, detail=f"Task {task_id} cannot move there")
    return {"status": "moved", **runner.snapshot()}
