"""HTTP routes, grouped by the resource they expose."""

from prosper.api.routes.backtest import router as backtest_router
from prosper.api.routes.data import router as data_router
from prosper.api.routes.evaluations import router as evaluations_router
from prosper.api.routes.meta import router as meta_router
from prosper.api.routes.pipeline import router as pipeline_router
from prosper.api.routes.runs import router as runs_router
from prosper.api.routes.tasks import router as tasks_router

ALL_ROUTERS = (
    tasks_router,
    data_router,
    meta_router,
    runs_router,
    pipeline_router,
    evaluations_router,
    backtest_router,
)

__all__ = ["ALL_ROUTERS"]
