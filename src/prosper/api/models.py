"""Request models for the structured API.

The frontend posts these; the server is the only place that knows how to turn
an intent into a CLI invocation.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from prosper.predict.defaults import HORIZON_NAMES


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
    # No default of its own. Sending a value the caller never chose is what
    # made the dashboard train 5 epochs where the CLI trained 20: Typer's
    # default only applies when the flag is absent, and the API used to append
    # it unconditionally. `None` means "say nothing and let the CLI decide".
    epochs: int | None = None
    epochs_by_horizon: dict[str, int] = Field(default_factory=dict)
    # Same rule as the epochs above: empty means "say nothing". The CLI derives
    # each horizon's window from its span and the history available, and an
    # unasked-for value here would override a decision the data made.
    train_window_by_horizon: dict[str, int] = Field(default_factory=dict)
    # Extra symbols pooled into the training set. Predictions are always for
    # `symbol` alone, and only the tabular predictors accept a pool — the
    # sequence models need sequences built inside a symbol first.
    pool_symbols: list[str] = Field(default_factory=list)
    horizons: list[str] = Field(default_factory=lambda: list(HORIZON_NAMES))
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


class AggregateRequest(BaseModel):
    """Aggregate 1m into the chosen intervals, then rebuild features and labels."""
    symbol: str
    start: str
    end: str
    intervals: list[str] = Field(default_factory=list)
    flags: ResearchFlags = Field(default_factory=ResearchFlags)


class RepairRequest(BaseModel):
    """Repair a symbol's data. The server derives the steps from the inventory."""
    symbol: str
    flags: ResearchFlags = Field(default_factory=ResearchFlags)


