"""Request models for the structured API.

The frontend posts these; the server is the only place that knows how to turn
an intent into a CLI invocation.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


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


