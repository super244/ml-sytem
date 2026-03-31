from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict


class AutoMLRunSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    search_id: str
    status: Literal["running", "completed", "pruned", "promoted", "failed"]
    hyperparams: dict
    eval_loss: float | None = None
    composite_score: float | None = None
    training_minutes: float | None = None
    step_pruned: int | None = None
    prune_reason: str | None = None
    job_id: str | None = None


class AutoMLSearchSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    strategy: Literal["bayesian", "evolutionary", "grid", "random"]
    status: Literal["running", "paused", "completed", "failed"]
    total_runs: int
    completed_runs: int
    running_runs: int
    pruned_runs: int
    promoted_runs: int
    best_loss: float | None = None
    best_config: dict | None = None
    created_at: datetime


class AutoMLSearchDetail(AutoMLSearchSummary):
    search_space: dict
    runs: list[AutoMLRunSchema]
