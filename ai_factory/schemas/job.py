from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict


class LogLine(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    level: str
    message: str
    source: str
    timestamp: str
    job_id: str | None = None


class JobSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    type: Literal["lora_finetune", "dpo", "rlhf", "eval_only", "data_pack"]
    status: Literal["queued", "running", "completed", "failed", "stopped", "degraded"]
    base_model: str
    progress: float
    current_step: int
    total_steps: int
    current_loss: float | None = None
    loss_delta: float | None = None
    eta_seconds: int | None = None
    gpu_utilization: list[float]
    vram_used_gb: float
    vram_total_gb: float
    started_at: datetime
    estimated_completion: datetime | None = None
    node_id: str | None = None


class JobDetail(JobSummary):
    config: dict
    dataset_hash: str | None = None
    parent_model_id: str | None = None
    loss_history: list[float]
    step_history: list[int]
    eval_scores: dict[str, float] | None = None
    logs_tail: list[LogLine]
    lineage_id: str | None = None


class TrainingConfig(BaseModel):
    learning_rate: float = 2.4e-4
    epochs: int = 3
    batch_size: int = 4
    max_steps: int = 10000
    warmup_steps: int = 100
    lora_rank: int = 16
    lora_alpha: int = 32


class JobCreateRequest(BaseModel):
    type: Literal["lora_finetune", "dpo", "rlhf"]
    base_model: str
    dataset_id: str
    config: TrainingConfig = TrainingConfig()
    node_preference: str | None = None
    priority: int = 5


class JobStopResponse(BaseModel):
    status: str
    job_id: str
