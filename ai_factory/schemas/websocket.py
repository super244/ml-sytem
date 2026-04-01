from typing import Literal, Union
from pydantic import BaseModel
from ai_factory.schemas.cluster import GPUMetric, NodeStatus


class GPUTelemetryMessage(BaseModel):
    type: Literal["gpu_telemetry"] = "gpu_telemetry"
    node_id: str
    timestamp: float
    gpus: list[GPUMetric]


class JobUpdateMessage(BaseModel):
    type: Literal["job_update"] = "job_update"
    job_id: str
    timestamp: float
    progress: float
    current_step: int
    current_loss: float | None = None
    loss_delta: float | None = None
    eta_seconds: int | None = None
    status: str


class JobCompleteMessage(BaseModel):
    type: Literal["job_complete"] = "job_complete"
    job_id: str
    timestamp: float
    final_loss: float
    eval_scores: dict[str, float]
    checkpoint_path: str
    model_id: str


class JobFailedMessage(BaseModel):
    type: Literal["job_failed"] = "job_failed"
    job_id: str
    timestamp: float
    error: str
    step_failed_at: int | None = None


class AgentDecisionMessage(BaseModel):
    type: Literal["agent_decision"] = "agent_decision"
    agent_id: str
    agent_type: str
    action: str
    target_id: str
    reasoning: str
    timestamp: float


class LogLineMessage(BaseModel):
    type: Literal["log_line"] = "log_line"
    source: str
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "AGENT", "METRIC"]
    message: str
    job_id: str | None = None
    timestamp: float


class ClusterUpdateMessage(BaseModel):
    type: Literal["cluster_update"] = "cluster_update"
    nodes: list[NodeStatus]
    timestamp: float


class AutoMLUpdateMessage(BaseModel):
    type: Literal["automl_update"] = "automl_update"
    search_id: str
    run_id: str
    status: str
    eval_loss: float | None = None
    composite_score: float | None = None
    timestamp: float


TelemetryMessage = Union[
    GPUTelemetryMessage,
    JobUpdateMessage,
    JobCompleteMessage,
    JobFailedMessage,
    AgentDecisionMessage,
    LogLineMessage,
    ClusterUpdateMessage,
    AutoMLUpdateMessage,
]
