from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


AgentType = Literal[
    "data_processing",
    "training_orchestration",
    "evaluation_benchmarking",
    "monitoring_telemetry",
    "optimization_feedback",
    "deployment",
]
RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
TaskStatus = Literal[
    "queued",
    "ready",
    "running",
    "completed",
    "failed",
    "retry_waiting",
    "cancelled",
    "dead_lettered",
    "blocked",
]
AttemptStatus = Literal["running", "completed", "failed", "cancelled"]
CircuitStatus = Literal["closed", "open", "half_open"]
ResourceClass = Literal["control", "cpu", "gpu", "io", "network"]
TaskKind = Literal[
    "prepare",
    "train",
    "finetune",
    "evaluate",
    "report",
    "inference",
    "deploy",
    "monitor",
    "optimize",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RetryPolicy(BaseModel):
    max_attempts: int = 1
    base_delay_s: int = 5
    max_delay_s: int = 300
    multiplier: float = 2.0
    jitter_s: float = 0.25


class AgentCapability(BaseModel):
    agent_type: AgentType
    task_types: list[TaskKind] = Field(default_factory=list)
    resource_classes: list[ResourceClass] = Field(default_factory=list)
    max_concurrency: int = 1
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


class TaskInputEnvelope(BaseModel):
    task_type: TaskKind
    legacy_instance_id: str | None = None
    config_path: str | None = None
    command: list[str] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    labels: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None
    resource_class: ResourceClass = "control"


class TaskOutputEnvelope(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    status_updates: list[dict[str, Any]] = Field(default_factory=list)
    checkpoint_hint: str | None = None
    recommendations: list[dict[str, Any]] = Field(default_factory=list)


class OrchestrationRun(BaseModel):
    id: str
    legacy_instance_id: str | None = None
    name: str
    status: RunStatus = "queued"
    root_run_id: str | None = None
    parent_run_id: str | None = None
    idempotency_key: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrchestrationTask(BaseModel):
    id: str
    run_id: str
    legacy_instance_id: str | None = None
    parent_task_id: str | None = None
    task_type: TaskKind
    agent_type: AgentType
    status: TaskStatus = "queued"
    priority: int = 100
    resource_class: ResourceClass = "control"
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    current_attempt: int = 0
    last_error_code: str | None = None
    last_error_message: str | None = None
    checkpoint_hint: str | None = None
    queued_at: str = Field(default_factory=utc_now_iso)
    available_at: str = Field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    input: TaskInputEnvelope
    output: TaskOutputEnvelope = Field(default_factory=TaskOutputEnvelope)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskDependency(BaseModel):
    task_id: str
    depends_on_task_id: str
    created_at: str = Field(default_factory=utc_now_iso)


class TaskAttempt(BaseModel):
    id: str
    task_id: str
    sequence: int
    status: AttemptStatus = "running"
    lease_owner: str | None = None
    started_at: str = Field(default_factory=utc_now_iso)
    heartbeat_at: str = Field(default_factory=utc_now_iso)
    finished_at: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    exit_code: int | None = None
    checkpoint_hint: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskLease(BaseModel):
    task_id: str
    attempt_id: str
    lease_owner: str
    acquired_at: str = Field(default_factory=utc_now_iso)
    heartbeat_at: str = Field(default_factory=utc_now_iso)
    expires_at: str


class CircuitState(BaseModel):
    agent_type: AgentType
    status: CircuitStatus = "closed"
    failure_count: int = 0
    opened_at: str | None = None
    reopen_after: str | None = None
    last_error: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class OrchestrationEvent(BaseModel):
    id: str
    run_id: str
    task_id: str | None = None
    attempt_id: str | None = None
    event_type: str
    level: Literal["debug", "info", "warning", "error"] = "info"
    agent_type: AgentType | None = None
    message: str
    created_at: str = Field(default_factory=utc_now_iso)
    payload: dict[str, Any] = Field(default_factory=dict)
