from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


InstanceType = Literal["prepare", "train", "finetune", "evaluate", "inference", "deploy", "report"]
InstanceStatus = Literal["pending", "running", "completed", "failed"]
EnvironmentKind = Literal["local", "cloud"]
DecisionAction = Literal["retrain", "finetune", "deploy"]
UserLevel = Literal["beginner", "hobbyist", "dev"]
OrchestrationMode = Literal["single", "local_parallel", "cloud_parallel", "hybrid"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EnvironmentSpec(BaseModel):
    kind: EnvironmentKind = "local"
    profile_name: str | None = None
    host: str | None = None
    user: str | None = None
    port: int = 22
    key_path: str | None = None
    agent_socket: str | None = None
    remote_repo_root: str | None = None
    python_bin: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    port_forwards: list["PortForward"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_cloud_fields(self) -> "EnvironmentSpec":
        if self.kind == "cloud" and not (self.profile_name or self.host):
            raise ValueError("cloud environments require either a profile_name or host")
        return self


class PortForward(BaseModel):
    local_port: int
    remote_port: int
    bind_host: str = "127.0.0.1"
    description: str | None = None


class ExecutionHandle(BaseModel):
    backend: str
    pid: int | None = None
    remote_job_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    remote_stdout_path: str | None = None
    remote_stderr_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricPoint(BaseModel):
    timestamp: str = Field(default_factory=utc_now_iso)
    name: str
    value: float | int | bool | None
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionResult(BaseModel):
    action: DecisionAction
    rule: str
    thresholds: dict[str, float | int | None] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    explanation: str


class FeedbackRecommendation(BaseModel):
    action: str
    reason: str
    priority: int = 1
    target_instance_type: InstanceType | None = None
    config_path: str | None = None
    deployment_target: str | None = None
    command: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstanceError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ProgressSnapshot(BaseModel):
    stage: str
    status_message: str | None = None
    completed_steps: int | None = None
    total_steps: int | None = None
    percent: float | None = None
    eta_seconds: float | None = None
    updated_at: str = Field(default_factory=utc_now_iso)
    metrics: dict[str, Any] = Field(default_factory=dict)


class InstanceManifest(BaseModel):
    id: str
    type: InstanceType
    status: InstanceStatus = "pending"
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    user_level: UserLevel = "hobbyist"
    orchestration_mode: OrchestrationMode = "single"
    name: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    parent_instance_id: str | None = None
    config_path: str | None = None
    config_snapshot_path: str | None = None
    artifact_refs: dict[str, Any] = Field(default_factory=dict)
    execution: ExecutionHandle | None = None
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    progress: ProgressSnapshot | None = None
    decision: DecisionResult | None = None
    recommendations: list[FeedbackRecommendation] = Field(default_factory=list)
    error: InstanceError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()
