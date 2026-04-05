from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_factory.core.instances.models import InstanceManifest, MetricPoint
from ai_factory.core.orchestration.models import (
    AgentType,
    CircuitStatus,
    ResourceClass,
    RunStatus,
    TaskStatus,
    TaskType,
)
from ai_factory.core.plugins.base import PluginDescriptor


class InstanceLogsView(BaseModel):
    stdout: str = ""
    stderr: str = ""
    stdout_path: str | None = None
    stderr_path: str | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class InstanceMetricsView(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    points: list[MetricPoint | dict[str, Any]] = Field(default_factory=list)
    points_truncated: bool = False


class OrchestrationRuntime(BaseModel):
    run_id: str | None = None
    run_status: RunStatus | None = None
    run_updated_at: str | None = None
    task_id: str | None = None
    task_type: TaskType | None = None
    agent_type: AgentType | None = None
    task_status: TaskStatus | None = None
    resource_class: ResourceClass | None = None
    current_attempt: int = 0
    max_attempts: int = 0
    attempt_count: int = 0
    dependency_count: int = 0
    dependency_ready_count: int = 0
    dependency_blocked_count: int = 0
    retryable: bool = False
    next_retry_at: str | None = None
    checkpoint_hint: str | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None
    lease_owner: str | None = None
    last_heartbeat_at: str | None = None
    last_event_type: str | None = None
    last_event_at: str | None = None


class OrchestrationCircuitSnapshot(BaseModel):
    agent_type: AgentType
    status: CircuitStatus = "closed"
    failure_count: int = 0
    opened_at: str | None = None
    reopen_after: str | None = None
    last_error: str | None = None
    updated_at: str | None = None
    is_open: bool = False
    is_half_open: bool = False


class ManagedInstanceDetail(InstanceManifest):
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    logs: InstanceLogsView = Field(default_factory=InstanceLogsView)
    metrics: InstanceMetricsView = Field(default_factory=InstanceMetricsView)
    orchestration_runtime: OrchestrationRuntime = Field(default_factory=OrchestrationRuntime)
    children: list[InstanceManifest] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    available_actions: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    orchestration_events: list[dict[str, Any]] = Field(default_factory=list)
    orchestration_summary: dict[str, Any] = Field(default_factory=dict)


class LiveInstanceSnapshot(BaseModel):
    instance: InstanceManifest
    logs: InstanceLogsView = Field(default_factory=InstanceLogsView)
    metrics: InstanceMetricsView = Field(default_factory=InstanceMetricsView)
    orchestration_runtime: OrchestrationRuntime = Field(default_factory=OrchestrationRuntime)
    events: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    available_actions: list[dict[str, Any]] = Field(default_factory=list)
    orchestration_summary: dict[str, Any] = Field(default_factory=dict)


class InstanceStreamFrame(BaseModel):
    sequence: int
    emitted_at: str
    snapshot: LiveInstanceSnapshot


class FoundationInterface(BaseModel):
    id: str
    label: str
    transport: str
    entrypoint: str
    status: Literal["available", "planned"] = "available"
    shared_backend: bool = True


class FoundationOverview(BaseModel):
    repo_root: str
    artifacts_dir: str
    control_db_path: str
    summary: dict[str, Any] = Field(default_factory=dict)
    interfaces: list[FoundationInterface] = Field(default_factory=list)
    plugins: list[PluginDescriptor] = Field(default_factory=list)
