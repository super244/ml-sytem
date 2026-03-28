from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_factory.core.instances.models import InstanceManifest, MetricPoint
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


class ManagedInstanceDetail(InstanceManifest):
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    logs: InstanceLogsView = Field(default_factory=InstanceLogsView)
    metrics: InstanceMetricsView = Field(default_factory=InstanceMetricsView)
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
    interfaces: list[FoundationInterface] = Field(default_factory=list)
    plugins: list[PluginDescriptor] = Field(default_factory=list)
