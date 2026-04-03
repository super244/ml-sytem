from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, TypeVar

from ai_factory.core.control.models import (
    FoundationInterface,
    FoundationOverview,
    InstanceLogsView,
    InstanceMetricsView,
    InstanceStreamFrame,
    LiveInstanceSnapshot,
    ManagedInstanceDetail,
)
from ai_factory.core.instances.models import (
    DeploymentTarget,
    EnvironmentSpec,
    InstanceManifest,
    InstanceStatus,
    InstanceType,
    LifecycleProfile,
    UserLevel,
    utc_now_iso,
)
from ai_factory.core.lineage.models import LineageRecord
from ai_factory.core.plugins.base import PluginDescriptor

if TYPE_CHECKING:
    from ai_factory.core.instances.manager import InstanceManager
    from ai_factory.core.instances.store import FileInstanceStore
    from ai_factory.core.lineage.registry import LineageRegistry
    from ai_factory.core.platform.settings import PlatformSettings
    from ai_factory.core.plugins.registry import PluginRegistry


def _tail_text(value: str, max_chars: int | None) -> tuple[str, bool]:
    if max_chars is None or max_chars <= 0 or len(value) <= max_chars:
        return value, False
    return value[-max_chars:], True


T = TypeVar("T")


def _tail_list(values: list[T], limit: int | None) -> tuple[list[T], bool]:
    if limit is None or limit <= 0 or len(values) <= limit:
        return list(values), False
    return list(values[-limit:]), True


class FactoryControlService:
    def __init__(
        self,
        *,
        manager: InstanceManager,
        store: FileInstanceStore,
        settings: PlatformSettings,
        plugin_registry: PluginRegistry,
        lineage_registry: LineageRegistry,
    ) -> None:
        self.manager = manager
        self.store = store
        self.settings = settings
        self.plugin_registry = plugin_registry
        self.lineage_registry = lineage_registry

    def record_lineage(self, record: LineageRecord) -> None:
        """Register a new lineage record."""
        self.lineage_registry.record_lineage(record)

    def get_lineage(self, record_id: str) -> LineageRecord | None:
        """Fetch a lineage record."""
        return self.lineage_registry.get_lineage(record_id)

    def list_lineage(self, *, limit: int = 50) -> list[LineageRecord]:
        """List lineage records."""
        return self.lineage_registry.list_lineage(limit=limit)

    def create_instance(
        self,
        config_path: str,
        *,
        start: bool = True,
        environment_override: EnvironmentSpec | None = None,
        parent_instance_id: str | None = None,
        name_override: str | None = None,
        user_level_override: UserLevel | None = None,
        lifecycle_override: LifecycleProfile | None = None,
        subsystem_updates: dict[str, Any] | None = None,
        execution_updates: dict[str, Any] | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        return self.manager.create_instance(
            config_path,
            start=start,
            environment_override=environment_override,
            parent_instance_id=parent_instance_id,
            name_override=name_override,
            user_level_override=user_level_override,
            lifecycle_override=lifecycle_override,
            subsystem_updates=subsystem_updates,
            execution_updates=execution_updates,
            metadata_updates=metadata_updates,
        )

    def create_evaluation_instance(
        self,
        source_instance_id: str,
        *,
        config_path: str = "configs/eval.yaml",
        start: bool = True,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        return self.manager.create_evaluation_instance(
            source_instance_id,
            config_path=config_path,
            start=start,
            metadata_updates=metadata_updates,
        )

    def create_deployment_instance(
        self,
        source_instance_id: str,
        *,
        target: DeploymentTarget,
        config_path: str = "configs/deploy.yaml",
        start: bool = True,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        return self.manager.create_deployment_instance(
            source_instance_id,
            target=target,
            config_path=config_path,
            start=start,
            metadata_updates=metadata_updates,
        )

    def create_inference_instance(
        self,
        source_instance_id: str,
        *,
        config_path: str = "configs/inference.yaml",
        start: bool = True,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InstanceManifest:
        return self.manager.create_inference_instance(
            source_instance_id,
            config_path=config_path,
            start=start,
            metadata_updates=metadata_updates,
        )

    def execute_action(
        self,
        instance_id: str,
        *,
        action: str,
        config_path: str | None = None,
        deployment_target: DeploymentTarget | None = None,
        start: bool = True,
    ) -> InstanceManifest:
        return self.manager.execute_action(
            instance_id,
            action=action,
            config_path=config_path,
            deployment_target=deployment_target,
            start=start,
        )

    def retry_instance(self, instance_id: str) -> InstanceManifest:
        return self.manager.retry_instance(instance_id)

    def cancel_instance(self, instance_id: str) -> InstanceManifest:
        return self.manager.cancel_instance(instance_id)

    def watch_instance(self, instance_id: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
        return self.manager.watch_instance(instance_id, timeout_s=timeout_s)

    def get_instance(self, instance_id: str) -> InstanceManifest:
        return self.manager.get_instance(instance_id)

    def list_instances(
        self,
        *,
        instance_type: InstanceType | None = None,
        status: InstanceStatus | None = None,
        parent_instance_id: str | None = None,
    ) -> list[InstanceManifest]:
        return self.manager.list_instances(
            instance_type=instance_type,
            status=status,
            parent_instance_id=parent_instance_id,
        )

    def get_children(self, instance_id: str) -> list[InstanceManifest]:
        return self.manager.get_children(instance_id)

    def list_tasks(self, target: str | None = None) -> list[dict[str, Any]]:
        return self.manager.list_tasks(target)

    def list_orchestration_runs(self) -> list[dict[str, Any]]:
        return self.manager.list_orchestration_runs()

    def get_orchestration_run(self, target: str) -> dict[str, Any]:
        return self.manager.get_orchestration_run(target)

    def list_orchestration_events(self, target: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.manager.list_orchestration_events(target, limit=limit)

    def monitoring_summary(self) -> dict[str, Any]:
        return self.manager.monitoring_summary()

    def get_logs(self, instance_id: str, *, tail_chars: int | None = None) -> InstanceLogsView:
        logs = self.store.read_logs(instance_id)
        stdout, stdout_truncated = _tail_text(logs.get("stdout", ""), tail_chars)
        stderr, stderr_truncated = _tail_text(logs.get("stderr", ""), tail_chars)
        return InstanceLogsView(
            stdout=stdout,
            stderr=stderr,
            stdout_path=str(self.store.stdout_path(instance_id)),
            stderr_path=str(self.store.stderr_path(instance_id)),
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def get_metrics(self, instance_id: str, *, tail_points: int | None = None) -> InstanceMetricsView:
        metrics = self.manager.get_metrics(instance_id)
        points, points_truncated = _tail_list(list(metrics.get("points", [])), tail_points)
        return InstanceMetricsView(
            summary=metrics.get("summary", {}),
            points=points,
            points_truncated=points_truncated,
        )

    def get_instance_detail(self, instance_id: str) -> ManagedInstanceDetail:
        manifest = self.get_instance(instance_id)
        run_target = manifest.orchestration_run_id or manifest.id
        return ManagedInstanceDetail.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "config_snapshot": self.store.load_config_snapshot(instance_id),
                "logs": self.get_logs(instance_id).model_dump(mode="json"),
                "metrics": self.get_metrics(instance_id).model_dump(mode="json"),
                "children": [item.model_dump(mode="json") for item in self.get_children(instance_id)],
                "events": self.store.read_events(instance_id),
                "available_actions": self.manager.get_available_actions(instance_id),
                "tasks": self.list_tasks(run_target),
                "orchestration_events": self.list_orchestration_events(run_target, limit=200),
                "orchestration_summary": self.monitoring_summary(),
            }
        )

    def get_live_instance_snapshot(
        self,
        instance_id: str,
        *,
        log_tail_chars: int = 4000,
        metric_tail_points: int = 200,
        event_limit: int = 50,
        task_limit: int = 20,
    ) -> LiveInstanceSnapshot:
        manifest = self.get_instance(instance_id)
        run_target = manifest.orchestration_run_id or manifest.id
        tasks, _ = _tail_list(self.list_tasks(run_target), task_limit)
        events, _ = _tail_list(self.list_orchestration_events(run_target, limit=event_limit), event_limit)
        return LiveInstanceSnapshot(
            instance=manifest,
            logs=self.get_logs(instance_id, tail_chars=log_tail_chars),
            metrics=self.get_metrics(instance_id, tail_points=metric_tail_points),
            events=events,
            tasks=tasks,
            available_actions=self.manager.get_available_actions(instance_id),
            orchestration_summary=self.monitoring_summary(),
        )

    async def stream_instance(
        self,
        instance_id: str,
        *,
        poll_interval_s: float = 1.0,
        log_tail_chars: int = 4000,
        metric_tail_points: int = 200,
        event_limit: int = 50,
        task_limit: int = 20,
    ) -> AsyncIterator[InstanceStreamFrame]:
        sequence = 0
        previous_payload: str | None = None
        while True:
            snapshot = self.get_live_instance_snapshot(
                instance_id,
                log_tail_chars=log_tail_chars,
                metric_tail_points=metric_tail_points,
                event_limit=event_limit,
                task_limit=task_limit,
            )
            payload = snapshot.model_dump_json()
            if payload != previous_payload:
                sequence += 1
                yield InstanceStreamFrame(
                    sequence=sequence,
                    emitted_at=utc_now_iso(),
                    snapshot=snapshot,
                )
                previous_payload = payload
            if snapshot.instance.status in {"completed", "failed"}:
                return
            await asyncio.sleep(max(poll_interval_s, 0.25))

    def list_plugins(self) -> list[PluginDescriptor]:
        return self.plugin_registry.list_plugins()

    def describe_foundation(self) -> FoundationOverview:
        return FoundationOverview(
            repo_root=str(self.settings.repo_root),
            artifacts_dir=str(self.settings.artifacts_dir),
            control_db_path=str(self.settings.control_db_path),
            interfaces=[
                FoundationInterface(
                    id="cli",
                    label="CLI control surface",
                    transport="direct-python",
                    entrypoint="ai_factory.cli:main",
                ),
                FoundationInterface(
                    id="tui",
                    label="Terminal dashboard",
                    transport="direct-python",
                    entrypoint="ai_factory.tui:run_tui",
                ),
                FoundationInterface(
                    id="web_api",
                    label="FastAPI control API",
                    transport="http+sse",
                    entrypoint="inference.app.main:app",
                ),
                FoundationInterface(
                    id="desktop",
                    label="Desktop shell",
                    transport="http+sse",
                    entrypoint="desktop/",
                    status="planned",
                ),
            ],
            plugins=self.list_plugins(),
        )
