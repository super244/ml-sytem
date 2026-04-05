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
    OrchestrationCircuitSnapshot,
    OrchestrationRuntime,
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


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


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

    def _control_plane(self) -> Any | None:
        orchestration = getattr(self.manager, "orchestration", None)
        return getattr(orchestration, "control_plane", None)

    def _list_task_attempts(self, task_id: str) -> list[Any]:
        control_plane = self._control_plane()
        if control_plane is None or not hasattr(control_plane, "list_attempts"):
            return []
        return list(control_plane.list_attempts(task_id))

    def _list_task_dependencies(self, task_id: str) -> list[Any]:
        control_plane = self._control_plane()
        if control_plane is None or not hasattr(control_plane, "list_dependencies"):
            return []
        return list(control_plane.list_dependencies(task_id))

    def _list_circuits(self) -> list[OrchestrationCircuitSnapshot]:
        control_plane = self._control_plane()
        if control_plane is None or not hasattr(control_plane, "get_circuit"):
            return []
        circuit_types = (
            "data_processing",
            "training_orchestration",
            "evaluation_benchmarking",
            "monitoring_telemetry",
            "optimization_feedback",
            "deployment",
        )
        snapshots: list[OrchestrationCircuitSnapshot] = []
        for agent_type in circuit_types:
            circuit = control_plane.get_circuit(agent_type)
            if circuit is None:
                continue
            snapshots.append(
                OrchestrationCircuitSnapshot(
                    agent_type=circuit.agent_type,
                    status=circuit.status,
                    failure_count=circuit.failure_count,
                    opened_at=circuit.opened_at,
                    reopen_after=circuit.reopen_after,
                    last_error=circuit.last_error,
                    updated_at=circuit.updated_at,
                    is_open=circuit.status == "open",
                    is_half_open=circuit.status == "half_open",
                )
            )
        return snapshots

    def _runtime_summary(
        self,
        manifest: InstanceManifest,
        *,
        tasks: list[Any] | None = None,
        events: list[Any] | None = None,
    ) -> OrchestrationRuntime:
        run_target = manifest.orchestration_run_id or manifest.id
        control_plane = self._control_plane()
        run = None
        if control_plane is not None:
            if hasattr(control_plane, "get_run"):
                run = control_plane.get_run(run_target)
            if run is None and hasattr(control_plane, "get_run_by_legacy_instance"):
                run = control_plane.get_run_by_legacy_instance(run_target)

        task_objects = tasks
        if task_objects is None:
            task_objects = []
            orchestration = getattr(self.manager, "orchestration", None)
            if orchestration is not None and hasattr(orchestration, "list_tasks"):
                task_objects = list(orchestration.list_tasks(run_target))

        task = next(
            (item for item in task_objects if _value(item, "legacy_instance_id") == manifest.id),
            None,
        )
        if task is None and task_objects:
            task = task_objects[0]

        task_id = _value(task, "id") if task is not None else None
        attempts = self._list_task_attempts(task_id) if task_id is not None else []
        latest_attempt = attempts[-1] if attempts else None
        dependencies = self._list_task_dependencies(task_id) if task_id is not None else []

        task_events = events
        if task_events is None:
            task_events = []
            if run is not None:
                task_events = list(self.list_orchestration_events(run.id, limit=25))
        latest_event = task_events[-1] if task_events else None

        completed_task_ids = {
            _value(item, "id")
            for item in task_objects
            if _value(item, "status") == "completed"
        }
        dependency_ids = [_value(dependency, "depends_on_task_id") for dependency in dependencies]
        ready_dependency_count = sum(1 for dependency_id in dependency_ids if dependency_id in completed_task_ids)

        task_status = _value(task, "status")
        retry_policy = _value(task, "retry_policy") if task is not None else None
        current_attempt = int(_value(task, "current_attempt", 0) or 0) if task is not None else 0
        max_attempts = int(_value(retry_policy, "max_attempts", 0) or 0) if retry_policy is not None else 0
        retryable = bool(task is not None and current_attempt < max_attempts)
        next_retry_at = _value(task, "available_at") if task_status == "retry_waiting" else None

        checkpoint_hint = _value(task, "checkpoint_hint") if task is not None else None
        last_error_code = _value(task, "last_error_code") if task is not None else None
        last_error_message = _value(task, "last_error_message") if task is not None else None
        lease_owner = None
        last_heartbeat_at = None
        if latest_attempt is not None:
            lease_owner = _value(latest_attempt, "lease_owner")
            last_heartbeat_at = _value(latest_attempt, "heartbeat_at")
            checkpoint_hint = checkpoint_hint or _value(latest_attempt, "checkpoint_hint")
            last_error_code = last_error_code or _value(latest_attempt, "error_code")
            last_error_message = last_error_message or _value(latest_attempt, "error_message")

        return OrchestrationRuntime(
            run_id=_value(run, "id"),
            run_status=_value(run, "status"),
            run_updated_at=_value(run, "updated_at"),
            task_id=_value(task, "id"),
            task_type=_value(task, "task_type"),
            agent_type=_value(task, "agent_type"),
            task_status=task_status,
            resource_class=_value(task, "resource_class"),
            current_attempt=current_attempt,
            max_attempts=max_attempts,
            attempt_count=len(attempts),
            dependency_count=len(dependency_ids),
            dependency_ready_count=ready_dependency_count,
            dependency_blocked_count=len(dependency_ids) - ready_dependency_count,
            retryable=retryable,
            next_retry_at=next_retry_at,
            checkpoint_hint=checkpoint_hint,
            last_error_code=last_error_code,
            last_error_message=last_error_message,
            lease_owner=lease_owner,
            last_heartbeat_at=last_heartbeat_at,
            last_event_type=_value(latest_event, "event_type"),
            last_event_at=_value(latest_event, "created_at"),
        )

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
        base = dict(self.manager.monitoring_summary())
        control_plane = self._control_plane()
        if control_plane is not None and hasattr(control_plane, "list_stale_leases"):
            stale_task_ids: list[str] = []
            for lease in control_plane.list_stale_leases(stale_before=utc_now_iso()):
                task_id = _value(lease, "task_id")
                if task_id is None:
                    continue
                stale_task_ids.append(str(task_id))
        else:
            stale_task_ids = []
        circuits = self._list_circuits()
        status_counts = dict(base.get("task_status_counts", {}))
        base.update(
            {
                "completed_tasks": status_counts.get("completed", 0),
                "failed_tasks": status_counts.get("failed", 0),
                "retry_waiting_tasks": status_counts.get("retry_waiting", 0),
                "blocked_tasks": status_counts.get("blocked", 0),
                "dead_lettered_tasks": status_counts.get("dead_lettered", 0),
                "stale_task_ids": stale_task_ids,
                "stale_task_count": len(stale_task_ids),
                "circuits": [circuit.model_dump(mode="json") for circuit in circuits],
                "open_circuits": [circuit.agent_type for circuit in circuits if circuit.is_open],
                "half_open_circuits": [circuit.agent_type for circuit in circuits if circuit.is_half_open],
            }
        )
        return base

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
        tasks = self.list_tasks(run_target)
        events = self.list_orchestration_events(run_target, limit=200)
        return ManagedInstanceDetail.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "config_snapshot": self.store.load_config_snapshot(instance_id),
                "logs": self.get_logs(instance_id).model_dump(mode="json"),
                "metrics": self.get_metrics(instance_id).model_dump(mode="json"),
                "orchestration_runtime": self._runtime_summary(manifest, tasks=tasks, events=events).model_dump(
                    mode="json"
                ),
                "children": [item.model_dump(mode="json") for item in self.get_children(instance_id)],
                "events": self.store.read_events(instance_id),
                "available_actions": self.manager.get_available_actions(instance_id),
                "tasks": tasks,
                "orchestration_events": events,
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
            orchestration_runtime=self._runtime_summary(manifest, tasks=tasks, events=events),
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
        interfaces = [
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
        ]
        plugins = self.list_plugins()
        return FoundationOverview(
            repo_root=str(self.settings.repo_root),
            artifacts_dir=str(self.settings.artifacts_dir),
            control_db_path=str(self.settings.control_db_path),
            summary={
                "interface_count": len(interfaces),
                "available_interface_count": len([item for item in interfaces if item.status == "available"]),
                "planned_interface_count": len([item for item in interfaces if item.status == "planned"]),
                "plugin_count": len(plugins),
                "repo_root_exists": self.settings.repo_root.exists(),
                "artifacts_dir_exists": self.settings.artifacts_dir.exists(),
                "control_db_exists": self.settings.control_db_path.exists(),
            },
            interfaces=interfaces,
            plugins=plugins,
        )
