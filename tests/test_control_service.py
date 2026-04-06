from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ai_factory.core.control.models import LiveInstanceSnapshot
from ai_factory.core.control.service import FactoryControlService
from ai_factory.core.instances.models import (
    EnvironmentSpec,
    InstanceManifest,
    MetricPoint,
    ProgressSnapshot,
)
from ai_factory.core.orchestration.models import (
    CircuitState,
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
    RetryPolicy,
    TaskAttempt,
    TaskDependency,
    TaskInputEnvelope,
    TaskOutputEnvelope,
)
from ai_factory.core.platform.settings import PlatformSettings
from ai_factory.core.plugins.base import PluginDescriptor


class _FakeStore:
    def __init__(self, root: Path):
        self.root = root
        self._config_snapshot = {
            "instance": {"type": "train"},
            "subsystem": {"config_ref": "training/configs/profiles/baseline.yaml"},
        }
        self._logs = {"stdout": "step 1\nstep 2\n", "stderr": "warn\n"}
        self._events = [{"type": "instance.running", "message": "running"}]

    def load_config_snapshot(self, instance_id: str) -> dict[str, object]:
        return self._config_snapshot

    def read_logs(self, instance_id: str) -> dict[str, str]:
        return self._logs

    def stdout_path(self, instance_id: str) -> Path:
        return self.root / f"{instance_id}.out"

    def stderr_path(self, instance_id: str) -> Path:
        return self.root / f"{instance_id}.err"

    def read_events(self, instance_id: str) -> list[dict[str, object]]:
        return self._events


class _FakeControlPlane:
    def __init__(
        self,
        *,
        run: OrchestrationRun,
        task: OrchestrationTask,
        attempts: list[TaskAttempt],
        dependencies: list[TaskDependency],
        events: list[OrchestrationEvent],
        circuits: dict[str, CircuitState],
        stale_leases: list[dict[str, str]],
    ) -> None:
        self._run = run
        self._task = task
        self._attempts = attempts
        self._dependencies = dependencies
        self._events = events
        self._circuits = circuits
        self._stale_leases = stale_leases

    def get_run(self, run_id: str) -> OrchestrationRun | None:
        return self._run if self._run.id == run_id else None

    def get_run_by_legacy_instance(self, legacy_instance_id: str) -> OrchestrationRun | None:
        return self._run if self._run.legacy_instance_id == legacy_instance_id else None

    def list_tasks(self, run_id: str | None = None) -> list[OrchestrationTask]:
        if run_id is not None and run_id != self._task.run_id:
            return []
        return [self._task]

    def list_attempts(self, task_id: str) -> list[TaskAttempt]:
        return list(self._attempts) if task_id == self._task.id else []

    def list_dependencies(self, task_id: str | None = None) -> list[TaskDependency]:
        if task_id is not None and task_id != self._task.id:
            return []
        return list(self._dependencies)

    def list_events(self, *, run_id: str, limit: int | None = None) -> list[OrchestrationEvent]:
        if run_id != self._run.id:
            return []
        events = list(self._events)
        return events[-limit:] if limit else events

    def list_stale_leases(self, *, stale_before: str) -> list[dict[str, str]]:
        return list(self._stale_leases)

    def get_circuit(self, agent_type: str) -> CircuitState | None:
        return self._circuits.get(agent_type)


class _FakeOrchestration:
    def __init__(self, control_plane: _FakeControlPlane, summary: dict[str, object]) -> None:
        self.control_plane = control_plane
        self._summary = summary

    def list_tasks(self, run_id: str | None = None) -> list[OrchestrationTask]:
        return self.control_plane.list_tasks(run_id)

    def monitoring_summary(self) -> dict[str, object]:
        return self._summary


class _FakeManager:
    def __init__(self, manifest: InstanceManifest, orchestration: _FakeOrchestration) -> None:
        self._manifest = manifest
        self.orchestration = orchestration

    def get_instance(self, instance_id: str) -> InstanceManifest:
        return self._manifest

    def list_tasks(self, target: str | None = None) -> list[dict[str, object]]:
        return [task.model_dump(mode="json") for task in self.orchestration.control_plane.list_tasks(target)]

    def list_orchestration_runs(self) -> list[dict[str, object]]:
        return [
            self.orchestration.control_plane.get_run(
                self._manifest.orchestration_run_id or self._manifest.id
            ).model_dump(mode="json")
        ]

    def get_orchestration_run(self, target: str) -> dict[str, object]:
        run = self.orchestration.control_plane.get_run(target)
        assert run is not None
        return {
            "run": run.model_dump(mode="json"),
            "tasks": self.list_tasks(run.id),
            "events": self.list_orchestration_events(run.id),
        }

    def list_orchestration_events(self, target: str, *, limit: int | None = None) -> list[dict[str, object]]:
        events = self.orchestration.control_plane.list_events(run_id=target, limit=limit)
        return [event.model_dump(mode="json") for event in events]

    def monitoring_summary(self) -> dict[str, object]:
        summary = dict(self.orchestration._summary)
        summary.setdefault("task_status_counts", {})
        return summary

    def get_available_actions(self, instance_id: str) -> list[dict[str, object]]:
        return [{"action": "inspect", "label": "Inspect", "description": "Inspect the instance"}]

    def get_children(self, instance_id: str) -> list[InstanceManifest]:
        return []

    def get_metrics(self, instance_id: str) -> dict[str, object]:
        return {"summary": {"accuracy": 0.91}, "points": [MetricPoint(name="loss", value=1.2)]}


class _FakeLineageRegistry:
    def record_lineage(self, record) -> None:  # noqa: ANN001
        return None

    def get_lineage(self, record_id: str):  # noqa: ANN201, ANN001
        return None

    def list_lineage(self, limit: int = 50):  # noqa: ANN201, ANN001
        return []


def _build_service(tmp_path: Path) -> tuple[FactoryControlService, InstanceManifest, OrchestrationTask]:
    manifest = InstanceManifest(
        id="train-001",
        type="train",
        status="running",
        environment=EnvironmentSpec(kind="local"),
        orchestration_mode="single",
        name="train-001",
        orchestration_run_id="run-001",
        task_summary={"status": "running", "attempts": 2, "current_attempt": 2},
        last_heartbeat_at="2026-04-04T01:00:00+00:00",
        active_agents=["training_orchestration"],
        progress=ProgressSnapshot(stage="training", status_message="running", percent=0.5),
    )
    task = OrchestrationTask(
        id="task-001",
        run_id="run-001",
        legacy_instance_id="train-001",
        task_type="train",
        agent_type="training_orchestration",
        status="retry_waiting",
        resource_class="gpu",
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=5, max_delay_s=30),
        current_attempt=2,
        available_at="2026-04-04T01:05:00+00:00",
        last_error_code="lease_expired",
        last_error_message="Task heartbeat expired.",
        checkpoint_hint="/checkpoints/latest",
        input=TaskInputEnvelope(task_type="train", legacy_instance_id="train-001", resource_class="gpu"),
        output=TaskOutputEnvelope(),
    )
    run = OrchestrationRun(
        id="run-001",
        legacy_instance_id="train-001",
        name="train-001",
        status="running",
        root_run_id="run-001",
    )
    attempts = [
        TaskAttempt(
            id="att-001",
            task_id="task-001",
            sequence=1,
            lease_owner="worker-1",
            finished_at="2026-04-04T00:58:00+00:00",
        ),
        TaskAttempt(
            id="att-002",
            task_id="task-001",
            sequence=2,
            lease_owner="worker-2",
            heartbeat_at="2026-04-04T00:58:30+00:00",
            checkpoint_hint="/checkpoints/latest",
            error_code="lease_expired",
            error_message="Task heartbeat expired.",
        ),
    ]
    dependencies = [TaskDependency(task_id="task-001", depends_on_task_id="task-prep")]
    events = [
        OrchestrationEvent(
            id="evt-001",
            run_id="run-001",
            task_id="task-001",
            event_type="task.running",
            message="Task started.",
            agent_type="training_orchestration",
        ),
        OrchestrationEvent(
            id="evt-002",
            run_id="run-001",
            task_id="task-001",
            event_type="task.failed",
            level="error",
            message="Task failed.",
            agent_type="training_orchestration",
        ),
    ]
    circuits = {
        "training_orchestration": CircuitState(
            agent_type="training_orchestration",
            status="open",
            failure_count=3,
            opened_at="2026-04-04T00:59:00+00:00",
            reopen_after="2026-04-04T01:10:00+00:00",
            last_error="repeated failures",
        )
    }
    control_plane = _FakeControlPlane(
        run=run,
        task=task,
        attempts=attempts,
        dependencies=dependencies,
        events=events,
        circuits=circuits,
        stale_leases=[{"task_id": "task-stale", "attempt_id": "att-stale"}],
    )
    orchestration = _FakeOrchestration(
        control_plane,
        summary={
            "runs": 1,
            "tasks": 1,
            "active_runs": 1,
            "task_status_counts": {"retry_waiting": 1, "running": 0, "completed": 0},
            "ready_tasks": 0,
            "ready_by_resource": {},
        },
    )
    manifest = manifest.model_copy(update={"metrics_summary": {"accuracy": 0.91}})
    manager = _FakeManager(manifest, orchestration)
    store = _FakeStore(tmp_path)
    settings = PlatformSettings(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        control_plane_dir=tmp_path / "artifacts" / "control_plane",
        control_db_path=tmp_path / "artifacts" / "control_plane" / "control_plane.db",
        worker_concurrency=4,
        heartbeat_interval_s=5,
        stale_after_s=60,
        telemetry_sink=str(tmp_path / "artifacts" / "control_plane" / "events.jsonl"),
        local_execution_backend="local",
        cloud_execution_backend="ssh",
        plugin_modules=(),
    )
    plugins = [PluginDescriptor(kind="instance_handler", name="demo", label="Demo", capabilities=["control"])]
    service = FactoryControlService(
        manager=manager,
        store=store,
        settings=settings,
        plugin_registry=SimpleNamespace(list_plugins=lambda: plugins),
        lineage_registry=_FakeLineageRegistry(),
    )
    return service, manifest, task


def test_control_service_detail_and_snapshot_surface_runtime_context(tmp_path: Path) -> None:
    service, manifest, task = _build_service(tmp_path)

    detail = service.get_instance_detail(manifest.id)
    snapshot = service.get_live_instance_snapshot(manifest.id, task_limit=1, event_limit=1)

    assert detail.orchestration_runtime.task_id == task.id
    assert detail.orchestration_runtime.dependency_count == 1
    assert detail.orchestration_runtime.dependency_ready_count == 0
    assert detail.orchestration_runtime.retryable is True
    assert detail.orchestration_runtime.last_event_type == "task.failed"
    assert detail.orchestration_summary["stale_task_count"] == 1
    assert detail.orchestration_summary["open_circuits"] == ["training_orchestration"]
    assert detail.orchestration_summary["circuits"][0]["failure_count"] == 3

    assert isinstance(snapshot, LiveInstanceSnapshot)
    assert snapshot.orchestration_runtime.task_status == "retry_waiting"
    assert snapshot.orchestration_runtime.last_heartbeat_at == "2026-04-04T00:58:30+00:00"
    assert len(snapshot.tasks) == 1
    assert len(snapshot.events) == 1


def test_control_service_foundation_overview_reports_operational_summary(tmp_path: Path) -> None:
    service, _, _ = _build_service(tmp_path)

    foundation = service.describe_foundation()

    assert foundation.summary["interface_count"] == 4
    assert foundation.summary["available_interface_count"] == 3
    assert foundation.summary["planned_interface_count"] == 1
    assert foundation.summary["plugin_count"] == 1
    assert foundation.summary["control_db_exists"] is False
