from __future__ import annotations

from pathlib import Path

from ai_factory.core.instances.manager import InstanceManager
from ai_factory.core.instances.models import EnvironmentSpec, InstanceManifest
from ai_factory.core.instances.store import FileInstanceStore


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_control_plane_tracks_attempts_retries_and_projection(tmp_path: Path) -> None:
    profile_dir = tmp_path / "training" / "configs" / "profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "baseline_qlora.yaml").write_text("run_name: demo-run\ntraining:\n  artifacts_dir: artifacts\n")
    config_path = tmp_path / "configs" / "finetune.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: finetune",
                "  name: demo-finetune",
                "execution:",
                "  retry_limit: 2",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        )
    )

    store = FileInstanceStore(tmp_path)
    manager = InstanceManager(store)

    created = manager.create_instance(str(config_path), start=False)
    task = manager.orchestration.list_tasks(created.id)[0]
    assert task.status == "queued"

    run, active_task, attempt = manager.orchestration.begin_attempt(
        legacy_instance_id=created.id,
        stdout_path=str(store.stdout_path(created.id)),
        stderr_path=str(store.stderr_path(created.id)),
    )
    manager.orchestration.finalize_attempt(
        legacy_instance_id=created.id,
        attempt_id=attempt.id,
        exit_code=1,
        error_code="boom",
        error_message="first failure",
    )
    projected = manager.get_instance(created.id)
    assert projected.orchestration_run_id == run.id
    assert projected.task_summary["status"] == "retry_waiting"
    assert projected.task_summary["attempts"] == 1
    assert manager.orchestration.latest_checkpoint(created.id) is None
    assert active_task.run_id == run.id


def test_control_plane_dependency_resolution_and_circuit_breaking(tmp_path: Path) -> None:
    store = FileInstanceStore(tmp_path)
    manager = InstanceManager(store)
    manifest = InstanceManifest(
        id="prepare-001",
        type="prepare",
        name="prepare",
        environment=EnvironmentSpec(kind="local"),
    )
    snapshot = {"instance": {"type": "prepare"}, "subsystem": {"config_ref": "data/configs/processing.yaml"}}
    store.create(manifest, snapshot)
    run, primary = manager.orchestration.ensure_run_for_instance(manifest, snapshot)

    child = manager.orchestration.create_task(
        run_id=run.id,
        task_type="report",
        input_payload={"source": "per_example.jsonl"},
        dependencies=[primary.id],
    )
    ready_ids = [task.id for task in manager.orchestration.ready_tasks(run.id)]
    assert child.id not in ready_ids

    attempt = manager.orchestration.begin_attempt(legacy_instance_id=manifest.id)[2]
    manager.orchestration.finalize_attempt(
        legacy_instance_id=manifest.id,
        attempt_id=attempt.id,
        exit_code=0,
        summary={"ok": True},
    )
    ready_ids = [task.id for task in manager.orchestration.ready_tasks(run.id)]
    assert child.id in ready_ids

    manager.orchestration.record_agent_failure("deployment", "fail-1")
    manager.orchestration.record_agent_failure("deployment", "fail-2")
    circuit = manager.orchestration.record_agent_failure("deployment", "fail-3")
    assert circuit.status == "open"
    assert manager.orchestration.is_circuit_open("deployment") is True


def test_control_plane_seeds_sub_agent_workloads_with_dependencies(tmp_path: Path) -> None:
    store = FileInstanceStore(tmp_path)
    manager = InstanceManager(store)
    manifest = InstanceManifest(
        id="train-001",
        type="train",
        name="train",
        environment=EnvironmentSpec(kind="local"),
    )
    snapshot = {
        "instance": {"type": "train"},
        "subsystem": {"config_ref": "training/configs/profiles/baseline_qlora.yaml"},
        "sub_agents": {
            "enabled": True,
            "max_parallelism": 2,
            "workloads": ["preprocess", "evaluation", "metrics", "publish"],
        },
    }
    store.create(manifest, snapshot)
    run, primary = manager.orchestration.ensure_run_for_instance(manifest, snapshot)

    tasks = manager.orchestration.list_tasks(run.id)
    task_types = {task.task_type for task in tasks}
    assert {"train", "prepare", "evaluate", "monitor", "deploy"} <= task_types

    dependencies_by_task = {task.id: manager.orchestration.control_plane.list_dependencies(task.id) for task in tasks}
    preprocess_task = next(task for task in tasks if task.task_type == "prepare")
    evaluate_task = next(task for task in tasks if task.task_type == "evaluate")
    deploy_task = next(task for task in tasks if task.task_type == "deploy")

    primary_dependencies = {dep.depends_on_task_id for dep in dependencies_by_task[primary.id]}
    assert preprocess_task.id in primary_dependencies

    evaluate_dependencies = {dep.depends_on_task_id for dep in dependencies_by_task[evaluate_task.id]}
    assert primary.id in evaluate_dependencies

    deploy_dependencies = {dep.depends_on_task_id for dep in dependencies_by_task[deploy_task.id]}
    assert evaluate_task.id in deploy_dependencies


def test_sqlite_control_plane_query_helpers_surface_active_state_and_filters(tmp_path: Path) -> None:
    from ai_factory.core.orchestration.models import (
        OrchestrationEvent,
        OrchestrationRun,
        OrchestrationTask,
        RetryPolicy,
        TaskAttempt,
        TaskInputEnvelope,
        TaskOutputEnvelope,
    )
    from ai_factory.core.orchestration.sqlite import SqliteControlPlane

    control = SqliteControlPlane(tmp_path / "control_plane.db")
    run = OrchestrationRun(
        id="run-001",
        legacy_instance_id="legacy-001",
        name="legacy-001",
        idempotency_key="idem-123",
    )
    control.upsert_run(run)
    primary = OrchestrationTask(
        id="task-primary",
        run_id=run.id,
        legacy_instance_id=run.legacy_instance_id,
        task_type="train",
        agent_type="training_orchestration",
        status="completed",
        resource_class="gpu",
        retry_policy=RetryPolicy(max_attempts=2),
        input=TaskInputEnvelope(task_type="train", legacy_instance_id=run.legacy_instance_id),
        output=TaskOutputEnvelope(),
    )
    active = OrchestrationTask(
        id="task-active",
        run_id=run.id,
        legacy_instance_id=run.legacy_instance_id,
        task_type="evaluate",
        agent_type="evaluation_benchmarking",
        status="running",
        resource_class="cpu",
        retry_policy=RetryPolicy(max_attempts=2),
        input=TaskInputEnvelope(task_type="evaluate", legacy_instance_id=run.legacy_instance_id),
        output=TaskOutputEnvelope(),
    )
    control.upsert_task(primary)
    control.upsert_task(active)

    attempt = TaskAttempt(id="attempt-001", task_id=active.id, sequence=1, lease_owner="runner")
    control.upsert_attempt(attempt)
    control.write_lease(
        task_id=active.id,
        attempt_id=attempt.id,
        lease_owner="runner",
        acquired_at="2000-01-01T00:00:00+00:00",
        heartbeat_at="2000-01-01T00:00:00+00:00",
        expires_at="2000-01-01T00:05:00+00:00",
    )

    control.append_event(
        OrchestrationEvent(
            id="evt-001",
            run_id=run.id,
            task_id=primary.id,
            event_type="task.completed",
            message="Primary task completed.",
        )
    )
    control.append_event(
        OrchestrationEvent(
            id="evt-002",
            run_id=run.id,
            task_id=active.id,
            attempt_id=attempt.id,
            event_type="task.running",
            message="Active task running.",
        )
    )
    control.append_event(
        OrchestrationEvent(
            id="evt-003",
            run_id=run.id,
            task_id=active.id,
            attempt_id=attempt.id,
            event_type="task.heartbeat",
            message="Active task heartbeat.",
        )
    )

    selected = control.get_task_by_legacy_instance(run.legacy_instance_id)
    events = control.list_events(run_id=run.id, task_id=active.id, limit=2)
    lease = control.get_lease(active.id)
    stale_leases = control.list_leases(stale_before="2000-01-01T00:10:00+00:00")

    assert control.get_run_by_idempotency_key("idem-123") is not None
    assert selected is not None
    assert selected.id == active.id
    assert [event.id for event in events] == ["evt-002", "evt-003"]
    assert lease is not None and lease.attempt_id == attempt.id
    assert len(stale_leases) == 1
    assert stale_leases[0].task_id == active.id


def test_sqlite_control_plane_filters_runs_tasks_attempts_and_events(tmp_path: Path) -> None:
    from ai_factory.core.orchestration.models import (
        OrchestrationEvent,
        OrchestrationRun,
        OrchestrationTask,
        RetryPolicy,
        TaskAttempt,
        TaskInputEnvelope,
        TaskOutputEnvelope,
    )
    from ai_factory.core.orchestration.sqlite import SqliteControlPlane

    control = SqliteControlPlane(tmp_path / "control_plane.db")

    queued_run = OrchestrationRun(
        id="run-queued",
        legacy_instance_id="legacy-queued",
        name="legacy-queued",
        status="queued",
    )
    completed_run = OrchestrationRun(
        id="run-completed",
        legacy_instance_id="legacy-completed",
        name="legacy-completed",
        status="completed",
    )
    control.upsert_run(queued_run)
    control.upsert_run(completed_run)

    queued_task = OrchestrationTask(
        id="task-queued",
        run_id=queued_run.id,
        legacy_instance_id=queued_run.legacy_instance_id,
        task_type="train",
        agent_type="training_orchestration",
        status="ready",
        resource_class="gpu",
        retry_policy=RetryPolicy(max_attempts=2),
        current_attempt=1,
        input=TaskInputEnvelope(task_type="train", legacy_instance_id=queued_run.legacy_instance_id),
        output=TaskOutputEnvelope(),
    )
    completed_task = OrchestrationTask(
        id="task-completed",
        run_id=completed_run.id,
        legacy_instance_id=completed_run.legacy_instance_id,
        task_type="report",
        agent_type="monitoring_telemetry",
        status="completed",
        resource_class="io",
        retry_policy=RetryPolicy(max_attempts=2),
        current_attempt=1,
        input=TaskInputEnvelope(task_type="report", legacy_instance_id=completed_run.legacy_instance_id),
        output=TaskOutputEnvelope(),
    )
    control.upsert_task(queued_task)
    control.upsert_task(completed_task)

    queued_attempt = TaskAttempt(id="attempt-queued", task_id=queued_task.id, sequence=1, status="running", lease_owner="runner")
    completed_attempt = TaskAttempt(
        id="attempt-completed",
        task_id=completed_task.id,
        sequence=1,
        status="completed",
        lease_owner="runner",
    )
    control.upsert_attempt(queued_attempt)
    control.upsert_attempt(completed_attempt)

    control.append_event(
        OrchestrationEvent(
            id="evt-queued",
            run_id=queued_run.id,
            task_id=queued_task.id,
            attempt_id=queued_attempt.id,
            event_type="task.running",
            message="Queued task running.",
        )
    )
    control.append_event(
        OrchestrationEvent(
            id="evt-completed",
            run_id=completed_run.id,
            task_id=completed_task.id,
            attempt_id=completed_attempt.id,
            event_type="task.completed",
            level="warning",
            message="Completed task emitted warning.",
        )
    )

    assert [run.id for run in control.list_runs(status="completed")] == [completed_run.id]
    assert [run.id for run in control.list_runs(legacy_instance_id=queued_run.legacy_instance_id)] == [queued_run.id]
    assert [task.id for task in control.list_tasks(run_id=queued_run.id, status="ready", agent_type="training_orchestration")] == [
        queued_task.id
    ]
    assert [attempt.id for attempt in control.list_attempts(queued_task.id, status="running")] == [queued_attempt.id]
    assert [event.id for event in control.list_events(run_id=queued_run.id, task_id=queued_task.id, attempt_id=queued_attempt.id, event_type="task.running")] == [
        "evt-queued"
    ]
    assert [event.id for event in control.list_events(run_id=completed_run.id, limit=1)] == ["evt-completed"]
