from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ai_factory.core.instances.models import EnvironmentSpec, InstanceManifest
from ai_factory.core.orchestration.models import CircuitState, OrchestrationEvent, OrchestrationRun, RetryPolicy
from ai_factory.core.orchestration.service import OrchestrationService
from ai_factory.core.orchestration.sqlite import SqliteControlPlane
from ai_factory.core.platform.settings import PlatformSettings


def _make_service(tmp_path: Path, *, worker_concurrency: int = 4, stale_after_s: int = 60) -> OrchestrationService:
    artifacts_dir = tmp_path / "artifacts"
    control_plane_dir = artifacts_dir / "control_plane"
    settings = PlatformSettings(
        repo_root=tmp_path,
        artifacts_dir=artifacts_dir,
        control_plane_dir=control_plane_dir,
        control_db_path=control_plane_dir / "control_plane.db",
        worker_concurrency=worker_concurrency,
        heartbeat_interval_s=5,
        stale_after_s=stale_after_s,
        telemetry_sink=str(control_plane_dir / "events.jsonl"),
        local_execution_backend="local",
        cloud_execution_backend="ssh",
        plugin_modules=(),
    )
    return OrchestrationService(control_plane=SqliteControlPlane(settings.control_db_path), settings=settings)


def _upsert_run(service: OrchestrationService, run_id: str, *, legacy_instance_id: str | None = None) -> OrchestrationRun:
    run = OrchestrationRun(id=run_id, legacy_instance_id=legacy_instance_id, name=run_id)
    service.control_plane.upsert_run(run)
    return run


def test_generic_task_lifecycle_supports_task_ids_and_pragmatic_run_aggregation(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    run = _upsert_run(service, "run-generic", legacy_instance_id="legacy-001")

    primary = service.create_task(run_id=run.id, task_type="train", legacy_instance_id="legacy-001")
    assert primary.legacy_instance_id == "legacy-001"
    generic = service.create_task(run_id=run.id, task_type="monitor", input_payload={"workload": "metrics"})

    _, _, primary_attempt = service.begin_attempt(legacy_instance_id="legacy-001")
    run_after_primary, _, _ = service.finalize_attempt(
        legacy_instance_id="legacy-001",
        attempt_id=primary_attempt.id,
        exit_code=0,
        checkpoint_hint="artifacts/checkpoints/step-9",
        summary={"ok": True},
    )

    assert run_after_primary.status == "completed"
    assert service.latest_checkpoint("legacy-001") == "artifacts/checkpoints/step-9"

    running_run, _, generic_attempt = service.begin_task_attempt(task_id=generic.id)
    assert running_run.status == "running"

    retrying_run, retrying_task, _ = service.finalize_task_attempt(
        task_id=generic.id,
        attempt_id=generic_attempt.id,
        exit_code=1,
        error_code="boom",
        error_message="synthetic failure",
    )
    assert retrying_task.status == "retry_waiting"
    assert retrying_run.status == "running"

    retried = service.retry_task(generic.id)
    assert retried.status == "ready"

    _, _, generic_attempt_2 = service.begin_task_attempt(task_id=generic.id)
    completed_run, _, _ = service.finalize_task_attempt(
        task_id=generic.id,
        attempt_id=generic_attempt_2.id,
        exit_code=0,
        summary={"recovered": True},
    )

    described = service.describe_task(generic.id)
    assert completed_run.status == "completed"
    assert len(described["attempts"]) == 2
    assert described["readiness"]["ready"] is False


def test_cancel_run_cleans_up_running_attempts_and_leases(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    run = _upsert_run(service, "run-cancel", legacy_instance_id="legacy-cancel")
    task = service.create_task(run_id=run.id, task_type="train", legacy_instance_id="legacy-cancel")

    _, _, attempt = service.begin_attempt(legacy_instance_id="legacy-cancel")
    assert service.control_plane.get_lease(task.id) is not None

    cancelled_run = service.cancel_run(run.id)
    cancelled_attempt = service.control_plane.get_attempt(attempt.id)
    cancelled_task = service.control_plane.get_task(task.id)

    assert cancelled_run.status == "cancelled"
    assert cancelled_attempt is not None and cancelled_attempt.status == "cancelled"
    assert cancelled_task is not None and cancelled_task.status == "cancelled"
    assert service.control_plane.get_lease(task.id) is None


def test_dispatch_ready_tasks_respects_global_capacity_agent_concurrency_and_circuits(tmp_path: Path) -> None:
    service = _make_service(tmp_path, worker_concurrency=2)
    run = _upsert_run(service, "run-dispatch")

    train_task = service.create_task(run_id=run.id, task_type="train", input_payload={"slot": 1})
    finetune_task = service.create_task(run_id=run.id, task_type="finetune", input_payload={"slot": 2})
    report_task = service.create_task(run_id=run.id, task_type="report", input_payload={"slot": 3})

    first_pass = asyncio.run(service.dispatch_ready_tasks())
    first_ids = {task.id for task in first_pass}

    assert train_task.id in first_ids
    assert report_task.id in first_ids
    assert finetune_task.id not in first_ids

    service.begin_task_attempt(task_id=train_task.id)
    second_pass = asyncio.run(service.dispatch_ready_tasks())
    assert [task.id for task in second_pass] == [report_task.id]

    service.record_agent_failure("evaluation_benchmarking", "fail-1")
    service.record_agent_failure("evaluation_benchmarking", "fail-2")
    service.record_agent_failure("evaluation_benchmarking", "fail-3")
    third_pass = asyncio.run(service.dispatch_ready_tasks())
    assert third_pass == []


def test_beginning_heartbeat_and_finalize_rejects_stale_leases(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    run = _upsert_run(service, "run-stale-lease", legacy_instance_id="legacy-stale-lease")
    task = service.create_task(run_id=run.id, task_type="train", legacy_instance_id="legacy-stale-lease")

    _, _, attempt = service.begin_attempt(legacy_instance_id="legacy-stale-lease")
    service.control_plane.drop_lease(task.id)

    with pytest.raises(ValueError, match="active lease"):
        service.heartbeat(attempt.id)

    with pytest.raises(ValueError, match="active lease"):
        service.finalize_task_attempt(task_id=task.id, attempt_id=attempt.id, exit_code=0)


def test_list_events_limit_returns_most_recent_events_in_chronological_order(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    run = _upsert_run(service, "run-events")

    service.control_plane.append_event(
        OrchestrationEvent(
            id="evt-001",
            run_id=run.id,
            event_type="run.created",
            message="Created.",
            created_at="2000-01-01T00:00:01+00:00",
        )
    )
    service.control_plane.append_event(
        OrchestrationEvent(
            id="evt-002",
            run_id=run.id,
            event_type="task.running",
            message="Running.",
            created_at="2000-01-01T00:00:02+00:00",
        )
    )
    service.control_plane.append_event(
        OrchestrationEvent(
            id="evt-003",
            run_id=run.id,
            event_type="task.completed",
            message="Completed.",
            created_at="2000-01-01T00:00:03+00:00",
        )
    )

    events = service.control_plane.list_events(run_id=run.id, limit=2)

    assert [event.id for event in events] == ["evt-002", "evt-003"]


def test_compute_retry_delay_is_clamped_to_the_policy_maximum(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    policy = RetryPolicy(max_attempts=3, base_delay_s=10, max_delay_s=5, multiplier=3.0, jitter_s=10.0)

    assert service.compute_retry_delay(policy, current_attempt=4) == 5


def test_recover_stalled_tasks_marks_retry_waiting_and_updates_summary(tmp_path: Path) -> None:
    service = _make_service(tmp_path, stale_after_s=5)
    run = _upsert_run(service, "run-stale", legacy_instance_id="legacy-stale")
    task = service.create_task(run_id=run.id, task_type="train", legacy_instance_id="legacy-stale")

    _, _, attempt = service.begin_attempt(legacy_instance_id="legacy-stale")
    lease = service.control_plane.get_lease(task.id)
    assert lease is not None
    service.control_plane.write_lease(
        task_id=task.id,
        attempt_id=attempt.id,
        lease_owner=attempt.lease_owner or "local-runner",
        acquired_at=attempt.started_at,
        heartbeat_at=attempt.heartbeat_at,
        expires_at=lease.acquired_at,
    )

    recovered = service.recover_stalled_tasks()
    recovered_task = service.control_plane.get_task(task.id)
    recovered_attempt = service.control_plane.get_attempt(attempt.id)
    summary = service.monitoring_summary()
    run_summary = service.summarize_run(run.id)

    assert len(recovered) == 1
    assert recovered_task is not None and recovered_task.status == "retry_waiting"
    assert recovered_attempt is not None and recovered_attempt.status == "failed"
    assert summary["retry_waiting_tasks"] == 1
    assert run_summary["retry_waiting_tasks"] == 1


def test_sub_agent_workflow_dependencies_remain_correct_when_workloads_are_out_of_order(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    manifest = InstanceManifest(
        id="train-out-of-order",
        type="train",
        name="train-out-of-order",
        environment=EnvironmentSpec(kind="local"),
        config_path="configs/train.yaml",
    )
    snapshot = {
        "instance": {"type": "train"},
        "subsystem": {"config_ref": "training/configs/profiles/baseline_qlora.yaml"},
        "sub_agents": {
            "enabled": True,
            "max_parallelism": 2,
            "workloads": ["publish", "evaluation", "preprocess"],
        },
    }

    run, primary = service.ensure_run_for_instance(manifest, snapshot)
    tasks = service.list_tasks(run.id)
    tasks_by_type = {task.task_type: task for task in tasks}

    preprocess_dependencies = {
        dep.depends_on_task_id for dep in service.control_plane.list_dependencies(tasks_by_type["prepare"].id)
    }
    evaluation_dependencies = {
        dep.depends_on_task_id for dep in service.control_plane.list_dependencies(tasks_by_type["evaluate"].id)
    }
    deploy_dependencies = {
        dep.depends_on_task_id for dep in service.control_plane.list_dependencies(tasks_by_type["deploy"].id)
    }

    assert primary.id in evaluation_dependencies
    assert preprocess_dependencies == set()
    assert tasks_by_type["evaluate"].id in deploy_dependencies


def test_dispatch_ready_legacy_tasks_skips_generic_tasks(tmp_path: Path) -> None:
    service = _make_service(tmp_path, worker_concurrency=2)
    run = _upsert_run(service, "run-legacy", legacy_instance_id="legacy-001")

    legacy_task = service.create_task(run_id=run.id, task_type="train", legacy_instance_id="legacy-001")
    generic_task = service.create_task(run_id=run.id, task_type="report", input_payload={"workload": "metrics"})

    dispatched = asyncio.run(service.dispatch_ready_legacy_tasks())

    assert [task.id for task in dispatched] == [legacy_task.id]
    assert generic_task.id not in {task.id for task in dispatched}


def test_watch_run_returns_terminal_snapshot_and_missing_run(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    run = _upsert_run(service, "run-watch", legacy_instance_id="legacy-watch")

    service.create_task(run_id=run.id, task_type="train", legacy_instance_id="legacy-watch")
    _, _, attempt = service.begin_attempt(legacy_instance_id="legacy-watch")
    completed_run, _, _ = service.finalize_attempt(
        legacy_instance_id="legacy-watch",
        attempt_id=attempt.id,
        exit_code=0,
        summary={"ok": True},
    )

    snapshot = asyncio.run(service.watch_run("legacy-watch", poll_interval_s=0.01, timeout_s=0.1))
    missing = asyncio.run(service.watch_run("missing-run", poll_interval_s=0.01, timeout_s=0.01))

    assert snapshot["run"]["id"] == completed_run.id
    assert snapshot["run"]["status"] == "completed"
    assert [event["event_type"] for event in snapshot["events"]][-1] == "task.completed"
    assert missing == {"run": None, "events": []}


def test_open_circuits_transition_to_half_open_after_cooldown(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    service.control_plane.upsert_circuit(
        CircuitState(
            agent_type="training_orchestration",
            status="open",
            failure_count=3,
            opened_at="2000-01-01T00:00:00+00:00",
            reopen_after="2000-01-01T00:05:00+00:00",
            last_error="repeated failures",
        )
    )

    assert service.is_circuit_open("training_orchestration") is False

    circuit = service.control_plane.get_circuit("training_orchestration")
    summary = service.monitoring_summary()

    assert circuit is not None
    assert circuit.status == "half_open"
    assert summary["open_circuits"] == []
    assert summary["circuits"]["training_orchestration"]["status"] == "half_open"
