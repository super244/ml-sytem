from __future__ import annotations

from pathlib import Path

from ai_factory.core.instances.manager import InstanceManager
from ai_factory.core.instances.models import EnvironmentSpec, InstanceManifest
from ai_factory.core.instances.store import FileInstanceStore


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_control_plane_tracks_attempts_retries_and_projection(tmp_path: Path):
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


def test_control_plane_dependency_resolution_and_circuit_breaking(tmp_path: Path):
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
