from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from inference.app.config import AppSettings
from inference.app.main import app
from inference.app.services.instance_service import InstanceService


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _seed_inference_defaults(tmp_path: Path) -> None:
    _write(tmp_path / "inference" / "app" / "defaults" / "model_registry.yaml", "models: []\n")
    _write(tmp_path / "inference" / "app" / "defaults" / "prompt_presets.yaml", "presets: []\n")
    _write(tmp_path / "inference" / "app" / "defaults" / "benchmarks_registry.yaml", "benchmarks: []\n")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_orchestration_routes_expose_runs_tasks_and_summary(tmp_path: Path, monkeypatch) -> None:
    from inference.app.routers import orchestration as orchestration_router

    _seed_inference_defaults(tmp_path)
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "experience:\n"
            "  level: dev\n"
            "subsystem:\n"
            "  command_override: [python, -c, \"print(1)\"]\n"
        ),
    )
    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "app" / "defaults" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "benchmarks_registry.yaml"),
        artifacts_dir=str(tmp_path / "artifacts"),
        cache_dir=str(tmp_path / "artifacts" / "cache"),
        telemetry_path=str(tmp_path / "artifacts" / "telemetry.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
    )
    service = InstanceService(settings)
    detail = service.create_instance(
        orchestration_router.InstanceCreateRequest(
            config_path=str(tmp_path / "configs" / "finetune.yaml"),
            start=False,
        )
    )
    app.dependency_overrides[orchestration_router.get_instance_service] = lambda: service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        runs_response = await client.get("/v1/orchestration/runs")
        assert runs_response.status_code == 200
        assert len(runs_response.json()["runs"]) == 1

        run_id = detail.orchestration_run_id
        assert run_id is not None
        detail_response = await client.get(f"/v1/orchestration/runs/{run_id}")
        tasks_response = await client.get(f"/v1/orchestration/runs/{run_id}/tasks")
        task_id = tasks_response.json()["tasks"][0]["id"]
        task = service.manager.orchestration.control_plane.get_task(task_id)
        assert task is not None
        task.status = "retry_waiting"
        task.available_at = "2000-01-01T00:00:00+00:00"
        service.manager.orchestration.control_plane.upsert_task(task)
        retry_response = await client.post(f"/v1/orchestration/tasks/{task_id}/retry")
        summary_response = await client.get("/v1/orchestration/summary")

    assert detail_response.status_code == 200
    assert tasks_response.status_code == 200
    assert len(tasks_response.json()["tasks"]) == 1
    assert detail_response.json()["summary"]["task_count"] == 1
    assert detail_response.json()["summary"]["ready_tasks"] == 1
    assert retry_response.status_code == 200
    assert retry_response.json()["run"]["id"] == run_id
    assert summary_response.status_code == 200
    assert summary_response.json()["summary"]["runs"] == 1


@pytest.mark.anyio
async def test_orchestration_filters_and_recovery_endpoint(tmp_path: Path) -> None:
    from inference.app.routers import orchestration as orchestration_router

    _seed_inference_defaults(tmp_path)
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "execution:\n"
            "  retry_limit: 2\n"
            "experience:\n"
            "  level: dev\n"
            "subsystem:\n"
            "  command_override: [python, -c, \"print(1)\"]\n"
        ),
    )
    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "app" / "defaults" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "benchmarks_registry.yaml"),
        artifacts_dir=str(tmp_path / "artifacts"),
        cache_dir=str(tmp_path / "artifacts" / "cache"),
        telemetry_path=str(tmp_path / "artifacts" / "telemetry.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
    )
    service = InstanceService(settings)
    app.dependency_overrides[orchestration_router.get_instance_service] = lambda: service

    first = service.create_instance(
        orchestration_router.InstanceCreateRequest(
            config_path=str(tmp_path / "configs" / "finetune.yaml"),
            start=False,
        )
    )
    second = service.create_instance(
        orchestration_router.InstanceCreateRequest(
            config_path=str(tmp_path / "configs" / "finetune.yaml"),
            start=False,
        )
    )
    assert second.id != first.id
    completed = service.create_instance(
        orchestration_router.InstanceCreateRequest(
            config_path=str(tmp_path / "configs" / "finetune.yaml"),
            start=False,
        )
    )
    completed_run = service.manager.orchestration.control_plane.get_run_by_legacy_instance(completed.id)
    assert completed_run is not None
    completed_run.status = "completed"
    service.manager.orchestration.control_plane.upsert_run(completed_run)

    run = service.manager.orchestration.control_plane.get_run_by_legacy_instance(first.id)
    assert run is not None
    service.manager.orchestration.create_task(
        run_id=run.id,
        task_type="report",
        input_payload={"kind": "audit"},
    )

    _run, task, attempt = service.manager.orchestration.begin_attempt(legacy_instance_id=first.id)
    service.manager.orchestration.control_plane.write_lease(
        task_id=task.id,
        attempt_id=attempt.id,
        lease_owner=attempt.lease_owner or "local-runner",
        acquired_at=attempt.started_at,
        heartbeat_at=attempt.heartbeat_at,
        expires_at="2000-01-01T00:00:00+00:00",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        completed_runs = await client.get("/v1/orchestration/runs?status=completed")
        limited_runs = await client.get("/v1/orchestration/runs?limit=1")
        report_tasks = await client.get(f"/v1/orchestration/runs/{run.id}/tasks?task_type=report")
        recovery = await client.post("/v1/orchestration/recover-stalled")
        recovered_events = await client.get(
            f"/v1/orchestration/runs/{run.id}/events?event_type=task.recovered&level=warning&limit=1"
        )
        refreshed_run = await client.get(f"/v1/orchestration/runs/{run.id}")

    assert completed_runs.status_code == 200
    assert [item["id"] for item in completed_runs.json()["runs"]] == [completed_run.id]
    assert limited_runs.status_code == 200
    assert len(limited_runs.json()["runs"]) == 1
    assert report_tasks.status_code == 200
    assert len(report_tasks.json()["tasks"]) == 1
    assert report_tasks.json()["tasks"][0]["task_type"] == "report"
    assert recovery.status_code == 200
    recovery_payload = recovery.json()
    assert recovery_payload["recovered_count"] == 1
    assert recovery_payload["recovered_task_ids"] == [task.id]
    assert recovered_events.status_code == 200
    assert [event["event_type"] for event in recovered_events.json()["events"]] == ["task.recovered"]
    assert refreshed_run.status_code == 200
    assert refreshed_run.json()["summary"]["retry_waiting_tasks"] == 1


@pytest.mark.anyio
async def test_orchestration_filters_reject_invalid_values(tmp_path: Path) -> None:
    from inference.app.routers import orchestration as orchestration_router

    _seed_inference_defaults(tmp_path)
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "experience:\n"
            "  level: dev\n"
            "subsystem:\n"
            "  command_override: [python, -c, \"print(1)\"]\n"
        ),
    )
    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "app" / "defaults" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "benchmarks_registry.yaml"),
        artifacts_dir=str(tmp_path / "artifacts"),
        cache_dir=str(tmp_path / "artifacts" / "cache"),
        telemetry_path=str(tmp_path / "artifacts" / "telemetry.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
    )
    service = InstanceService(settings)
    app.dependency_overrides[orchestration_router.get_instance_service] = lambda: service

    created = service.create_instance(
        orchestration_router.InstanceCreateRequest(
            config_path=str(tmp_path / "configs" / "finetune.yaml"),
            start=False,
        )
    )
    run_id = created.orchestration_run_id
    assert run_id is not None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        invalid_run_status = await client.get("/v1/orchestration/runs?status=bogus")
        invalid_task_type = await client.get(f"/v1/orchestration/runs/{run_id}/tasks?task_type=bogus")
        invalid_event_level = await client.get(f"/v1/orchestration/runs/{run_id}/events?level=verbose")

    assert invalid_run_status.status_code == 422
    assert invalid_task_type.status_code == 422
    assert invalid_event_level.status_code == 422


@pytest.mark.anyio
async def test_orchestration_create_route_uses_injected_service_and_preserves_not_found(tmp_path: Path) -> None:
    from inference.app.routers import orchestration as orchestration_router

    _seed_inference_defaults(tmp_path)
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "experience:\n"
            "  level: dev\n"
            "subsystem:\n"
            "  command_override: [python, -c, \"print(1)\"]\n"
        ),
    )
    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "app" / "defaults" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "benchmarks_registry.yaml"),
        artifacts_dir=str(tmp_path / "artifacts"),
        cache_dir=str(tmp_path / "artifacts" / "cache"),
        telemetry_path=str(tmp_path / "artifacts" / "telemetry.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
    )
    service = InstanceService(settings)
    app.dependency_overrides[orchestration_router.get_instance_service] = lambda: service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post(
            "/v1/orchestration/runs",
            json={
                "config_path": str(tmp_path / "configs" / "finetune.yaml"),
                "start": False,
            },
        )
        missing_response = await client.get("/v1/orchestration/runs/does-not-exist")

    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["run"]["id"] == payload["tasks"][0]["run_id"]
    assert missing_response.status_code == 404
    assert "Unknown orchestration run" in missing_response.json()["detail"]


@pytest.mark.anyio
async def test_instance_routes_support_control_center_creation_and_inference(tmp_path: Path, monkeypatch) -> None:
    from inference.app.routers import instances as instances_router

    _seed_inference_defaults(tmp_path)
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "experience:\n"
            "  level: dev\n"
            "subsystem:\n"
            "  command_override: [python, -c, \"print(1)\"]\n"
        ),
    )
    _write(
        tmp_path / "configs" / "inference.yaml",
        (
            "instance:\n"
            "  type: inference\n"
            "  environment: local\n"
            "execution:\n"
            "  env:\n"
            f"    MODEL_REGISTRY_PATH: {tmp_path / 'inference' / 'app' / 'defaults' / 'model_registry.yaml'}\n"
            "subsystem:\n"
            "  model_variant: finetuned\n"
        ),
    )
    _write(
        tmp_path / "inference" / "app" / "defaults" / "model_registry.yaml",
        "models:\n  - name: base\n    base_model: Qwen/Qwen2.5-Math-1.5B-Instruct\n",
    )

    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "app" / "defaults" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "benchmarks_registry.yaml"),
        artifacts_dir=str(tmp_path / "artifacts"),
        cache_dir=str(tmp_path / "artifacts" / "cache"),
        telemetry_path=str(tmp_path / "artifacts" / "telemetry.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
    )
    service = InstanceService(settings)
    app.dependency_overrides[instances_router.get_instance_service] = lambda: service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_response = await client.post(
            "/v1/instances",
            json={
                "config_path": str(tmp_path / "configs" / "finetune.yaml"),
                "start": False,
                "name": "scratch-branch",
                "user_level": "dev",
                "lifecycle": {
                    "origin": "from_scratch",
                    "learning_mode": "supervised",
                    "architecture": {
                        "family": "transformer",
                        "hidden_size": 768,
                        "num_layers": 12,
                    },
                    "deployment_targets": ["ollama"],
                },
            },
        )

        assert create_response.status_code == 200
        created = create_response.json()
        assert created["name"] == "scratch-branch"
        assert created["lifecycle"]["origin"] == "from_scratch"
        assert created["lifecycle"]["architecture"]["family"] == "transformer"

        source_manifest = service.store.load(created["id"])
        source_manifest.artifact_refs["published"] = {"final_adapter": "artifacts/models/demo"}
        service.store.save(source_manifest)

        inference_response = await client.post(
            f"/v1/instances/{created['id']}/inference",
            json={"config_path": str(tmp_path / "configs" / "inference.yaml"), "start": False},
        )

    assert inference_response.status_code == 200
    child = inference_response.json()
    assert child["type"] == "inference"
    assert child["parent_instance_id"] == created["id"]
    assert child["lifecycle"]["stage"] == "infer"


@pytest.mark.anyio
async def test_instance_live_stream_and_foundation_routes(tmp_path: Path, monkeypatch) -> None:
    from inference.app.routers import instances as instances_router

    _seed_inference_defaults(tmp_path)
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "experience:\n"
            "  level: dev\n"
            "subsystem:\n"
            "  command_override: [python, -c, \"print(1)\"]\n"
        ),
    )
    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "app" / "defaults" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "benchmarks_registry.yaml"),
        artifacts_dir=str(tmp_path / "artifacts"),
        cache_dir=str(tmp_path / "artifacts" / "cache"),
        telemetry_path=str(tmp_path / "artifacts" / "telemetry.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
    )
    service = InstanceService(settings)
    app.dependency_overrides[instances_router.get_instance_service] = lambda: service

    created = service.create_instance(
        instances_router.InstanceCreateRequest(
            config_path=str(tmp_path / "configs" / "finetune.yaml"),
            start=False,
        )
    )
    manifest = service.store.load(created.id)
    manifest.status = "completed"
    service.store.save(manifest)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        live_response = await client.get(f"/v1/instances/{created.id}/live")
        foundation_response = await client.get("/v1/foundation")

    assert live_response.status_code == 200
    assert live_response.json()["instance"]["id"] == created.id
    assert foundation_response.status_code == 200
    interfaces = {item["id"] for item in foundation_response.json()["interfaces"]}
    assert {"cli", "tui", "web_api", "desktop"} <= interfaces


@pytest.mark.anyio
async def test_instance_action_route_supports_generic_follow_up_actions(tmp_path: Path, monkeypatch) -> None:
    from inference.app.routers import instances as instances_router

    _seed_inference_defaults(tmp_path)
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "experience:\n"
            "  level: dev\n"
            "subsystem:\n"
            "  command_override: [python, -c, \"print(1)\"]\n"
        ),
    )

    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "app" / "defaults" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "inference" / "app" / "defaults" / "benchmarks_registry.yaml"),
        artifacts_dir=str(tmp_path / "artifacts"),
        cache_dir=str(tmp_path / "artifacts" / "cache"),
        telemetry_path=str(tmp_path / "artifacts" / "telemetry.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
    )
    service = InstanceService(settings)
    app.dependency_overrides[instances_router.get_instance_service] = lambda: service

    source = service.create_instance(
        instances_router.InstanceCreateRequest(
            config_path=str(tmp_path / "configs" / "finetune.yaml"),
            start=False,
        )
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        action_response = await client.post(
            f"/v1/instances/{source.id}/actions",
            json={
                "action": "finetune",
                "config_path": str(tmp_path / "configs" / "finetune.yaml"),
                "start": False,
            },
        )

    assert action_response.status_code == 200
    child = action_response.json()
    assert child["type"] == "finetune"
    assert child["parent_instance_id"] == source.id
