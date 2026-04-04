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


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_orchestration_routes_expose_runs_tasks_and_summary(tmp_path: Path, monkeypatch) -> None:
    from inference.app.routers import orchestration as orchestration_router

    _write(
        tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml",
        "run_name: baseline\ntraining:\n  artifacts_dir: artifacts\n",
    )
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "subsystem:\n"
            "  config_ref: ../training/configs/profiles/baseline_qlora.yaml\n"
        ),
    )
    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "configs" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "configs" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "evaluation" / "benchmarks" / "registry.yaml"),
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
        retry_response = await client.post(f"/v1/orchestration/tasks/{task_id}/retry")
        summary_response = await client.get("/v1/orchestration/summary")

    assert detail_response.status_code == 200
    assert tasks_response.status_code == 200
    assert len(tasks_response.json()["tasks"]) == 1
    assert retry_response.status_code == 200
    retry_payload = retry_response.json()
    assert retry_payload["run"]["id"] == run_id
    assert retry_payload["tasks"][0]["run_id"] == run_id
    assert summary_response.status_code == 200
    assert summary_response.json()["summary"]["runs"] == 1


@pytest.mark.anyio
async def test_instance_routes_support_control_center_creation_and_inference(tmp_path: Path, monkeypatch) -> None:
    from inference.app.routers import instances as instances_router

    _write(
        tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml",
        "run_name: baseline\ntraining:\n  artifacts_dir: artifacts\n",
    )
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "subsystem:\n"
            "  config_ref: ../training/configs/profiles/baseline_qlora.yaml\n"
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
            f"    MODEL_REGISTRY_PATH: {tmp_path / 'inference' / 'configs' / 'model_registry.yaml'}\n"
            "subsystem:\n"
            "  model_variant: finetuned\n"
        ),
    )
    _write(
        tmp_path / "inference" / "configs" / "model_registry.yaml",
        "models:\n  - name: base\n    base_model: Qwen/Qwen2.5-Math-1.5B-Instruct\n",
    )

    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "configs" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "configs" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "evaluation" / "benchmarks" / "registry.yaml"),
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

    _write(
        tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml",
        "run_name: baseline\ntraining:\n  artifacts_dir: artifacts\n",
    )
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "subsystem:\n"
            "  config_ref: ../training/configs/profiles/baseline_qlora.yaml\n"
        ),
    )
    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "configs" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "configs" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "evaluation" / "benchmarks" / "registry.yaml"),
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
        stream_response = await client.get(f"/v1/instances/{created.id}/stream")

    assert live_response.status_code == 200
    assert live_response.json()["instance"]["id"] == created.id
    assert foundation_response.status_code == 200
    interfaces = {item["id"] for item in foundation_response.json()["interfaces"]}
    assert {"cli", "tui", "web_api", "desktop"} <= interfaces
    assert stream_response.status_code == 200
    assert "event: snapshot" in stream_response.text
    assert created.id in stream_response.text


@pytest.mark.anyio
async def test_instance_action_route_supports_generic_follow_up_actions(tmp_path: Path, monkeypatch) -> None:
    from inference.app.routers import instances as instances_router

    _write(
        tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml",
        "run_name: baseline\ntraining:\n  artifacts_dir: artifacts\n",
    )
    _write(
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "  environment: local\n"
            "subsystem:\n"
            "  config_ref: ../training/configs/profiles/baseline_qlora.yaml\n"
        ),
    )

    settings = AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "configs" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "configs" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "evaluation" / "benchmarks" / "registry.yaml"),
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
