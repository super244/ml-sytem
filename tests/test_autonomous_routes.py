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


def _seed_workspace(tmp_path: Path) -> AppSettings:
    _write(tmp_path / "data" / "catalog.json", '{"summary": {"num_datasets": 3}}')
    _write(tmp_path / "data" / "processed" / "pack_summary.json", '{"packs": [{"id": "core_train_mix"}]}')
    _write(tmp_path / "data" / "processed" / "manifest.json", '{"schema_version": "v2"}')
    _write(
        tmp_path / "evaluation" / "benchmarks" / "registry.yaml",
        (
            "benchmarks:\n"
            "  - id: benchmark_holdout\n"
            "    title: Holdout\n"
            "    path: data.jsonl\n"
            "    description: Test\n"
            "    tags: []\n"
        ),
    )
    _write(
        tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml",
        "run_name: demo\ntraining:\n  artifacts_dir: artifacts\n",
    )
    _write(
        tmp_path / "configs" / "train.yaml",
        (
            "instance:\n"
            "  type: train\n"
            "  environment: local\n"
            "subsystem:\n"
            "  config_ref: ../training/configs/profiles/baseline_qlora.yaml\n"
        ),
    )
    _write(
        tmp_path / "configs" / "eval.yaml",
        (
            "instance:\n"
            "  type: evaluate\n"
            "  environment: local\n"
            "subsystem:\n"
            "  config_ref: ../evaluation/benchmarks/registry.yaml\n"
        ),
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
        tmp_path / "inference" / "configs" / "model_registry.yaml",
        "models:\n  - name: base\n    base_model: Qwen/Qwen2.5-Math-1.5B-Instruct\n",
    )
    _write(tmp_path / "inference" / "configs" / "prompt_presets.yaml", "presets: []\n")
    (tmp_path / "frontend" / "node_modules").mkdir(parents=True, exist_ok=True)

    return AppSettings(
        title="test",
        version="0.0.0",
        repo_root=str(tmp_path),
        cors_origins=["*"],
        model_registry_path=str(tmp_path / "inference" / "configs" / "model_registry.yaml"),
        prompt_library_path=str(tmp_path / "inference" / "configs" / "prompt_presets.yaml"),
        benchmark_registry_path=str(tmp_path / "evaluation" / "benchmarks" / "registry.yaml"),
        artifacts_dir=str(tmp_path / "artifacts"),
        cache_dir=str(tmp_path / "artifacts" / "cache"),
        telemetry_path=str(tmp_path / "artifacts" / "inference" / "telemetry" / "requests.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
        demo_mode=False,
    )


@pytest.mark.anyio
async def test_autonomous_overview_exposes_real_loop_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai_factory.platform.monitoring import hardware
    from inference.app import workspace as workspace_module
    from inference.app.routers import autonomous as autonomous_router

    settings = _seed_workspace(tmp_path)
    instance_service = InstanceService(settings)

    monkeypatch.setattr(workspace_module, "_has_package", lambda _name: True)
    app.dependency_overrides[autonomous_router.get_settings] = lambda: settings
    app.dependency_overrides[autonomous_router.get_instance_service] = lambda: instance_service
    monkeypatch.setattr(hardware, "get_cluster_nodes", lambda: [{"id": "gpu-1", "status": "idle"}])

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        snapshot_response = await client.get("/v1/experiments/autonomous")
        overview_response = await client.get("/v1/experiments/autonomous/overview")

    assert snapshot_response.status_code == 200
    assert overview_response.status_code == 200
    snapshot = snapshot_response.json()
    overview = overview_response.json()
    assert snapshot["ready"] is True
    assert snapshot["summary"]["executable_actions"] >= 1
    assert any(action["kind"] == "launch_training" for action in snapshot["actions"])
    assert overview["capacity"]["idle_nodes"] == 1
    assert any(stage["id"] == "datasets" for stage in overview["stages"])


@pytest.mark.anyio
async def test_autonomous_execute_queues_managed_instances_without_starting_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai_factory.platform.monitoring import hardware
    from inference.app import workspace as workspace_module
    from inference.app.routers import autonomous as autonomous_router

    settings = _seed_workspace(tmp_path)
    instance_service = InstanceService(settings)

    monkeypatch.setattr(workspace_module, "_has_package", lambda _name: True)
    app.dependency_overrides[autonomous_router.get_settings] = lambda: settings
    app.dependency_overrides[autonomous_router.get_instance_service] = lambda: instance_service
    monkeypatch.setattr(hardware, "get_cluster_nodes", lambda: [{"id": "gpu-1", "status": "idle"}])

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/experiments/autonomous/run",
            json={"max_actions": 1, "dry_run": False, "start_instances": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "executed"
    assert payload["summary"]["created_instances"] == 1
    created_id = payload["summary"]["created_instance_ids"][0]
    detail = instance_service.get_instance(created_id)
    assert detail.id == created_id
    assert detail.status in {"pending", "queued"}
    assert (tmp_path / "data" / "autonomous" / "loops.jsonl").exists()
