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
async def test_orchestration_routes_expose_runs_tasks_and_summary(tmp_path: Path, monkeypatch):
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
    monkeypatch.setattr(orchestration_router, "get_instance_service", lambda: service)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        runs_response = await client.get("/v1/orchestration/runs")
        assert runs_response.status_code == 200
        assert len(runs_response.json()["runs"]) == 1

        run_id = detail.orchestration_run_id
        assert run_id is not None
        detail_response = await client.get(f"/v1/orchestration/runs/{run_id}")
        tasks_response = await client.get(f"/v1/orchestration/runs/{run_id}/tasks")
        summary_response = await client.get("/v1/orchestration/summary")

    assert detail_response.status_code == 200
    assert tasks_response.status_code == 200
    assert len(tasks_response.json()["tasks"]) == 1
    assert summary_response.status_code == 200
    assert summary_response.json()["summary"]["runs"] == 1
