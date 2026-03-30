from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from inference.app.config import AppSettings
from inference.app.main import app
from inference.app.schemas import InstanceCreateRequest
from inference.app.services.instance_service import InstanceService


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_instance_service_resolves_relative_config_paths_against_repo_root(tmp_path: Path):
    _write(
        tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml",
        "run_name: demo\ntraining:\n  artifacts_dir: artifacts\n",
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
        telemetry_path=str(tmp_path / "artifacts" / "inference" / "telemetry" / "requests.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
    )
    service = InstanceService(settings)

    detail = service.create_instance(InstanceCreateRequest(config_path="configs/finetune.yaml", start=False))

    assert detail.config_path == str((tmp_path / "configs" / "finetune.yaml").resolve())
    assert detail.lifecycle.stage == "finetune"


@pytest.mark.anyio
async def test_mission_control_endpoint_aggregates_lab_surfaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ai_factory.platform.monitoring import hardware
    from inference.app.routers import lab as lab_router

    _write(
        tmp_path / "data" / "catalog.json",
        '{"summary": {"num_datasets": 3}}',
    )
    _write(
        tmp_path / "data" / "processed" / "pack_summary.json",
        '{"packs": [{"id": "core_train_mix"}]}',
    )
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
        tmp_path / "evaluation" / "configs" / "base_vs_finetuned.yaml",
        "output_dir: evaluation/results/demo\n",
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
        ("models:\n  - name: base\n    base_model: Qwen/Qwen2.5-Math-1.5B-Instruct\n"),
    )
    _write(
        tmp_path / "data" / "agents" / "registry.jsonl",
        "\n".join(
            [
                '{"id":"agent-data-01","name":"Data Curator","role":"Curates rows","model":"gpt-4o","status":"active","created_at":10,"tokens_used":123}',
            ]
        ),
    )
    _write(
        tmp_path / "data" / "automl" / "sweeps.jsonl",
        "\n".join(
            [
                '{"id":"sweep-001","name":"Sweep One","base_model":"base","strategy":"bayesian","status":"running","num_trials":3,"completed_trials":2,"created_at":20,"best_trial":{"trial_id":"trial-00","status":"completed","params":{"learning_rate":0.0001,"batch_size":8,"warmup_ratio":0.03,"lora_rank":8},"metrics":{"final_loss":0.4,"accuracy":0.88,"perplexity":1.49},"duration_s":120},"trials":[]}',
            ]
        ),
    )
    _write(
        tmp_path / "data" / "telemetry" / "flagged.jsonl",
        "\n".join(
            [
                '{"timestamp": 5, "prompt": "a", "assistant_output": "b", "expected_output": "c", "model_variant": "base", "latency_s": 1.2}',
            ]
        ),
    )
    _write(
        tmp_path / "artifacts" / "inference" / "telemetry" / "requests.jsonl",
        "\n".join(
            [
                '{"timestamp": 7, "event_type": "generate", "model_variant": "base"}',
                '{"timestamp": 8, "event_type": "generate", "model_variant": "finetuned"}',
            ]
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
        telemetry_path=str(tmp_path / "artifacts" / "inference" / "telemetry" / "requests.jsonl"),
        cache_enabled=False,
        telemetry_enabled=False,
    )
    instance_service = InstanceService(settings)
    instance_service.create_instance(InstanceCreateRequest(config_path="configs/finetune.yaml", start=False))

    monkeypatch.setattr(lab_router, "get_settings", lambda: settings)
    monkeypatch.setattr(lab_router, "get_instance_service", lambda: instance_service)
    monkeypatch.setattr(
        hardware,
        "get_cluster_nodes",
        lambda: [
            {
                "id": "gpu-1",
                "name": "Cluster Worker 1",
                "type": "NVIDIA T4",
                "memory": "16GB",
                "usage": 42,
                "status": "idle",
                "activeJobs": 0,
            }
        ],
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/v1/lab/mission-control")

    assert response.status_code == 200
    payload = response.json()
    assert payload["repo_root"] == str(tmp_path)
    assert payload["workspace"]["summary"]["datasets"] == 3
    assert payload["control_plane"]["orchestration_summary"]["runs"] >= 1
    assert payload["agents"]["count"] == 1
    assert payload["automl"]["count"] == 1
    assert payload["cluster"]["nodes"][0]["id"] == "gpu-1"
    assert payload["telemetry"]["flagged"]["count"] == 1
    assert payload["telemetry"]["requests"]["by_model"]["finetuned"] == 1
    assert payload["criticality"]["counts"]["warning"] >= 1
    assert any(
        item["id"] == "telemetry-backlog" and item["surface"] == "datasets" and item["metric_value"] == "1"
        for item in payload["recommendations"]
    )
    assert payload["summary"]["telemetry_requests"] == 2
