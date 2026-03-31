from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from inference.app.config import AppSettings
from inference.app.main import app
from inference.app.metadata import MetadataService


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_dataset_telemetry_routes_support_promote_and_discard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from inference.app.routers import datasets as datasets_router

    telemetry_dir = tmp_path / "data" / "telemetry"
    promoted_file = telemetry_dir / "promoted.jsonl"
    discarded_file = telemetry_dir / "discarded.jsonl"
    _write_jsonl(
        telemetry_dir / "flagged.jsonl",
        [
            {
                "timestamp": 101.0,
                "prompt": "bad prompt",
                "assistant_output": "wrong answer",
                "expected_output": "right answer",
                "model_variant": "demo",
            },
            {
                "timestamp": 99.0,
                "prompt": "another prompt",
                "assistant_output": "bad answer",
                "expected_output": "better answer",
                "model_variant": "demo-2",
            },
        ],
    )
    monkeypatch.setattr(datasets_router, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(datasets_router, "TELEMETRY_DIR", telemetry_dir)
    monkeypatch.setattr(datasets_router, "PROMOTED_TELEMETRY_FILE", promoted_file)
    monkeypatch.setattr(datasets_router, "DISCARDED_TELEMETRY_FILE", discarded_file)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        list_response = await client.get("/v1/datasets/telemetry")
        assert list_response.status_code == 200
        records = list_response.json()["telemetry"]
        assert len(records) == 2
        assert records[0]["id"]

        promote_response = await client.post(f"/v1/datasets/telemetry/{records[0]['id']}/promote")
        discard_response = await client.post(f"/v1/datasets/telemetry/{records[1]['id']}/discard")
        remaining_response = await client.get("/v1/datasets/telemetry")

    assert promote_response.status_code == 200
    assert promote_response.json()["status"] == "promoted"
    assert "promoted.jsonl" in promote_response.json()["destination"]
    assert promoted_file.exists()

    assert discard_response.status_code == 200
    assert discard_response.json()["status"] == "discarded"
    assert discarded_file.exists()

    assert remaining_response.status_code == 200
    assert remaining_response.json()["telemetry"] == []


@pytest.mark.anyio
async def test_agent_routes_support_updating_registered_agents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from inference.app.routers import agents as agents_router

    registry = tmp_path / "data" / "agents" / "registry.jsonl"
    monkeypatch.setattr(agents_router, "AGENTS_FILE", registry)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        list_response = await client.get("/v1/agents/swarm")
        assert list_response.status_code == 200
        agent = list_response.json()["swarm"][0]

        update_response = await client.patch(
            f"/v1/agents/{agent['id']}",
            json={
                "role": "Owns telemetry triage and synthetic replay curation.",
                "status": "sleeping",
            },
        )
        refreshed_response = await client.get("/v1/agents/swarm")

    assert update_response.status_code == 200
    payload = update_response.json()
    assert payload["agent"]["status"] == "sleeping"
    assert "telemetry triage" in payload["agent"]["role"]

    refreshed = refreshed_response.json()["swarm"]
    assert any(item["id"] == agent["id"] and item["status"] == "sleeping" for item in refreshed)


@pytest.mark.anyio
async def test_dataset_dashboard_exposes_processed_and_pack_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from inference.app.routers import datasets as datasets_router

    _write(
        tmp_path / "data" / "catalog.json",
        """
{
  "generated_at": "local-build",
  "summary": {"num_datasets": 1, "custom_datasets": 1, "public_datasets": 0, "total_bytes": 128, "total_rows": 4},
  "datasets": [
    {
      "id": "custom_derivative_mastery",
      "title": "Derivative Mastery Corpus",
      "kind": "custom",
      "family": "derivatives",
      "topic": "calculus",
      "path": "data/custom/custom_derivative_mastery.jsonl",
      "num_rows": 4,
      "size_bytes": 128,
      "description": "toy corpus",
      "preview_examples": []
    }
  ]
}
""".strip(),
    )
    _write(
        tmp_path / "data" / "processed" / "manifest.json",
        """
{
  "schema_version": "v2",
  "manifest_type": "dataset",
  "build": {
    "build_id": "atlas-core-test",
    "created_at": "2026-03-29T12:00:00Z",
    "git_sha": "abc1234",
    "config_path": "data/configs/processing.yaml",
    "config_sha256": "deadbeef",
    "seed": 42,
    "notes": ["atlas_math_lab_v2"]
  },
  "metadata": {
    "lineage_summary_path": "data/processed/lineage_summary.json",
    "source_summaries": [{"id": "custom_derivative_mastery", "kind": "local", "path": "data/custom/custom_derivative_mastery.jsonl", "version": "2026.03", "sample_ratio": 1.0, "optional": false, "status": "loaded", "rows_loaded": 4, "rows_selected": 4}],
    "source_warnings": []
  },
  "outputs": [],
  "source_lineage": []
}
""".strip(),
    )
    _write(
        tmp_path / "data" / "processed" / "lineage_summary.json",
        """
{
  "total_records": 4,
  "contamination": {"exact_matches": 0, "near_matches": 1, "contaminated_records": 1, "failure_cases": 0},
  "by_split": {"train": 4},
  "by_loader": {"local": 4},
  "groups": [
    {
      "source_id": "custom_derivative_mastery",
      "loader": "local",
      "version": "2026.03",
      "origin_path": "data/custom/custom_derivative_mastery.jsonl",
      "dataset_split": "train",
      "failure_case": false,
      "record_count": 4,
      "exact_matches": 0,
      "near_matches": 1,
      "contaminated_records": 1,
      "max_similarity": 0.81
    }
  ]
}
""".strip(),
    )
    _write(
        tmp_path / "data" / "processed" / "pack_summary.json",
        """
{
  "packs": [
    {
      "id": "core_train_mix",
      "description": "Primary mixed-source training corpus.",
      "num_rows": 4,
      "size_bytes": 128,
      "path": "data/processed/packs/core_train_mix/records.jsonl",
      "manifest_path": "data/processed/packs/core_train_mix/manifest.json",
      "card_path": "data/processed/packs/core_train_mix/card.md",
      "build_id": "atlas-core-test",
      "build": {
        "build_id": "atlas-core-test",
        "created_at": "2026-03-29T12:00:00Z",
        "git_sha": "abc1234",
        "config_path": "data/configs/processing.yaml",
        "seed": 42,
        "notes": []
      },
      "stats": {"num_records": 4}
    }
  ]
}
""".strip(),
    )
    _write(
        tmp_path / "data" / "processed" / "packs" / "core_train_mix" / "manifest.json",
        """
{
  "schema_version": "v2",
  "manifest_type": "pack",
  "build": {
    "build_id": "atlas-core-test",
    "created_at": "2026-03-29T12:00:00Z",
    "git_sha": "abc1234",
    "config_path": "data/configs/processing.yaml",
    "seed": 42,
    "notes": []
  },
  "pack_id": "core_train_mix",
  "description": "Primary mixed-source training corpus.",
  "outputs": [
    {
      "path": "data/processed/packs/core_train_mix/records.jsonl",
      "sha256": "0123",
      "size_bytes": 128,
      "num_rows": 4
    }
  ],
  "stats": {
    "num_records": 4
  },
  "metadata": {
    "card_path": "data/processed/packs/core_train_mix/card.md"
  }
}
""".strip(),
    )
    _write(
        tmp_path / "data" / "processed" / "packs" / "core_train_mix" / "card.md",
        "# core_train_mix\n",
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
    metadata_service = MetadataService(settings, [], {}, None)
    monkeypatch.setattr(datasets_router, "get_metadata_service", lambda: metadata_service)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/v1/datasets")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provenance"]["processed_manifest"]["build"]["build_id"] == "atlas-core-test"
    assert payload["provenance"]["processed_manifest"]["metadata"]["lineage_summary_path"].endswith(
        "lineage_summary.json"
    )
    assert payload["provenance"]["lineage_summary"]["total_records"] == 4
    assert payload["provenance"]["pack_manifests"][0]["pack_id"] == "core_train_mix"
    assert payload["packs"][0]["manifest_path"].endswith("manifest.json")
