from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from inference.app.main import app


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))


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
