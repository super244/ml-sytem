from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from inference.app.config import AppSettings
from inference.app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_titan_status_route_returns_probe_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from inference.app.routers import titan as titan_router

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
    monkeypatch.setattr(titan_router, "get_settings", lambda: settings)
    monkeypatch.setattr(
        titan_router,
        "detect_titan_status",
        lambda repo_root: {
            "backend": "metal",
            "mode": "Metal-Direct",
            "silicon": "Apple M5 Max",
            "gpu_cap_pct": 90,
        },
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/v1/titan/status")

    assert response.status_code == 200
    assert response.json()["backend"] == "metal"
    assert response.json()["mode"] == "Metal-Direct"


@pytest.mark.anyio
async def test_titan_hardware_doc_route_returns_generated_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from inference.app.routers import titan as titan_router

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
    output_path = tmp_path / "HARDWARE.md"

    monkeypatch.setattr(titan_router, "get_settings", lambda: settings)
    monkeypatch.setattr(titan_router, "write_hardware_markdown", lambda repo_root: output_path)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/v1/titan/hardware-doc")

    assert response.status_code == 200
    assert response.json() == {"status": "generated", "path": str(output_path)}


@pytest.mark.anyio
async def test_titan_diagnostics_route_returns_runtime_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from inference.app.routers import titan as titan_router

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

    monkeypatch.setattr(titan_router, "get_settings", lambda: settings)
    monkeypatch.setattr(
        titan_router,
        "titan_diagnostics",
        lambda repo_root: {
            "status": {"backend": "metal"},
            "runtime": {"selected": "rust-canary", "canary_generation_enabled": False},
            "engine": {"runtime_ready": True},
            "rust_status": {"runtime": {"selected": "rust-canary"}},
        },
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/v1/titan/diagnostics")

    assert response.status_code == 200
    assert response.json()["runtime"]["selected"] == "rust-canary"
    assert response.json()["engine"]["runtime_ready"] is True
