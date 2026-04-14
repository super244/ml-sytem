from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ai_factory.titan import detect_titan_status
from ai_factory.version import VERSION
from inference.app.config import get_settings
from inference.app.dependencies import get_metadata_service

router = APIRouter(tags=["health"])


@router.get("/health")
def healthcheck() -> dict[str, str]:
    """Lightweight health check that always responds quickly."""
    return {"status": "ok"}


@router.get("/health/detailed")
def detailed_health() -> dict[str, Any]:
    """Detailed health check with Titan telemetry and workspace status."""
    try:
        from inference.app.workspace import build_workspace_overview_fast

        workspace_status = build_workspace_overview_fast()
        settings = get_settings()
        titan_status = detect_titan_status(settings.repo_root)

        return {
            "status": "healthy" if workspace_status.get("status") == "available" else "degraded",
            "errors": workspace_status.get("errors", []),
            "workspace": workspace_status["summary"],
            "checks": workspace_status["readiness_checks"],
            "titan": {
                "backend": titan_status.get("backend"),
                "gpu_name": titan_status.get("gpu_name"),
                "gpu_count": titan_status.get("gpu_count"),
                "supports_metal": titan_status.get("supports_metal"),
                "supports_cuda": titan_status.get("supports_cuda"),
                "runtime_mode": titan_status.get("engine", {}).get("runtime_mode"),
            },
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
            "workspace": {"ready_checks": 0, "total_checks": 0},
            "checks": [],
            "titan": {"error": str(exc)},
        }


@router.get("/health/metrics")
def metrics() -> dict[str, Any]:
    """Prometheus-style metrics endpoint."""
    try:
        from inference.app.workspace import build_workspace_overview_fast

        workspace_status = build_workspace_overview_fast()
        settings = get_settings()
        titan_status = detect_titan_status(settings.repo_root)

        return {
            "api_version": VERSION,
            "status": workspace_status.get("status", "unknown"),
            "uptime_seconds": workspace_status.get("uptime_seconds", 0),
            "ready_checks": workspace_status.get("summary", {}).get("ready_checks", 0),
            "total_checks": workspace_status.get("summary", {}).get("total_checks", 0),
            "error_count": len(workspace_status.get("errors", [])),
            "titan_backend": titan_status.get("backend", "unknown"),
            "titan_gpu_count": titan_status.get("gpu_count", 0),
            "titan_runtime_ready": titan_status.get("engine", {}).get("runtime_ready", False),
        }
    except Exception as exc:
        return {
            "api_version": VERSION,
            "status": "error",
            "error": str(exc),
            "ready_checks": 0,
            "total_checks": 0,
            "error_count": 1,
        }


@router.get("/status")
def status(service: Any = Depends(get_metadata_service)) -> dict[str, Any]:
    """Get system status with caching to avoid performance issues."""
    try:
        return service.status()
    except Exception as exc:
        return {
            "title": "AI-Factory API",
            "version": VERSION,
            "status": "degraded",
            "error": str(exc),
            "models": [],
            "cache": {"enabled": False},
            "benchmarks": 0,
            "runs": 0,
            "instance_count": 0,
            "running_count": 0,
            "uptime_seconds": 0,
        }
