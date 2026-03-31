from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from inference.app.dependencies import get_metadata_service

router = APIRouter(tags=["health"])


@router.get("/health")
def healthcheck() -> dict[str, str]:
    """Lightweight health check that always responds quickly."""
    return {"status": "ok"}


@router.get("/status")
def status() -> dict[str, Any]:
    """Get system status with caching to avoid performance issues."""
    try:
        service = get_metadata_service()
        return service.status()
    except Exception as exc:
        return {
            "title": "AI-Factory API",
            "version": "0.2.0",
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


@router.get("/health/detailed")
def detailed_health() -> dict[str, Any]:
    """Detailed health check for monitoring systems."""
    try:
        from inference.app.workspace import build_workspace_overview_fast

        workspace_status = build_workspace_overview_fast()

        return {
            "status": "healthy",
            "workspace": workspace_status["summary"],
            "checks": workspace_status["readiness_checks"],
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
            "workspace": {"ready_checks": 0, "total_checks": 0},
            "checks": [],
        }
