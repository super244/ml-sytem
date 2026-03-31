from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Any
from pathlib import Path

from inference.app.workspace import build_workspace_overview, build_workspace_overview_fast

router = APIRouter(tags=["workspace"])


@router.get("/workspace")
def workspace_overview() -> dict:
    """Get full workspace overview."""
    try:
        return build_workspace_overview()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build workspace overview: {str(exc)}") from exc


@router.get("/workspace/fast")
def workspace_overview_fast() -> dict:
    """Get fast workspace overview for health checks."""
    try:
        return build_workspace_overview_fast()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build fast workspace overview: {str(exc)}") from exc


@router.get("/workspace/status")
def workspace_status() -> dict[str, Any]:
    """Get workspace status without heavy operations."""
    try:
        from inference.app.workspace import REPO_ROOT, _build_readiness_checks
        repo_root = REPO_ROOT
        readiness_checks = _build_readiness_checks(repo_root)
        ready_count = sum(1 for item in readiness_checks if item["ok"])
        
        return {
            "status": "ready" if ready_count == len(readiness_checks) else "partial",
            "repo_root": str(repo_root),
            "ready_checks": ready_count,
            "total_checks": len(readiness_checks),
            "checks": readiness_checks,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get workspace status: {str(exc)}") from exc
