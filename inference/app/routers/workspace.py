from __future__ import annotations

from fastapi import APIRouter

from inference.app.workspace import build_workspace_overview

router = APIRouter(tags=["workspace"])


@router.get("/workspace")
def workspace_overview() -> dict:
    return build_workspace_overview()
