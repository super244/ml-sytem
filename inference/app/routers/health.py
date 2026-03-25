from __future__ import annotations

from fastapi import APIRouter

from inference.app.dependencies import get_metadata_service


router = APIRouter(tags=["health"])


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status")
def status() -> dict:
    return get_metadata_service().status()
