from typing import Any

from fastapi import APIRouter

from ai_factory.platform.monitoring.hardware import get_cluster_nodes

router = APIRouter(prefix="/cluster", tags=["cluster"])


@router.get("/nodes")
async def get_nodes() -> dict[str, Any]:
    nodes = get_cluster_nodes()
    return {"nodes": nodes}
