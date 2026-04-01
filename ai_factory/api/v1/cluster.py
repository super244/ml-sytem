from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.database import get_session
from ai_factory.services.cluster_service import ClusterService
from ai_factory.schemas.cluster import NodeStatus

router = APIRouter(prefix="/cluster", tags=["cluster"])


@router.get("/nodes", response_model=list[NodeStatus])
async def list_nodes(db: AsyncSession = Depends(get_session)):
    service = ClusterService(db)
    return await service.list_nodes()
