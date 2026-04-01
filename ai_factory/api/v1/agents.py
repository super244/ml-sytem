from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.database import get_session
from ai_factory.services.agent_service import AgentService
from ai_factory.schemas.agent import AgentStatus, AgentDecision

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentStatus])
async def list_agents(db: AsyncSession = Depends(get_session)):
    service = AgentService(db)
    return await service.list_agents()


@router.get("/decisions", response_model=list[AgentDecision])
async def list_decisions(
    limit: int = Query(50, ge=1, le=200),
    agent_id: str | None = Query(None),
    db: AsyncSession = Depends(get_session),
):
    service = AgentService(db)
    return await service.list_decisions(limit=limit, agent_id=agent_id)
