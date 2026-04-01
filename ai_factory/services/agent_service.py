from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ai_factory.models.agent import AgentRecord, AgentDecisionRecord
from ai_factory.schemas.agent import AgentStatus, AgentDecision


class AgentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_agents(self) -> list[AgentStatus]:
        result = await self.db.execute(select(AgentRecord))
        agents = result.scalars().all()
        return [
            AgentStatus(
                id=a.id,
                type=a.type,
                status=a.status,
                last_action_at=a.last_action_at,
                decisions_today=a.decisions_today,
                current_target=a.current_target,
            )
            for a in agents
        ]

    async def list_decisions(self, limit: int = 50, agent_id: str | None = None) -> list[AgentDecision]:
        query = select(AgentDecisionRecord).order_by(AgentDecisionRecord.timestamp.desc()).limit(limit)
        if agent_id:
            query = query.where(AgentDecisionRecord.agent_id == agent_id)
        result = await self.db.execute(query)
        decisions = result.scalars().all()
        return [
            AgentDecision(
                id=d.id,
                agent_id=d.agent_id,
                agent_type=d.agent_type,
                action=d.action,
                target_id=d.target_id,
                reasoning=d.reasoning,
                metrics_snapshot=d.metrics_snapshot or {},
                timestamp=d.timestamp,
            )
            for d in decisions
        ]
