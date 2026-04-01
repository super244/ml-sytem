from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict


class AgentStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    type: Literal["evaluator", "pruner", "promoter"]
    status: Literal["active", "standby", "paused", "error"]
    last_action_at: datetime | None = None
    decisions_today: int
    current_target: str | None = None


class AgentDecision(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    agent_id: str
    agent_type: Literal["evaluator", "pruner", "promoter"]
    action: Literal["scored", "pruned", "promoted", "escalated", "flagged"]
    target_id: str
    reasoning: str
    metrics_snapshot: dict
    timestamp: datetime
