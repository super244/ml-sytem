from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai_factory.core.instances.models import utc_now_iso


class InstanceEvent(BaseModel):
    timestamp: str = Field(default_factory=utc_now_iso)
    type: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
