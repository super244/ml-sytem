from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OpenAIChatMessage(BaseModel):
    role: str
    content: Any | None = None
    name: str | None = None

    model_config = ConfigDict(extra="allow")


class OpenAIChatCompletionRequest(BaseModel):
    model: str = "finetuned"
    messages: list[OpenAIChatMessage] = Field(..., min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    user: str | None = None

    model_config = ConfigDict(extra="allow")


class OpenAIUsageSnapshot(BaseModel):
    object: str = "usage"
    requests: int = 0
    stream_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    by_model: dict[str, dict[str, int]] = Field(default_factory=dict)
