from typing import Literal
from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True


class CompletionResponse(BaseModel):
    id: str
    model_id: str
    prompt: str
    completion: str
    tokens_generated: int
    tokens_per_second: float
    time_to_first_token_ms: float
    confidence_score: float
    finish_reason: Literal["stop", "length", "error"]


class FeedbackFlagRequest(BaseModel):
    completion_id: str
    reason: Literal["incorrect", "hallucination", "refusal", "low_quality", "other"]
    notes: str | None = None
    add_to_dataset: bool = True


class FeedbackFlagResponse(BaseModel):
    status: str
    completion_id: str
