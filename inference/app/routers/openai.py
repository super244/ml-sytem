from __future__ import annotations

from fastapi import APIRouter, Request

from inference.app.dependencies import get_openai_service
from inference.app.openai_schemas import OpenAIChatCompletionRequest

router = APIRouter(tags=["openai"])


@router.post("/chat/completions")
def create_chat_completions(request: OpenAIChatCompletionRequest, raw_request: Request):
    service = get_openai_service()
    service.authorize_request(raw_request)
    if request.stream:
        include_usage = bool((request.stream_options or {}).get("include_usage"))
        return service.create_chat_stream(request, include_usage=include_usage)
    return service.create_chat_completion(request).response


@router.get("/usage")
def usage(raw_request: Request):
    service = get_openai_service()
    service.authorize_request(raw_request, apply_rate_limit=False)
    return service.usage_snapshot()
