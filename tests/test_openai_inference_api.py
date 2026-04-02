from __future__ import annotations

import httpx
import pytest
from starlette.requests import Request

from inference.app.config import get_settings
from inference.app.main import app
from inference.app.openai_schemas import OpenAIChatCompletionRequest
from inference.app.services.openai_service import OpenAIAuthError, OpenAIRateLimitError, OpenAIService


class DummyGenerationService:
    def __init__(self):
        self.calls = []

    def generate(self, params):
        self.calls.append(params)
        return {
            "model_variant": params.model_variant,
            "prompt": "system: solve the problem\nuser: What is 1+1?",
            "answer": "2",
            "raw_text": "2",
            "final_answer": "2",
            "reasoning_steps": [],
            "selected_score": 1.0,
            "candidates": [],
            "verification": None,
            "structured": None,
            "cache_hit": False,
            "telemetry_id": None,
            "latency_s": 0.01,
            "prompt_preset": params.prompt_preset,
            "candidate_agreement": 1.0,
        }


class DummyOpenAIService:
    def __init__(self):
        self.usage = {
            "object": "usage",
            "requests": 0,
            "stream_requests": 0,
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "by_model": {"finetuned": {"requests": 1}},
        }

    def authorize_request(self, raw_request, *, apply_rate_limit=True):
        return "anonymous"

    def create_chat_completion(self, request):
        self.usage["requests"] += 1
        return type(
            "Result",
            (),
            {
                "response": {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 123,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "2"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "total_tokens": 15,
                    },
                }
            },
        )()

    def create_chat_stream(self, request, include_usage):
        self.usage["requests"] += 1
        self.usage["stream_requests"] += 1

        async def event_stream():
            yield 'data: {"object": "chat.completion.chunk"}\n\n'
            yield "data: [DONE]\n\n"

        from fastapi.responses import StreamingResponse

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def usage_snapshot(self):
        return self.usage


def _make_request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [(key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": raw_headers,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "scheme": "http",
        "server": ("testserver", 80),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_openai_chat_completions_streams_sse(monkeypatch) -> None:
    from inference.app.routers import openai as openai_router

    monkeypatch.setattr(openai_router, "get_openai_service", lambda: DummyOpenAIService())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "finetuned",
                "messages": [{"role": "user", "content": "What is 1+1?"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            body = await response.aread()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b"chat.completion.chunk" in body
    assert b"[DONE]" in body


@pytest.mark.anyio
async def test_openai_chat_completions_returns_usage(monkeypatch) -> None:
    from inference.app.routers import openai as openai_router

    monkeypatch.setattr(openai_router, "get_openai_service", lambda: DummyOpenAIService())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "finetuned",
                "messages": [{"role": "user", "content": "What is 1+1?"}],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "2"
    assert body["usage"]["total_tokens"] == 15


def test_openai_service_auth_and_rate_limit() -> None:
    settings = get_settings()
    settings.openai_api_keys = ["secret"]
    settings.openai_rate_limit_requests_per_minute = 1
    settings.openai_rate_limit_window_seconds = 60

    auth_service = OpenAIService(DummyGenerationService(), settings)
    authorized_request = _make_request({"authorization": "Bearer secret"})
    assert auth_service.authorize_request(authorized_request) == "key:2bb80d537b1d"

    invalid_request = _make_request({"authorization": "Bearer wrong"})
    with pytest.raises(OpenAIAuthError):
        auth_service.authorize_request(invalid_request)

    limit_service = OpenAIService(DummyGenerationService(), settings)
    first_request = _make_request({"authorization": "Bearer secret"})
    limit_service.authorize_request(first_request)
    with pytest.raises(OpenAIRateLimitError):
        limit_service.authorize_request(first_request)


def test_openai_service_tracks_usage() -> None:
    settings = get_settings()
    settings.openai_api_keys = []
    settings.openai_rate_limit_requests_per_minute = 0
    settings.openai_rate_limit_window_seconds = 60
    service = OpenAIService(DummyGenerationService(), settings)

    payload = OpenAIChatCompletionRequest(
        model="finetuned",
        messages=[{"role": "user", "content": "What is 1+1?"}],
    )
    result = service.create_chat_completion(payload)

    assert result.response["choices"][0]["message"]["content"] == "2"
    snapshot = service.usage_snapshot()
    assert snapshot["requests"] == 1
    assert snapshot["total_tokens"] > 0
    assert snapshot["by_model"]["finetuned"]["requests"] == 1


def test_openai_service_tracks_stream_usage() -> None:
    settings = get_settings()
    settings.openai_api_keys = []
    settings.openai_rate_limit_requests_per_minute = 0
    settings.openai_rate_limit_window_seconds = 60
    service = OpenAIService(DummyGenerationService(), settings)

    payload = OpenAIChatCompletionRequest(
        model="finetuned",
        messages=[{"role": "user", "content": "What is 1+1?"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    response = service.create_chat_stream(payload, include_usage=True)

    assert response.media_type == "text/event-stream"
    snapshot = service.usage_snapshot()
    assert snapshot["requests"] == 1
    assert snapshot["stream_requests"] == 1
    assert snapshot["by_model"]["finetuned"]["requests"] == 1
    assert snapshot["by_model"]["finetuned"]["stream_requests"] == 1
