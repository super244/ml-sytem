from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from math import ceil
from threading import Lock
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from ai_factory.core.hashing import sha256_text
from ai_factory.core.tokens import approximate_token_count
from inference.app.config import AppSettings
from inference.app.openai_schemas import OpenAIChatCompletionRequest
from inference.app.parameters import GenerationParameters


class OpenAIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code
        self.headers = headers or {}


class OpenAIAuthError(OpenAIError):
    def __init__(self, message: str = "Invalid API key"):
        super().__init__(
            message,
            status_code=401,
            error_type="invalid_api_key",
            headers={"WWW-Authenticate": 'Bearer realm="OpenAI-compatible API"'},
        )


class OpenAIRateLimitError(OpenAIError):
    def __init__(self, message: str = "Rate limit exceeded", *, retry_after: int | None = None):
        headers = {}
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        super().__init__(message, status_code=429, error_type="rate_limit_error", headers=headers)


class OpenAIModelError(OpenAIError):
    def __init__(self, message: str):
        super().__init__(message, status_code=404, error_type="invalid_request_error", param="model")


@dataclass
class OpenAICompletionResult:
    id: str
    created: int
    model: str
    content: str
    response: dict[str, Any]
    usage: dict[str, int]


class OpenAIUsageTracker:
    def __init__(self):
        self._lock = Lock()
        self._totals = {
            "requests": 0,
            "stream_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._by_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "requests": 0,
                "stream_requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        )

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, *, streamed: bool) -> None:
        with self._lock:
            self._totals["requests"] += 1
            if streamed:
                self._totals["stream_requests"] += 1
            self._totals["prompt_tokens"] += prompt_tokens
            self._totals["completion_tokens"] += completion_tokens
            self._totals["total_tokens"] += prompt_tokens + completion_tokens
            bucket = self._by_model[model]
            bucket["requests"] += 1
            if streamed:
                bucket["stream_requests"] += 1
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["total_tokens"] += prompt_tokens + completion_tokens

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "object": "usage",
                **self._totals,
                "by_model": {model: bucket.copy() for model, bucket in self._by_model.items()},
            }


class OpenAIRateLimiter:
    def __init__(self, requests_per_window: int, window_seconds: int):
        self.requests_per_window = max(0, requests_per_window)
        self.window_seconds = max(1, window_seconds)
        self._lock = Lock()
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    @property
    def enabled(self) -> bool:
        return self.requests_per_window > 0

    def allow(self, subject: str) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            bucket = self._requests[subject]
            while bucket and now - bucket[0] >= self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.requests_per_window:
                retry_after = max(1, ceil(self.window_seconds - (now - bucket[0])))
                raise OpenAIRateLimitError(retry_after=retry_after)
            bucket.append(now)


class OpenAIService:
    def __init__(self, generation_service: Any, settings: AppSettings):
        self.generation_service = generation_service
        self.settings = settings
        self.usage_tracker = OpenAIUsageTracker()
        self.rate_limiter = OpenAIRateLimiter(
            settings.openai_rate_limit_requests_per_minute,
            settings.openai_rate_limit_window_seconds,
        )

    def _extract_api_key(self, request: Request) -> str | None:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            return authorization.split(None, 1)[1].strip() or None
        api_key = request.headers.get("x-api-key")
        return api_key.strip() if api_key else None

    def _subject_for_request(self, request: Request) -> str:
        api_key = self._extract_api_key(request)
        if api_key:
            return f"key:{sha256_text(api_key)[:12]}"
        if request.client and request.client.host:
            return f"ip:{request.client.host}"
        return "anonymous"

    def authorize_request(self, request: Request, *, apply_rate_limit: bool = True) -> str:
        subject = self._subject_for_request(request)
        if self.settings.openai_api_keys:
            api_key = self._extract_api_key(request)
            if api_key is None or api_key not in self.settings.openai_api_keys:
                raise OpenAIAuthError()
        if apply_rate_limit:
            self.rate_limiter.allow(subject)
        return subject

    def _render_question(self, messages: list[Any]) -> str:
        parts: list[str] = []
        for message in messages:
            content = self._message_content(message.content)
            if not content:
                continue
            parts.append(f"{message.role}: {content}")
        question = "\n".join(parts).strip()
        if len(question) < 3:
            raise OpenAIError("messages must contain at least one meaningful message", param="messages")
        return question

    def _message_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        text = item.get("text")
                        if text is not None:
                            parts.append(str(text))
                    else:
                        text = item.get("text") or item.get("content")
                        if text is not None:
                            parts.append(str(text))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _build_generation_parameters(self, request: OpenAIChatCompletionRequest) -> GenerationParameters:
        temperature = 0.2 if request.temperature is None else request.temperature
        top_p = 0.95 if request.top_p is None else request.top_p
        max_new_tokens = max(64, request.max_tokens or 768)
        return GenerationParameters(
            question=self._render_question(request.messages),
            model_variant=request.model,
            prompt_preset="atlas_rigorous",
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            show_reasoning=False,
            difficulty_target="hard",
            num_samples=1,
            use_calculator=True,
            solver_mode="rigorous",
            output_format="text",
            use_cache=True,
            reference_answer=None,
            step_checks=[],
        )

    def _build_response(
        self, request: OpenAIChatCompletionRequest, generation: dict[str, Any]
    ) -> OpenAICompletionResult:
        content = str(generation.get("answer") or generation.get("raw_text") or "").strip()
        prompt_text = str(generation.get("prompt") or "")
        prompt_tokens = approximate_token_count(prompt_text)
        completion_tokens = approximate_token_count(content)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        self.usage_tracker.record(request.model, prompt_tokens, completion_tokens, streamed=False)
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        return OpenAICompletionResult(
            id=response_id,
            created=created,
            model=request.model,
            content=content,
            response=response,
            usage=usage,
        )

    def create_chat_completion(self, request: OpenAIChatCompletionRequest) -> OpenAICompletionResult:
        params = self._build_generation_parameters(request)
        try:
            generation = self.generation_service.generate(params)
        except FileNotFoundError as exc:
            raise OpenAIModelError(str(exc)) from exc
        except KeyError as exc:
            raise OpenAIModelError(str(exc)) from exc
        except OpenAIError:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise OpenAIError(str(exc), status_code=500, error_type="server_error") from exc
        return self._build_response(request, generation)

    def _chunk_text(self, text: str, chunk_size: int = 96) -> list[str]:
        if not text:
            return [""]
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def create_chat_stream(
        self,
        request: OpenAIChatCompletionRequest,
        include_usage: bool,
    ) -> StreamingResponse:
        result = self.create_chat_completion(request)

        async def event_stream():
            initial_chunk = {
                "id": result.id,
                "object": "chat.completion.chunk",
                "created": result.created,
                "model": result.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(initial_chunk, ensure_ascii=False)}\n\n"
            for chunk_text in self._chunk_text(result.content):
                chunk = {
                    "id": result.id,
                    "object": "chat.completion.chunk",
                    "created": result.created,
                    "model": result.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            final_chunk = {
                "id": result.id,
                "object": "chat.completion.chunk",
                "created": result.created,
                "model": result.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            if include_usage:
                final_chunk["usage"] = result.usage
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def usage_snapshot(self) -> dict[str, Any]:
        return self.usage_tracker.snapshot()
