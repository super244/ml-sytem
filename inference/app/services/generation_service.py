from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from inference.app.config import AppSettings
from inference.app.parameters import GenerationParameters

if TYPE_CHECKING:
    from inference.app.generation import MathGenerator

logger = logging.getLogger(__name__)


class GenerationService:
    """High-performance text generation service with hardware optimization."""

    def __init__(self, generator: MathGenerator, settings: AppSettings):
        self.generator = generator
        self.settings = settings
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
        }

    def generate(self, params: GenerationParameters) -> dict[str, Any]:
        """Generate text with the specified parameters."""
        self._stats["total_requests"] += 1
        result = self.generator.generate(params)
        if "usage" in result and "completion_tokens" in result["usage"]:
            self._stats["total_tokens"] += result["usage"]["completion_tokens"]
        return result

    async def generate_stream(self, params: GenerationParameters) -> AsyncIterator[dict[str, Any]]:
        """Stream generation results token by token."""
        self._stats["total_requests"] += 1
        async for chunk in self.generator.generate_stream(params):
            yield chunk

    def batch_generate(self, params_list: list[GenerationParameters], batch_size: int = 8) -> list[dict[str, Any]]:
        """Generate multiple outputs in batches for efficiency."""
        results = []
        for i in range(0, len(params_list), batch_size):
            batch = params_list[i : i + batch_size]
            batch_results = [self.generate(p) for p in batch]
            results.extend(batch_results)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return self._stats.copy()
