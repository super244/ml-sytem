from __future__ import annotations

import pytest
import pytest_asyncio

from ai_factory.core.cache import CacheManager, cached


@pytest_asyncio.fixture(autouse=True)
async def _reset_cache_manager() -> None:
    await CacheManager.reset_instance()
    yield
    await CacheManager.reset_instance()


@pytest.mark.asyncio
async def test_cache_manager_set_get_delete_cycle() -> None:
    cache = await CacheManager.get_instance()
    await cache.set("alpha", {"value": 1}, category="api_responses", ttl=60)

    assert await cache.get("alpha", category="api_responses") == {"value": 1}
    assert await cache.delete("alpha", category="api_responses") is True
    assert await cache.get("alpha", category="api_responses") is None


@pytest.mark.asyncio
async def test_cached_decorator_reuses_cached_result() -> None:
    calls = 0

    @cached(category="computed_metrics", ttl=60)
    async def build_payload(value: int) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": value}

    first = await build_payload(7)
    second = await build_payload(7)

    assert first == {"value": 7}
    assert second == {"value": 7}
    assert calls == 1


@pytest.mark.asyncio
async def test_cache_manager_stats_exposes_memory_cache_info() -> None:
    cache = await CacheManager.get_instance()
    await cache.set("item", "value", ttl=60)
    _ = await cache.get("item")

    stats = await cache.get_stats()
    assert "memory_cache" in stats
    assert stats["memory_cache"]["size"] >= 1
