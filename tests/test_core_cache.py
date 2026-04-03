from __future__ import annotations

import asyncio

import pytest

from ai_factory.core.cache import CacheManager, cached
from ai_factory.core.cache.memory_cache import MemoryCache


def test_memory_cache_set_get_delete_and_ttl() -> None:
    cache = MemoryCache(max_size=10)
    cache.set("k1", {"x": 1}, ttl=1)
    assert cache.get("k1") == {"x": 1}
    assert cache.delete("k1") is True
    assert cache.get("k1") is None


def test_memory_cache_clear_prefix() -> None:
    cache = MemoryCache(max_size=10)
    cache.set("a:1", 1)
    cache.set("a:2", 2)
    cache.set("b:1", 3)
    assert cache.clear_prefix("a:") == 2
    assert cache.get("a:1") is None
    assert cache.get("b:1") == 3


@pytest.mark.asyncio
async def test_cache_manager_set_get_delete_without_redis() -> None:
    await CacheManager.reset_instance()
    manager = await CacheManager.get_instance(redis_url=None)
    await manager.set("item", {"ok": True}, category="api_responses", ttl=60)
    assert await manager.get("item", category="api_responses") == {"ok": True}
    assert await manager.delete("item", category="api_responses") is True
    assert await manager.get("item", category="api_responses") is None


@pytest.mark.asyncio
async def test_cached_decorator_memoizes_results() -> None:
    await CacheManager.reset_instance()
    calls = 0

    @cached(category="computed_metrics", ttl=60)
    async def expensive(value: int) -> int:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return value * 10

    assert await expensive(7) == 70
    assert await expensive(7) == 70
    assert calls == 1
