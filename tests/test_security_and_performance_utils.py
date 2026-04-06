from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from ai_factory.core.async_utils import BatchProcessor, async_wrap, gather_with_concurrency
from ai_factory.core.cache import CacheManager, cached
from ai_factory.core.security import SecureExecutor, SecureHasher, SecureSettings


def test_secure_hasher_hash_and_verify_round_trip() -> None:
    digest, salt = SecureHasher.hash_data("sensitive-value")
    assert len(digest) == 64
    assert SecureHasher.verify_data("sensitive-value", digest, salt) is True
    assert SecureHasher.verify_data("wrong", digest, salt) is False


def test_secure_settings_from_env_and_validation() -> None:
    env: Mapping[str, str] = {
        "AI_FACTORY_DATABASE_URL": "sqlite:///tmp/test.db",
        "AI_FACTORY_SECRET_KEY": "x" * 32,
        "AI_FACTORY_API_TOKEN": "token-123",
    }
    settings = SecureSettings.from_env(env)
    assert settings.database_url.startswith("sqlite://")
    assert settings.api_token == "token-123"


def test_secure_executor_rejects_unapproved_commands() -> None:
    with pytest.raises(ValueError):
        SecureExecutor.execute_command("rm -rf /tmp/test", check=False)


def test_secure_executor_runs_allowed_command() -> None:
    result = SecureExecutor.execute_command("python -c \"print('ok')\"")
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


@pytest.mark.asyncio
async def test_cache_manager_set_get_and_cached_decorator() -> None:
    CacheManager._instance = None
    manager = await CacheManager.get_instance()
    await manager.set("k1", {"value": 1}, category="api_responses", ttl=30)
    value = await manager.get("k1", category="api_responses")
    assert value == {"value": 1}

    calls = {"count": 0}

    @cached(category="computed_metrics", ttl=30)
    async def _expensive(x: int) -> int:
        calls["count"] += 1
        return x * 2

    first = await _expensive(4)
    second = await _expensive(4)
    assert first == 8 and second == 8
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_async_utils_batch_and_concurrency_helpers() -> None:
    wrapped = async_wrap(lambda value: value + 1)
    assert await wrapped(2) == 3

    async def _batch_processor(values: list[int]) -> list[int]:
        await asyncio.sleep(0.01)
        return [v * 2 for v in values]

    processor = BatchProcessor(batch_size=2, max_concurrency=2)
    batched = await processor.process_items([1, 2, 3, 4], _batch_processor)
    assert batched == [2, 4, 6, 8]

    results = await gather_with_concurrency(_batch_processor([5]), _batch_processor([6]), max_concurrency=1)
    assert results == [[10], [12]]
