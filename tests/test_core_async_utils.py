from __future__ import annotations

import asyncio

import pytest

from ai_factory.core.async_utils import (
    BatchProcessor,
    async_wrap,
    gather_with_concurrency,
    retry_with_backoff,
    run_with_timeout,
)


@pytest.mark.asyncio
async def test_async_wrap_executes_sync_function() -> None:
    @async_wrap
    def add(a: int, b: int) -> int:
        return a + b

    assert await add(2, 3) == 5


@pytest.mark.asyncio
async def test_gather_with_concurrency_enforces_limit() -> None:
    active = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def work() -> int:
        nonlocal active, max_seen
        async with lock:
            active += 1
            max_seen = max(max_seen, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return 1

    results = await gather_with_concurrency(*(work() for _ in range(8)), max_concurrency=3)
    assert sum(results) == 8
    assert max_seen <= 3


@pytest.mark.asyncio
async def test_gather_with_concurrency_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
        await gather_with_concurrency(max_concurrency=0)


@pytest.mark.asyncio
async def test_batch_processor_processes_and_flattens_results() -> None:
    processor = BatchProcessor(batch_size=2, max_concurrency=2)

    async def process_batch(batch: list[int]) -> list[int]:
        await asyncio.sleep(0)
        return [item * 2 for item in batch]

    result = await processor.process_items([1, 2, 3, 4, 5], process_batch)
    assert result == [2, 4, 6, 8, 10]


@pytest.mark.asyncio
async def test_run_with_timeout_times_out() -> None:
    with pytest.raises(asyncio.TimeoutError):
        await run_with_timeout(asyncio.sleep(0.2), timeout=0.01)


@pytest.mark.asyncio
async def test_retry_with_backoff_retries_until_success() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("try again")
        return "ok"

    assert await retry_with_backoff(flaky, max_retries=3, base_delay=0.0) == "ok"
    assert attempts == 3
