from __future__ import annotations

import asyncio

import pytest

from ai_factory.core.async_utils import BatchProcessor, async_wrap, gather_with_concurrency, retry_with_backoff


@pytest.mark.asyncio
async def test_async_wrap_runs_sync_function() -> None:
    def add(left: int, right: int) -> int:
        return left + right

    wrapped = async_wrap(add)
    assert await wrapped(2, 3) == 5


@pytest.mark.asyncio
async def test_gather_with_concurrency_enforces_limit() -> None:
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def worker() -> int:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return 1

    tasks = [worker() for _ in range(8)]
    results = await gather_with_concurrency(*tasks, max_concurrency=2)
    assert results == [1] * 8
    assert peak <= 2


@pytest.mark.asyncio
async def test_gather_with_concurrency_rejects_zero_limit() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        await gather_with_concurrency(max_concurrency=0)


@pytest.mark.asyncio
async def test_batch_processor_flattens_batch_results() -> None:
    processor = BatchProcessor(batch_size=3, max_concurrency=2)

    async def process_batch(batch: list[int]) -> list[int]:
        return [value * 2 for value in batch]

    output = await processor.process_items([1, 2, 3, 4, 5], process_batch)
    assert output == [2, 4, 6, 8, 10]


@pytest.mark.asyncio
async def test_retry_with_backoff_eventual_success() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("try again")
        return "ok"

    result = await retry_with_backoff(flaky, max_retries=3, base_delay=0.0, max_delay=0.0)
    assert result == "ok"
    assert attempts == 3
