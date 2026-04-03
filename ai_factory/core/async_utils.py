"""Async utilities for performance optimization."""

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


def async_wrap(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """
    Wrap synchronous function to run in thread pool.

    Args:
        func: Synchronous function to wrap.

    Returns:
        Async wrapper function.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    return wrapper


async def gather_with_concurrency(
    *tasks: Awaitable[Any],
    max_concurrency: int = 10,
) -> list[Any]:
    """
    Run tasks with limited concurrency.

    Args:
        *tasks: Tasks to run.
        max_concurrency: Maximum number of concurrent tasks.

    Returns:
        List of task results.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_task(task: Awaitable[Any]) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


class BatchProcessor:
    """
    Process items in batches for better performance.

    Useful for processing large datasets or API calls where individual
    processing would be slow or rate-limited.
    """

    def __init__(self, batch_size: int = 100, max_concurrency: int = 10):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch.
            max_concurrency: Maximum concurrent batch operations.
        """
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    async def process_items(
        self,
        items: list[Any],
        processor: Callable[[list[Any]], Awaitable[list[Any]]],
    ) -> list[Any]:
        """
        Process items in batches.

        Args:
            items: Items to process.
            processor: Async function to process a batch of items.

        Returns:
            Flattened list of processed results.
        """
        batches = [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]

        tasks = [processor(batch) for batch in batches]
        results = await gather_with_concurrency(*tasks, max_concurrency=self.max_concurrency)

        # Flatten results
        return [item for batch_result in results for item in batch_result]


class AsyncContextManager:
    """
    Context manager for async resource management.

    Provides a simple way to manage async resources with proper cleanup.
    """

    def __init__(
        self,
        acquire: Callable[[], Awaitable[R]],
        release: Callable[[R], Awaitable[None]],
    ):
        self._acquire = acquire
        self._resource: R | None = None
        self._release = release

    async def __aenter__(self) -> "AsyncContextManager":
        self._resource = await self._acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._resource is not None:
            await self._release(self._resource)


async def run_with_timeout(coro: Awaitable[T], timeout: float) -> T:
    """
    Run coroutine with timeout.

    Args:
        coro: Coroutine to run.
        timeout: Timeout in seconds.

    Returns:
        Result of the coroutine.

    Raises:
        asyncio.TimeoutError: If timeout is exceeded.
    """
    return await asyncio.wait_for(coro, timeout=timeout)


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs: Any,
) -> T:
    """
    Retry async function with exponential backoff.

    Args:
        func: Async function to retry.
        *args: Arguments to pass to function.
        max_retries: Maximum number of retries.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        **kwargs: Keyword arguments to pass to function.

    Returns:
        Result of the function.

    Raises:
        Exception: Last exception after all retries exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            if attempt < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                await asyncio.sleep(delay)

    if last_exception is None:
        raise RuntimeError("retry_with_backoff exhausted without capturing an exception")
    raise last_exception
