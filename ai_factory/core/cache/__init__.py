"""Advanced caching system for AI-Factory."""

from __future__ import annotations

import functools
import json
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

from .memory_cache import MemoryCache

__all__ = ["CacheManager", "cached"]

logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


class CacheManager:
    """
    Advanced cache manager with multiple backends.

    Supports both memory-based and Redis-based caching with automatic fallback.
    """

    _instance: CacheManager | None = None
    _lock = threading.Lock()

    def __init__(self, redis_url: str | None = None):
        self.memory_cache = MemoryCache()
        self.redis_client = None
        self._redis_url = redis_url

        self.ttl_settings: dict[str, int] = {
            "model_predictions": 300,  # 5 minutes
            "user_sessions": 1800,  # 30 minutes
            "api_responses": 60,  # 1 minute
            "computed_metrics": 600,  # 10 minutes
            "default": 300,
        }

    @classmethod
    async def get_instance(cls, redis_url: str | None = None) -> CacheManager:
        """Get singleton instance of CacheManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(redis_url)
                    if redis_url:
                        # Import redis only if needed
                        try:
                            import redis.asyncio as aioredis

                            cls._instance.redis_client = aioredis.from_url(redis_url)
                        except ImportError:
                            logger.debug("redis is not installed; using memory cache only")
        return cls._instance

    async def get(self, key: str, category: str = "default") -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key.
            category: Cache category for TTL selection.

        Returns:
            Cached value or None if not found/expired.
        """
        cache_key = f"{category}:{key}"

        # Try Redis first if available
        if self.redis_client:
            try:
                value = await self.redis_client.get(cache_key)
                if value:
                    return json.loads(value)
            except Exception as exc:  # noqa: BLE001
                logger.debug("redis cache get failed for %s: %s", cache_key, exc)

        # Fallback to memory cache
        return self.memory_cache.get(cache_key)

    async def set(
        self,
        key: str,
        value: Any,
        category: str = "default",
        ttl: int | None = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            category: Cache category for TTL selection.
            ttl: Optional custom TTL in seconds.
        """
        cache_key = f"{category}:{key}"
        ttl = ttl or self.ttl_settings.get(category, self.ttl_settings["default"])

        # Store in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(value, default=str),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("redis cache set failed for %s: %s", cache_key, exc)

        # Store in memory cache as fallback
        self.memory_cache.set(cache_key, value, ttl)

    async def delete(self, key: str, category: str = "default") -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key.
            category: Cache category.

        Returns:
            True if deleted, False otherwise.
        """
        cache_key = f"{category}:{key}"
        deleted = False

        if self.redis_client:
            try:
                await self.redis_client.delete(cache_key)
                deleted = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("redis cache delete failed for %s: %s", cache_key, exc)

        if self.memory_cache.delete(cache_key):
            deleted = True

        return deleted

    async def clear_category(self, category: str) -> None:
        """
        Clear all keys in a category.

        Args:
            category: Cache category to clear.
        """
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(f"{category}:*")
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as exc:  # noqa: BLE001
                logger.debug("redis category clear failed for %s: %s", category, exc)

        self.memory_cache.clear_prefix(f"{category}:")

    async def close(self) -> None:
        """Close cache connections."""
        if self.redis_client:
            try:
                await self.redis_client.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("redis close failed: %s", exc)

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset singleton cache manager state for tests and isolated tasks."""
        if cls._instance is not None:
            await cls._instance.close()
        cls._instance = None

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache": self.memory_cache.stats(),
            "redis_enabled": self.redis_client is not None,
        }

        if self.redis_client:
            try:
                info = await self.redis_client.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human", "unknown")
            except Exception as exc:  # noqa: BLE001
                logger.debug("redis info lookup failed: %s", exc)

        return stats


def cached(
    category: str = "default", ttl: int | None = None
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for caching async function results.

    Args:
        category: Cache category.
        ttl: Optional custom TTL.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache_manager = await CacheManager.get_instance()

            # Generate cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_result = await cache_manager.get(cache_key, category)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, category, ttl)

            return result

        return wrapper

    return decorator
