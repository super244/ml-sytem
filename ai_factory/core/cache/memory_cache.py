"""Memory-based cache implementation."""

import time
from typing import Any


class MemoryCache:
    """
    Simple in-memory cache with TTL support.

    This is a lightweight cache implementation for fallback when Redis is not available.
    """

    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        if key not in self._cache:
            self._misses += 1
            return None

        # Check if expired
        if key in self._expiry:
            if time.time() > self._expiry[key]:
                self.delete(key)
                self._misses += 1
                return None

        self._hits += 1
        return self._cache[key]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds. None means no expiry.
        """
        # Evict old entries if at max size
        if len(self._cache) >= self._max_size:
            self._evict_oldest()

        self._cache[key] = value

        if ttl is not None:
            self._expiry[key] = time.time() + ttl
        elif key in self._expiry:
            del self._expiry[key]

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """
        if key in self._cache:
            del self._cache[key]
            self._expiry.pop(key, None)
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._expiry.clear()

    def clear_prefix(self, prefix: str) -> int:
        """
        Clear all keys with a given prefix.

        Args:
            prefix: Key prefix to clear.

        Returns:
            Number of keys cleared.
        """
        keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
        for key in keys_to_delete:
            self.delete(key)
        return len(keys_to_delete)

    def _evict_oldest(self) -> None:
        """Evict oldest/expired entries."""
        # First, remove expired entries
        now = time.time()
        expired = [k for k, v in self._expiry.items() if v < now]
        for key in expired:
            self.delete(key)

        # If still at capacity, remove oldest 10%
        if len(self._cache) >= self._max_size:
            keys = list(self._cache.keys())
            to_remove = max(1, len(keys) // 10)
            for key in keys[:to_remove]:
                self.delete(key)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2%}",
        }
