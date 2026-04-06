"""Memory-based cache implementation with LRU eviction and TTL support."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class CacheEntry:
    """Represents a cached value with metadata."""

    __slots__ = ("value", "expiry", "access_count", "created_at")

    def __init__(self, value: Any, expiry: float | None = None) -> None:
        self.value = value
        self.expiry = expiry
        self.access_count = 0
        self.created_at = time.time()

    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.expiry is None:
            return False
        return time.time() > self.expiry

    def touch(self) -> None:
        """Increment access count."""
        self.access_count += 1


class MemoryCache:
    """
    Thread-safe in-memory cache with LRU eviction and TTL support.

    Features:
    - O(1) get/set/delete operations via OrderedDict
    - Automatic TTL-based expiration
    - LRU eviction when size limit reached
    - Thread-safe with fine-grained locking
    - Access tracking for analytics
    """

    def __init__(self, max_size: int = 1000, default_ttl: int | None = None) -> None:
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries to store.
            default_ttl: Default TTL in seconds for entries without explicit TTL.
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                self._delete_entry(key)
                self._expirations += 1
                self._misses += 1
                return None

            # Update LRU order - move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1

            return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds. None means use default TTL.
        """
        with self._lock:
            # Calculate expiry
            effective_ttl = ttl if ttl is not None else self._default_ttl
            expiry = time.time() + effective_ttl if effective_ttl is not None else None

            # If key exists, update it and move to end
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = CacheEntry(value, expiry)
                return

            # Check capacity and evict if needed
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            # Add new entry
            self._cache[key] = CacheEntry(value, expiry)

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            return self._delete_entry(key)

    def _delete_entry(self, key: str) -> bool:
        """Internal delete without locking."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def clear_prefix(self, prefix: str) -> int:
        """
        Clear all keys with a given prefix.

        Args:
            prefix: Key prefix to clear.

        Returns:
            Number of keys cleared.
        """
        with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_delete:
                self._delete_entry(key)
            return len(keys_to_delete)

    def clear_expired(self) -> int:
        """
        Clear all expired entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            expired_keys = [k for k, entry in self._cache.items() if entry.is_expired()]
            for key in expired_keys:
                self._delete_entry(key)
            self._expirations += len(expired_keys)
            return len(expired_keys)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # OrderedDict maintains insertion order
        # First item is oldest (LRU)
        oldest_key = next(iter(self._cache))
        self._delete_entry(oldest_key)
        self._evictions += 1

    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: int | None = None) -> Any:
        """
        Get value from cache or compute and store if not present.

        Args:
            key: Cache key.
            factory: Function to compute value if not in cache.
            ttl: Time-to-live in seconds.

        Returns:
            Cached or newly computed value.
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        value = factory()
        self.set(key, value, ttl)
        return value

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._delete_entry(key)
                return False
            return True

    def ttl(self, key: str) -> float | None:
        """
        Get remaining TTL for key in seconds.

        Returns:
            Remaining TTL in seconds, None if no TTL, or -1 if not found/expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired():
                return -1
            if entry.expiry is None:
                return None
            remaining = entry.expiry - time.time()
            return max(0, remaining)

    def stats(self) -> dict[str, Any]:
        """
        Get detailed cache statistics.

        Returns:
            Dictionary with cache metrics.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            # Count expired entries
            expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "utilization": len(self._cache) / self._max_size if self._max_size > 0 else 0,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "hit_rate_percent": f"{hit_rate:.2%}",
                "evictions": self._evictions,
                "expirations": self._expirations,
                "expired_entries": expired_count,
                "total_requests": total,
            }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0
