"""Advanced LRU Cache with TTL and metrics"""

import time
from collections import OrderedDict
from typing import Any


class LRUCache:
    """High-performance LRU cache with TTL support and comprehensive metrics"""

    def __init__(self, maxsize: int = 5000, ttl: float | None = 3600):
        self.maxsize = int(maxsize)
        self.ttl = ttl  # Time to live in seconds
        self._store = OrderedDict()
        self._timestamps = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with TTL check"""
        # Check TTL first
        if self.ttl and key in self._timestamps:
            if time.time() - self._timestamps[key] > self.ttl:
                # Expired - remove it
                self._store.pop(key, None)
                self._timestamps.pop(key, None)
                self._misses += 1
                return default

        if key in self._store:
            # Move to end (most recently used)
            val = self._store.pop(key)
            self._store[key] = val
            self._hits += 1
            return val

        self._misses += 1
        return default

    def set(self, key: str, value: Any) -> None:
        """Set value with timestamp"""
        if key in self._store:
            # Update existing
            self._store.pop(key)
        elif len(self._store) >= self.maxsize:
            # Evict oldest
            oldest = next(iter(self._store))
            self._store.pop(oldest)
            self._timestamps.pop(oldest, None)
            self._evictions += 1

        self._store[key] = value
        if self.ttl:
            self._timestamps[key] = time.time()

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        # Check TTL and remove if expired
        if self.ttl and key in self._timestamps:
            if time.time() - self._timestamps[key] > self.ttl:
                self._store.pop(key, None)
                self._timestamps.pop(key, None)
                return False
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        """Clear cache and reset metrics"""
        self._store.clear()
        self._timestamps.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics"""
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": (self._hits / total) if total else 0.0,
            "miss_rate": (self._misses / total) if total else 0.0,
        }

    def prune_expired(self) -> int:
        """Remove expired entries"""
        if not self.ttl:
            return 0

        now = time.time()
        expired = []
        for key, timestamp in self._timestamps.items():
            if now - timestamp > self.ttl:
                expired.append(key)

        for key in expired:
            self._store.pop(key, None)
            self._timestamps.pop(key, None)

        return len(expired)
