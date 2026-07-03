"""
Cache layer for repeated text analysis.

Avoids re-computing cohesion scores for texts that have already been
analyzed. Uses text hash as cache key with configurable TTL.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class CacheEntry:
    result: Any
    timestamp: float
    ttl: float


class AnalysisCache:
    """LRU-ish cache for TextAnalyzer results."""

    def __init__(self, max_size: int = 256, default_ttl: float = 300.0):
        self._cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        key = self._hash(text)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() - entry.timestamp > entry.ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry.result

    def set(self, text: str, result: Any, ttl: Optional[float] = None) -> None:
        key = self._hash(text)
        if len(self._cache) >= self.max_size:
            oldest = min(self._cache, key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest]
        self._cache[key] = CacheEntry(
            result=result,
            timestamp=time.time(),
            ttl=ttl if ttl is not None else self.default_ttl,
        )

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def invalidate(self, text: str) -> None:
        key = self._hash(text)
        self._cache.pop(key, None)

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 4),
        }

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return round(self._hits / max(total, 1), 4)
