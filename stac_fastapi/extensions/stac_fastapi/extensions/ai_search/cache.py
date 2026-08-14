"""Small asyncio-friendly TTL/LRU cache used by the AI Search extension."""

import asyncio
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, Hashable, Optional, Tuple


class TTLCache:
    """A bounded TTL cache with single-flight semantics.

    Values are best-effort: correctness never depends on a hit. Failed
    factories are never cached.
    """

    def __init__(self, maxsize: int = 1024, ttl: float = 300.0) -> None:
        """Create a cache holding at most `maxsize` entries for `ttl` seconds."""
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: "OrderedDict[Hashable, Tuple[float, Any]]" = OrderedDict()
        self._inflight: Dict[Hashable, asyncio.Future] = {}

    def get(self, key: Hashable) -> Optional[Any]:
        """Return a cached value, or None when absent/expired."""
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: Hashable, value: Any) -> None:
        """Store a value under `key`."""
        self._data[key] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    async def get_or_create(
        self, key: Hashable, factory: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Return the cached value or build it once, deduplicating concurrent calls."""
        cached = self.get(key)
        if cached is not None:
            return cached

        inflight = self._inflight.get(key)
        if inflight is not None:
            return await asyncio.shield(inflight)

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            value = await factory()
        except BaseException as exc:
            future.set_exception(exc)
            # Consume the exception so unawaited futures don't warn.
            future.exception()
            raise
        else:
            future.set_result(value)
            self.set(key, value)
            return value
        finally:
            self._inflight.pop(key, None)
