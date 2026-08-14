"""TTL-cached collection catalog fetched through the backend's own client."""

import hashlib
import inspect
import logging
from typing import Any, Callable, List, Optional

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from .cache import TTLCache
from .types import CollectionCandidate

logger = logging.getLogger(__name__)


async def call_client_method(method: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call a core-client method, dispatching sync clients to the threadpool."""
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    return await run_in_threadpool(method, *args, **kwargs)


class CollectionCatalog:
    """Collection id/title candidates for grounding, from `client.all_collections`.

    Fail-soft: on error the last known candidates (or an empty list) are
    returned and translation proceeds ungrounded.
    """

    def __init__(
        self,
        client: Any,
        ttl: float = 600.0,
        max_candidates: int = 1000,
    ) -> None:
        """Create a catalog reading through `client` with the given cache TTL."""
        self.client = client
        self.max_candidates = max_candidates
        self._cache = TTLCache(maxsize=4, ttl=ttl)
        self._last_known: List[CollectionCandidate] = []
        self._version: Optional[str] = None

    async def get(self, request: Request) -> List[CollectionCandidate]:
        """Return up to `max_candidates` collection candidates."""
        try:
            candidates = await self._cache.get_or_create(
                "catalog", lambda: self._fetch(request)
            )
        except Exception:
            logger.warning(
                "Collection catalog fetch failed; grounding disabled for this request",
                exc_info=True,
            )
            return self._last_known
        self._last_known = candidates
        return candidates

    async def _fetch(self, request: Request) -> List[CollectionCandidate]:
        response = await call_client_method(
            self.client.all_collections,
            request=request,
            limit=self.max_candidates,
        )
        collections = (response or {}).get("collections", [])
        candidates = [
            CollectionCandidate(id=c["id"], title=c.get("title"))
            for c in collections[: self.max_candidates]
            if isinstance(c, dict) and c.get("id")
        ]
        digest = hashlib.sha256(
            "\n".join(sorted(c.id for c in candidates)).encode()
        ).hexdigest()
        self._version = digest[:16]
        return candidates

    def version(self) -> str:
        """Identifier of the current candidate set, for cache keys."""
        return self._version or "unfetched"
