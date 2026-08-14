"""Dedicated AI-search rate limiter built on `limits`.

Independent of any slowapi limiter the application may have: it fires only on
AI usage (the /ai-search endpoint and prompt-carrying requests to wrapped
routes), so normal traffic on shared routes is never throttled by it, and no
`app.state.limiter` setup-ordering applies.
"""

from typing import Callable, Optional

from starlette.requests import Request

from .errors import ProviderConfigError, RateLimitedError


def _remote_address(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class AISearchRateLimiter:
    """Rate limiter applied to AI usage only. A `rate` of None disables it."""

    def __init__(
        self,
        rate: Optional[str] = None,
        storage_uri: Optional[str] = None,
        key_func: Callable[[Request], str] = _remote_address,
    ) -> None:
        """Create a limiter for `rate` (e.g. "10/minute") on `storage_uri`."""
        self.rate = rate
        self.storage_uri = storage_uri
        self.key_func = key_func
        self._limiter = None
        self._item = None

    def _setup(self) -> None:
        try:
            from limits import parse
            from limits.aio.strategies import MovingWindowRateLimiter
            from limits.storage import storage_from_string
        except ImportError as exc:
            raise ProviderConfigError(
                "An AI search rate limit is configured but the `limits` package "
                "is not installed (pip install 'stac-fastapi.extensions[ai-search]').",
            ) from exc

        uri = self.storage_uri or "async+memory://"
        # The async `limits` storages use "async+"-prefixed schemes.
        if not uri.startswith("async+"):
            uri = f"async+{uri}"
        self._item = parse(self.rate)
        self._limiter = MovingWindowRateLimiter(storage_from_string(uri))

    async def check(self, request: Request) -> None:
        """Consume one hit; raise RateLimitedError (429) when over the limit."""
        if not self.rate:
            return
        if self._limiter is None:
            self._setup()
        if not await self._limiter.hit(self._item, self.key_func(request)):
            raise RateLimitedError("AI search rate limit exceeded.")
