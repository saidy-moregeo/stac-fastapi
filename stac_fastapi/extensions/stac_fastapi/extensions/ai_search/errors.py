"""Error types and exception mapping for the AI Search extension."""

import asyncio
import logging
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class AISearchError(Exception):
    """Base error for the AI Search extension."""

    status_code: int = 500
    code: str = "AI_SEARCH_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        """Create an error with an optional code/status override."""
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code

    def to_http(self) -> HTTPException:
        """Render as an HTTPException with a machine-readable body."""
        return HTTPException(
            status_code=self.status_code,
            detail={"code": self.code, "message": self.message},
        )


class PromptValidationError(AISearchError):
    """Invalid prompt input (400)."""

    status_code = 400
    code = "AI_SEARCH_PROMPT_INVALID"


class RateLimitedError(AISearchError):
    """AI search rate limit exceeded (429)."""

    status_code = 429
    code = "AI_SEARCH_RATE_LIMITED"


class TranslationFailedError(AISearchError):
    """AI provider failed to translate the prompt (502)."""

    status_code = 502
    code = "AI_SEARCH_TRANSLATION_FAILED"


class ProviderConfigError(AISearchError):
    """AI provider configuration missing or invalid (503)."""

    status_code = 503
    code = "AI_SEARCH_PROVIDER_CONFIG_INVALID"


_AUTH_HTTP_STATUSES = {401, 403}


def map_translation_exception(exc: Exception) -> AISearchError:
    """Map a translator exception to an AISearchError.

    Uses exception class names (not imports) so mapping works whether or not
    the optional `pydantic-ai` dependency is installed. Provider internals are
    logged but never leaked into responses.
    """
    if isinstance(exc, AISearchError):
        return exc

    logger.exception("AI translation failed: %s", exc)

    names = {cls.__name__ for cls in type(exc).__mro__}

    if isinstance(exc, ImportError):
        return ProviderConfigError(
            "AI translation dependencies are not installed "
            "(pip install 'stac-fastapi.extensions[ai-search]').",
            code="AI_SEARCH_NOT_INSTALLED",
        )

    if isinstance(exc, asyncio.TimeoutError):
        return TranslationFailedError(
            "AI translation timed out.", code="AI_SEARCH_TRANSLATION_TIMEOUT"
        )

    if "UsageLimitExceeded" in names:
        return RateLimitedError(
            "AI provider usage limit exceeded.",
            code="AI_SEARCH_PROVIDER_RATE_LIMITED",
        )

    if "ModelHTTPError" in names:
        status = getattr(exc, "status_code", None)
        if status == 429:
            return RateLimitedError(
                "AI provider rate limit exceeded.",
                code="AI_SEARCH_PROVIDER_RATE_LIMITED",
            )
        if status in _AUTH_HTTP_STATUSES:
            return ProviderConfigError("AI provider rejected the configured credentials.")
        return TranslationFailedError("AI provider request failed.")

    if "UnexpectedModelBehavior" in names:
        return TranslationFailedError("AI provider returned an unusable translation.")

    return TranslationFailedError("AI translation failed.")
