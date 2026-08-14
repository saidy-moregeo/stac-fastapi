"""AI Search extension module.

Importable without the optional `ai-search` extra; the pydantic-ai-backed
translator is loaded lazily at request time.
"""

from .ai_search import (
    AISearchConformanceClasses,
    AISearchExtension,
    AISearchPromptExtension,
)
from .errors import (
    AISearchError,
    PromptValidationError,
    ProviderConfigError,
    RateLimitedError,
    TranslationFailedError,
)
from .request import (
    AI_SEARCH_DEFAULT_MAX_PROMPT_LENGTH,
    AISearchGetRequest,
    AISearchPromptGetRequest,
    AISearchPromptPostRequest,
    sanitize_prompt,
    validate_prompt,
)
from .settings import AISearchSettings
from .translator import Translator, validate_translation
from .types import (
    AISearchTarget,
    CollectionCandidate,
    PropertyFilter,
    SearchCapabilities,
    TranslationRequest,
    TranslationResult,
)

__all__ = [
    "AI_SEARCH_DEFAULT_MAX_PROMPT_LENGTH",
    "AISearchConformanceClasses",
    "AISearchError",
    "AISearchExtension",
    "AISearchGetRequest",
    "AISearchPromptExtension",
    "AISearchPromptGetRequest",
    "AISearchPromptPostRequest",
    "AISearchSettings",
    "AISearchTarget",
    "CollectionCandidate",
    "PropertyFilter",
    "PromptValidationError",
    "ProviderConfigError",
    "RateLimitedError",
    "SearchCapabilities",
    "TranslationFailedError",
    "TranslationRequest",
    "TranslationResult",
    "Translator",
    "sanitize_prompt",
    "validate_prompt",
    "validate_translation",
]
