"""Settings for the AI Search extension.

One convention for every backend: all options bind to environment variables
prefixed with ``STAC_FASTAPI_AI_SEARCH_`` (e.g. ``STAC_FASTAPI_AI_SEARCH_MODEL``),
read from the process environment or a ``.env`` file in the working directory
(the same convention as ``ApiSettings``). Real environment variables take
precedence over ``.env`` values.
"""

from typing import Literal, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .request import AI_SEARCH_DEFAULT_MAX_PROMPT_LENGTH


class AISearchSettings(BaseSettings):
    """AI Search extension settings."""

    model_config = SettingsConfigDict(
        env_prefix="STAC_FASTAPI_AI_SEARCH_", env_file=".env", extra="ignore"
    )

    #: master switch, read by backend applications
    enabled: bool = False

    #: pydantic-ai style model identifier, e.g. "openai:gpt-4o-mini"
    model: Optional[str] = None
    #: provider API key; applied server-side only, never accepted from requests
    api_key: Optional[SecretStr] = None
    #: custom endpoint for OpenAI-compatible/self-hosted gateways
    base_url: Optional[str] = None

    max_prompt_length: int = AI_SEARCH_DEFAULT_MAX_PROMPT_LENGTH
    #: budget for a single translation call, seconds
    request_timeout: float = 30.0

    #: include the AI-derived `parameters` echo in responses
    echo_parameters: bool = True
    #: add prompt handling to /search, /collections and .../items endpoints
    enable_prompt_on_endpoints: bool = True
    #: how an AI-derived filter combines with a user-supplied one:
    #: "user-wins" drops the AI filter; "and" composes when the user filter is
    #: already cql2-json
    filter_merge_strategy: Literal["user-wins", "and"] = "user-wins"

    #: ground platform keywords against the deployment's own collections
    resolve_collections: bool = True
    #: candidate count up to which collections are resolved in the translation call
    inline_candidates_max: int = 200
    catalog_ttl: float = 600.0
    catalog_max_candidates: int = 1000

    translation_cache_ttl: float = 300.0
    translation_cache_size: int = 1024

    #: dedicated AI rate limit, e.g. "10/minute"; None disables it
    rate_limit: Optional[str] = None
    #: `limits` storage URI, e.g. "async+redis://localhost:6379/0"; None = in-memory
    rate_limit_storage_uri: Optional[str] = None
