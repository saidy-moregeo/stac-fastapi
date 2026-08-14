"""Request models and prompt validation for the AI Search extension."""

import html
import re
from typing import Literal, Optional

import attr
from fastapi import Query
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from stac_fastapi.types.search import APIRequest

from .errors import PromptValidationError

AI_SEARCH_DEFAULT_MAX_PROMPT_LENGTH = 500

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")

_PROMPT_DESCRIPTION = (
    "Natural-language search intent to translate into STAC query parameters."
)


def sanitize_prompt(prompt: str) -> str:
    """Sanitize a natural language prompt (control chars, tags, whitespace)."""
    clean = html.unescape(prompt or "")
    clean = _CONTROL_CHARS_RE.sub(" ", clean)
    clean = _HTML_TAG_RE.sub(" ", clean)
    clean = _WHITESPACE_RE.sub(" ", clean).strip()
    return clean


def validate_prompt(prompt: str, max_length: int) -> str:
    """Sanitize and validate a prompt, raising 400-level errors on failure."""
    clean = sanitize_prompt(prompt)
    if not clean:
        raise PromptValidationError(
            "Prompt is empty after sanitization.", code="AI_SEARCH_PROMPT_EMPTY"
        )
    if len(clean) > max_length:
        raise PromptValidationError(
            f"Prompt exceeds maximum length of {max_length} characters.",
            code="AI_SEARCH_PROMPT_TOO_LONG",
        )
    return clean


@attr.s
class AISearchGetRequest(APIRequest):
    """GET /ai-search request (query parameters)."""

    prompt: Annotated[
        str,
        Query(
            min_length=1,
            max_length=AI_SEARCH_DEFAULT_MAX_PROMPT_LENGTH,
            description=_PROMPT_DESCRIPTION,
        ),
    ] = attr.ib()
    target_hint: Annotated[
        Optional[Literal["items", "collections"]],
        Query(
            description=(
                "Optional target hint. When omitted, items and collections are "
                "searched and combined."
            ),
        ),
    ] = attr.ib(default=None)


@attr.s
class AISearchPromptGetRequest(APIRequest):
    """`prompt` fragment composed into existing GET endpoints."""

    prompt: Annotated[
        Optional[str],
        Query(
            min_length=1,
            max_length=AI_SEARCH_DEFAULT_MAX_PROMPT_LENGTH,
            description=_PROMPT_DESCRIPTION,
        ),
    ] = attr.ib(default=None)


class AISearchPromptPostRequest(BaseModel):
    """`prompt` fragment composed into existing POST endpoints."""

    prompt: Optional[str] = Field(
        None,
        min_length=1,
        max_length=AI_SEARCH_DEFAULT_MAX_PROMPT_LENGTH,
        description=_PROMPT_DESCRIPTION,
    )
