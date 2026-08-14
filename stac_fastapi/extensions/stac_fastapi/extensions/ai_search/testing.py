"""Test helpers for the AI Search extension.

`StaticTranslator` implements the `Translator` protocol without any LLM
dependency, for extension, backend, and third-party test suites.
"""

from typing import Dict, List, Optional

from .types import CollectionCandidate, TranslationRequest, TranslationResult


class StaticTranslator:
    """A `Translator` returning a fixed result (or raising a fixed error)."""

    def __init__(
        self,
        result: Optional[TranslationResult] = None,
        selections: Optional[Dict[str, List[str]]] = None,
        raises: Optional[Exception] = None,
    ) -> None:
        """Return `result` from translate; raise `raises` instead when set.

        `selections` maps prompts to collection-id lists for
        `select_collections`.
        """
        self.result = result or TranslationResult()
        self.selections = selections or {}
        self.raises = raises
        self.calls: List[TranslationRequest] = []

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """Record the call and return the configured result."""
        self.calls.append(request)
        if self.raises is not None:
            raise self.raises
        return self.result

    async def select_collections(
        self, prompt: str, candidates: List[CollectionCandidate]
    ) -> List[str]:
        """Return the configured selection for `prompt`, filtered to candidates."""
        known = {c.id for c in candidates}
        return [c for c in self.selections.get(prompt, []) if c in known]
