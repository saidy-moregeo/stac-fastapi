"""Translator protocol and fail-soft validation of LLM-derived parameters."""

import logging
import math
import re
from typing import List, Optional, Protocol, Set, runtime_checkable

from stac_fastapi.types.rfc3339 import str_to_interval

from .types import (
    CollectionCandidate,
    PropertyFilter,
    TranslationRequest,
    TranslationResult,
)

logger = logging.getLogger(__name__)

_SORT_FIELD_RE = re.compile(r"^[A-Za-z0-9_:.\-]{1,80}$")
_PROPERTY_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_.\-]{0,80}$")

MAX_LIMIT = 10_000

#: Well-known scalar STAC extension properties users commonly filter on.
#: A deployment's /queryables (when available) is consulted in addition.
KNOWN_FILTER_PROPERTIES = frozenset(
    {
        "eo:cloud_cover",
        "eo:snow_cover",
        "gsd",
        "platform",
        "constellation",
        "mission",
        "sat:orbit_state",
        "sat:relative_orbit",
        "sar:instrument_mode",
        "sar:frequency_band",
        "view:off_nadir",
        "view:sun_elevation",
        "view:sun_azimuth",
        "view:incidence_angle",
        "proj:epsg",
        "created",
        "updated",
    }
)


@runtime_checkable
class Translator(Protocol):
    """Turns a natural-language prompt into structured STAC search parameters.

    Implementations may use any LLM backend; the shipped default is
    `PydanticAiTranslator`. Implementations should raise on failure and let
    the extension map exceptions to HTTP semantics.
    """

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate a prompt into search parameters."""
        ...

    async def select_collections(
        self, prompt: str, candidates: List[CollectionCandidate]
    ) -> List[str]:
        """Optionally resolve a prompt to collection ids from candidates.

        Must be fail-soft: return [] rather than raise.
        """
        ...


def _valid_bbox(bbox: List[float]) -> bool:
    if len(bbox) not in (4, 6):
        return False
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
        return False
    if len(bbox) == 4:
        minx, miny, maxx, maxy = bbox
    else:
        minx, miny, _, maxx, maxy, _ = bbox
    return (
        -180 <= minx <= 180
        and -180 <= maxx <= 180
        and -90 <= miny <= 90
        and -90 <= maxy <= 90
        and miny <= maxy
    )


def _valid_geometry(geom: dict) -> bool:
    return isinstance(geom, dict) and "type" in geom and "coordinates" in geom


def _validate_spatiotemporal(result: TranslationResult, update: dict) -> None:
    if result.bbox is not None and not _valid_bbox(result.bbox):
        logger.warning("Dropping invalid AI-derived bbox: %s", result.bbox)
        update["bbox"] = None

    if result.intersects is not None and not _valid_geometry(result.intersects):
        logger.warning("Dropping invalid AI-derived geometry")
        update["intersects"] = None

    if result.datetime is not None:
        try:
            str_to_interval(result.datetime)
        except Exception:
            logger.warning("Dropping invalid AI-derived datetime: %s", result.datetime)
            update["datetime"] = None


def _validate_grounded_fields(
    result: TranslationResult,
    candidates: Optional[List[CollectionCandidate]],
    update: dict,
) -> None:
    if result.sortby is not None:
        valid_sorts = [s for s in result.sortby if _SORT_FIELD_RE.match(s.field)]
        if len(valid_sorts) != len(result.sortby):
            logger.warning("Dropping AI-derived sort clauses with invalid fields")
        update["sortby"] = valid_sorts or None

    if result.collections is not None and candidates is not None:
        known = {c.id for c in candidates}
        valid_ids = [c for c in result.collections if c in known]
        if len(valid_ids) != len(result.collections):
            logger.warning(
                "Dropping AI-derived collection ids not present in the catalog: %s",
                sorted(set(result.collections) - known),
            )
        update["collections"] = valid_ids or None


def _valid_property_filter(pf: PropertyFilter, allowed_properties: Set[str]) -> bool:
    if not _PROPERTY_NAME_RE.match(pf.property):
        logger.warning("Dropping AI-derived filter with invalid property name: %r", pf)
        return False
    if pf.property not in allowed_properties:
        logger.warning("Dropping AI-derived filter on unknown property: %s", pf.property)
        return False
    if not isinstance(pf.value, (bool, int, float, str)):
        logger.warning("Dropping AI-derived filter with non-scalar value: %r", pf)
        return False
    return True


def _validate_property_filters(
    result: TranslationResult,
    queryables: Optional[Set[str]],
    update: dict,
) -> None:
    if result.property_filters is None:
        return
    allowed = KNOWN_FILTER_PROPERTIES | (queryables or set())
    valid = [f for f in result.property_filters if _valid_property_filter(f, allowed)]
    update["property_filters"] = valid or None


def validate_translation(
    result: TranslationResult,
    candidates: Optional[List[CollectionCandidate]] = None,
    queryables: Optional[Set[str]] = None,
) -> TranslationResult:
    """Drop (and log) invalid LLM-derived fields instead of failing the request."""
    update: dict = {}

    _validate_spatiotemporal(result, update)
    _validate_property_filters(result, queryables, update)

    if result.limit is not None:
        update["limit"] = min(max(int(result.limit), 1), MAX_LIMIT)

    _validate_grounded_fields(result, candidates, update)

    return result.model_copy(update=update) if update else result
