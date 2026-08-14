"""Shared types for the AI Search extension."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class AISearchTarget(str, Enum):
    """AI search target.

    ``COMBINED`` is internal only: the `target_hint` request parameter accepts
    `items` or `collections`; its absence means combined.
    """

    ITEMS = "items"
    COLLECTIONS = "collections"
    COMBINED = "combined"


class EndpointKind(str, Enum):
    """Kind of endpoint a prompt is being handled for."""

    ITEM_SEARCH = "item_search"
    COLLECTION_SEARCH = "collection_search"
    FEATURES_ITEMS = "features_items"
    AI_SEARCH = "ai_search"


@dataclass(frozen=True)
class SearchCapabilities:
    """Search capabilities derived from the server's own conformance classes.

    Used to keep AI-derived parameters within what the deployment can execute,
    mirroring the /conformance-based degradation a federating client would do,
    but without any HTTP round-trip.
    """

    item_search: bool = True
    collection_search: bool = False
    filter: bool = False
    item_free_text: bool = False
    collection_free_text: bool = False
    sort: bool = False
    query: bool = False
    #: basic free-text (`q` is a list) vs advanced free-text (`q` is a string)
    free_text_is_list: bool = True

    @classmethod
    def from_conformance(cls, classes: List[str]) -> "SearchCapabilities":
        """Derive capabilities from conformance class URIs."""

        def _any(*fragments: str) -> bool:
            return any(f in c for c in classes for f in fragments)

        return cls(
            item_search=_any("/item-search"),
            collection_search=_any("collection-search"),
            filter=_any("cql2-json", "item-search#filter"),
            item_free_text=_any(
                "item-search#free-text", "item-search#advanced-free-text"
            ),
            collection_free_text=_any(
                "collection-search#free-text", "collection-search#advanced-free-text"
            ),
            sort=_any("#sort"),
            query=_any("#query"),
            free_text_is_list=not _any("#advanced-free-text")
            or _any("item-search#free-text"),
        )

    def fingerprint(self) -> str:
        """Stable identifier of this capability set, for cache keys."""
        return "".join(
            "1" if v else "0"
            for v in (
                self.item_search,
                self.collection_search,
                self.filter,
                self.item_free_text,
                self.collection_free_text,
                self.sort,
                self.query,
                self.free_text_is_list,
            )
        )


class SortField(BaseModel):
    """A single sort clause."""

    field: str
    direction: Literal["asc", "desc"] = "asc"


class CollectionCandidate(BaseModel):
    """A collection id/title pair used to ground collection resolution."""

    id: str
    title: Optional[str] = None


FilterOp = Literal["=", "!=", "<", "<=", ">", ">="]


class PropertyFilter(BaseModel):
    """A single scalar property comparison (Basic CQL2)."""

    property: str = Field(description="STAC property name, e.g. 'eo:cloud_cover'.")
    op: FilterOp = Field(description="Comparison operator.")
    value: Union[bool, int, float, str] = Field(description="Scalar comparison value.")

    def to_cql2(self) -> Dict[str, Any]:
        """Render as a CQL2-JSON comparison expression."""
        return {"op": self.op, "args": [{"property": self.property}, self.value]}


class TranslationResult(BaseModel):
    """Structured output of a prompt translation.

    `platform_keyword` is internal only: it identifies which dataset the user
    means and is resolved to collection ids, never sent as a search term.
    """

    bbox: Optional[List[float]] = Field(
        None, description="Bounding box for spatial search (minx, miny, maxx, maxy)."
    )
    intersects: Optional[Dict[str, Any]] = Field(
        None,
        description="GeoJSON geometry for spatial search (only when bbox is not set).",
    )
    collections: Optional[List[str]] = Field(
        None, description="Exact STAC collection ids to restrict the search to."
    )
    ids: Optional[List[str]] = Field(None, description="Exact STAC item ids to return.")
    datetime: Optional[str] = Field(
        None, description="Temporal constraint (RFC 3339 datetime or interval)."
    )
    property_filters: Optional[List[PropertyFilter]] = Field(
        None,
        description=(
            "Scalar property comparisons (e.g. cloud cover, gsd, orbit state), "
            "AND-combined into a CQL2 filter."
        ),
    )
    platform_keyword: Optional[str] = Field(
        None,
        description=(
            "Sensor/platform/mission keyword (e.g. 'Sentinel-2') used internally "
            "to resolve provider-specific collection ids."
        ),
    )
    free_text: Optional[str] = Field(
        None,
        description=(
            "Free-text term for descriptive/semantic content unrelated to "
            "sensor or platform identification."
        ),
    )
    sortby: Optional[List[SortField]] = Field(
        None, description="Sort clauses for the results."
    )
    limit: Optional[int] = Field(None, description="Maximum number of results to return.")


@dataclass
class TranslationRequest:
    """Input to a `Translator`."""

    prompt: str
    target: AISearchTarget = AISearchTarget.COMBINED
    #: set when the search is already scoped to one collection (features endpoint)
    collection_id: Optional[str] = None
    #: collection candidates to ground collection-id resolution, when small enough
    candidates: Optional[List[CollectionCandidate]] = None
    #: queryable property names of the deployment, to ground property filters
    queryables: Optional[List[str]] = None
    capabilities: SearchCapabilities = field(default_factory=SearchCapabilities)


def property_filters_to_cql2(
    filters: List[PropertyFilter],
) -> Optional[Dict[str, Any]]:
    """AND-combine property filters into one CQL2-JSON expression."""
    if not filters:
        return None
    expressions = [f.to_cql2() for f in filters]
    if len(expressions) == 1:
        return expressions[0]
    return {"op": "and", "args": expressions}


_FREE_TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def free_text_terms(text: str) -> List[str]:
    """Tokenize LLM-derived free text into single-word search terms.

    Basic free-text `q` terms are OR-combined by the spec, and backend
    full-text parsers (e.g. pgstac's tsquery rewriting) choke on spaces and
    punctuation inside a term — so only plain word tokens survive (hyphens
    kept: `Sentinel-2`). May return an empty list, in which case `q` should
    be omitted entirely.
    """
    return _FREE_TEXT_TOKEN_RE.findall(text or "")


def _free_text_q(
    term: Optional[str], supported: bool, is_list: bool
) -> Optional[Union[List[str], str]]:
    """Project a free-text term onto `q`; None when unsupported or nothing survives."""
    if not term or not supported:
        return None
    if is_list:
        return free_text_terms(term) or None
    return term


def to_search_body(
    result: TranslationResult,
    capabilities: SearchCapabilities,
    collection_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Project a translation onto a POST `/search` body (wire keys).

    This canonical form is used both to execute item searches and as the
    `parameters` echo. Parameters the deployment cannot execute (per
    `capabilities`) and parameters that make no sense in context (e.g.
    `collections` when the search is already scoped to one collection) are
    omitted.
    """
    body: Dict[str, Any] = {}

    if result.bbox is not None:
        body["bbox"] = result.bbox
    elif result.intersects is not None:
        body["intersects"] = result.intersects

    if collection_id is None and result.collections:
        body["collections"] = result.collections
    if collection_id is None and result.ids:
        body["ids"] = result.ids

    if result.datetime is not None:
        body["datetime"] = result.datetime
    if result.limit is not None:
        body["limit"] = result.limit

    if result.sortby and capabilities.sort:
        body["sortby"] = [s.model_dump() for s in result.sortby]

    q = _free_text_q(
        result.free_text, capabilities.item_free_text, capabilities.free_text_is_list
    )
    if q is not None:
        body["q"] = q

    if result.property_filters and capabilities.filter:
        body["filter"] = property_filters_to_cql2(result.property_filters)
        body["filter-lang"] = "cql2-json"

    return body


def to_collection_search_params(
    result: TranslationResult,
    capabilities: SearchCapabilities,
) -> Dict[str, Any]:
    """Project a translation onto collection-search parameters (wire keys).

    Item-specific parameters (cloud cover, sort, ids) are intentionally left
    out; `q` prefers the platform keyword since that is what identifies a
    collection.
    """
    params: Dict[str, Any] = {}

    if result.bbox is not None:
        params["bbox"] = result.bbox
    if result.datetime is not None:
        params["datetime"] = result.datetime
    if result.limit is not None:
        params["limit"] = result.limit

    q = _free_text_q(
        result.platform_keyword or result.free_text,
        capabilities.collection_free_text,
        capabilities.free_text_is_list,
    )
    if q is not None:
        params["q"] = q

    return params


def dumps_compact(value: Any) -> str:
    """Serialize a JSON value compactly (for GET string parameters)."""
    return json.dumps(value, separators=(",", ":"))
