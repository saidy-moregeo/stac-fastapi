"""Merge AI-derived parameters into user requests under user precedence.

All decisions are made in "wire space" (the parameter names clients actually
send: `filter`, `filter-lang`, ...) because that is the only representation in
which "did the user supply this?" can be answered exactly:

- GET: presence in `request.query_params` (parsed attrs instances can't tell a
  user-sent `limit=10` from the default).
- POST: pydantic's `model_fields_set`, mapped through field aliases.
"""

import json
import logging
from typing import Any, Dict, Optional, Set, Tuple

from pydantic import BaseModel
from starlette.requests import Request

from .types import dumps_compact

logger = logging.getLogger(__name__)

FilterMergeStrategy = str  # "user-wins" | "and"

#: wire name -> attrs/pydantic field name
WIRE_TO_FIELD = {
    "filter": "filter_expr",
    "filter-lang": "filter_lang",
    "filter-crs": "filter_crs",
}

_SPATIAL_KEYS = ("bbox", "intersects")
_FILTER_KEYS = ("filter", "filter-lang")


def get_user_wire_keys(request: Request) -> Set[str]:
    """Wire names of the query parameters the user actually sent."""
    return set(request.query_params.keys()) - {"prompt"}


def post_user_wire_keys(model: BaseModel) -> Set[str]:
    """Wire names of the body members the user actually sent."""
    fields = type(model).model_fields
    keys = set()
    for name in model.model_fields_set:
        if name == "prompt":
            continue
        field = fields.get(name)
        keys.add(field.alias if field and field.alias else name)
    return keys


def apply_precedence(ai_body: Dict[str, Any], user_wire_keys: Set[str]) -> Dict[str, Any]:
    """Drop AI-derived parameters the user's request overrides.

    Key-by-key user precedence, plus two family rules: `bbox`/`intersects`
    form one spatial family, and `filter`/`filter-lang` form one filter
    family — a user-supplied member of a family suppresses the whole
    AI-derived family.
    """
    body = {k: v for k, v in ai_body.items() if k not in user_wire_keys}

    if any(k in user_wire_keys for k in _SPATIAL_KEYS):
        for key in _SPATIAL_KEYS:
            body.pop(key, None)
    elif all(k in body for k in _SPATIAL_KEYS):
        body.pop("intersects")

    if any(k in user_wire_keys for k in _FILTER_KEYS):
        for key in _FILTER_KEYS:
            body.pop(key, None)

    return body


def convert_for_get(wire_key: str, value: Any) -> Any:
    """Convert a canonical (POST-shaped) value to its parsed-GET representation.

    Values must arrive post-conversion because attrs converters are not
    idempotent (and would not run on `setattr` anyway).
    """
    if wire_key == "bbox":
        return tuple(float(v) for v in value)
    if wire_key in ("intersects", "filter"):
        return dumps_compact(value)
    if wire_key == "sortby":
        return [("-" if s.get("direction") == "desc" else "") + s["field"] for s in value]
    return value


def apply_to_get_request(
    request_data: Any,
    ai_body: Dict[str, Any],
    user_wire_keys: Set[str],
    *,
    filter_merge_strategy: FilterMergeStrategy = "user-wins",
) -> Dict[str, Any]:
    """Apply AI-derived parameters to a parsed GET request, in place.

    A parameter is applied only when the composed request model has the
    corresponding field — the model is the capability declaration. Returns
    the applied parameters (canonical wire shape) for the `parameters` echo.
    """
    body = apply_precedence(ai_body, user_wire_keys)
    applied: Dict[str, Any] = {}

    for wire_key, value in body.items():
        field = WIRE_TO_FIELD.get(wire_key, wire_key)
        if not hasattr(request_data, field):
            continue
        setattr(request_data, field, convert_for_get(wire_key, value))
        applied[wire_key] = value

    if filter_merge_strategy == "and" and "filter" in ai_body and "filter" not in body:
        composed = _compose_get_filter(request_data, ai_body["filter"], user_wire_keys)
        if composed is not None:
            applied["filter"] = ai_body["filter"]

    # An applied filter without its language is ambiguous; drop the orphan.
    if "filter" not in applied:
        applied.pop("filter-lang", None)

    if hasattr(request_data, "prompt"):
        request_data.prompt = None

    return applied


def _compose_get_filter(
    request_data: Any,
    ai_filter: Dict[str, Any],
    user_wire_keys: Set[str],
) -> Optional[Dict[str, Any]]:
    """AND-compose the AI filter into a user-supplied cql2-json GET filter."""
    if "filter" not in user_wire_keys:
        return None
    if getattr(request_data, "filter_lang", None) != "cql2-json":
        return None
    raw = getattr(request_data, "filter_expr", None)
    if not raw:
        return None
    try:
        user_filter = json.loads(raw)
    except (TypeError, ValueError):
        return None
    composed = {"op": "and", "args": [user_filter, ai_filter]}
    request_data.filter_expr = dumps_compact(composed)
    return composed


def apply_to_post_request(
    search_request: BaseModel,
    ai_body: Dict[str, Any],
    user_wire_keys: Set[str],
    *,
    filter_merge_strategy: FilterMergeStrategy = "user-wins",
) -> Tuple[BaseModel, Dict[str, Any]]:
    """Merge AI-derived parameters into a POST body model.

    Rebuilds the request through full model validation, so aliases and
    validators behave exactly as for a genuinely-sent body. Fail-soft: if the
    merged body does not validate, the user's original request is used
    unchanged. Returns `(request model, applied parameters)`.
    """
    model_cls = type(search_request)
    known_wire = {field.alias or name for name, field in model_cls.model_fields.items()}

    body = apply_precedence(ai_body, user_wire_keys)
    body = {k: v for k, v in body.items() if k in known_wire}
    if "filter" not in body:
        body.pop("filter-lang", None)

    user_wire = search_request.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude={"prompt"}
    )
    merged = {**body, **user_wire}
    applied = dict(body)

    if (
        filter_merge_strategy == "and"
        and "filter" in ai_body
        and "filter" not in body
        and isinstance(user_wire.get("filter"), dict)
    ):
        merged["filter"] = {
            "op": "and",
            "args": [user_wire["filter"], ai_body["filter"]],
        }
        applied["filter"] = ai_body["filter"]

    try:
        return model_cls.model_validate(merged), applied
    except Exception:
        logger.warning(
            "Merged AI parameters failed validation; using the user request unchanged",
            exc_info=True,
        )
        return search_request.model_copy(update={"prompt": None}), {}
