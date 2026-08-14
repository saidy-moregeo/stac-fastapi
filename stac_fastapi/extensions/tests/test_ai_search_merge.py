"""Unit tests for AI-search merge, validation, projection and cache logic."""

import asyncio

import attr
from starlette.requests import Request

from stac_fastapi.api.models import create_post_request_model
from stac_fastapi.extensions import AISearchPromptExtension
from stac_fastapi.extensions.ai_search.cache import TTLCache
from stac_fastapi.extensions.ai_search.merge import (
    apply_precedence,
    apply_to_get_request,
    apply_to_post_request,
    convert_for_get,
    get_user_wire_keys,
    post_user_wire_keys,
)
from stac_fastapi.extensions.ai_search.translator import validate_translation
from stac_fastapi.extensions.ai_search.types import (
    CollectionCandidate,
    PropertyFilter,
    SearchCapabilities,
    SortField,
    TranslationResult,
    free_text_terms,
    to_collection_search_params,
    to_search_body,
)
from stac_fastapi.extensions.filter import FilterExtension
from stac_fastapi.extensions.free_text import FreeTextExtension
from stac_fastapi.extensions.sort import SortExtension
from stac_fastapi.types.search import APIRequest

FULL_CAPS = SearchCapabilities(
    item_search=True,
    collection_search=True,
    filter=True,
    item_free_text=True,
    collection_free_text=True,
    sort=True,
    query=True,
    free_text_is_list=True,
)

AI_BODY = {
    "bbox": [1.0, 2.0, 3.0, 4.0],
    "datetime": "2022-01-01T00:00:00Z/2022-12-31T23:59:59Z",
    "limit": 42,
    "q": ["flood damage"],
    "sortby": [{"field": "datetime", "direction": "desc"}],
    "filter": {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 10]},
    "filter-lang": "cql2-json",
}


@attr.s
class FakeGetRequest(APIRequest):
    prompt = attr.ib(default=None)
    bbox = attr.ib(default=None)
    intersects = attr.ib(default=None)
    datetime = attr.ib(default=None)
    limit = attr.ib(default=10)
    sortby = attr.ib(default=None)
    q = attr.ib(default=None)
    filter_expr = attr.ib(default=None)
    filter_lang = attr.ib(default="cql2-text")


def _request(query_string: str) -> Request:
    return Request({"type": "http", "query_string": query_string.encode(), "headers": []})


def test_user_wire_keys_from_query_string():
    keys = get_user_wire_keys(_request("prompt=x&bbox=1,2,3,4&filter-lang=cql2-json"))
    assert keys == {"bbox", "filter-lang"}


def test_apply_precedence_user_wins_and_families():
    # plain key precedence
    assert "limit" not in apply_precedence(AI_BODY, {"limit"})
    # user spatial suppresses the whole AI spatial family
    body = apply_precedence({"bbox": [1, 2, 3, 4], "intersects": {}}, {"intersects"})
    assert "bbox" not in body and "intersects" not in body
    # AI producing both keeps bbox only
    body = apply_precedence(
        {"bbox": [1, 2, 3, 4], "intersects": {"type": "Point"}}, set()
    )
    assert "bbox" in body and "intersects" not in body
    # user filter-lang suppresses the whole AI filter family
    body = apply_precedence(AI_BODY, {"filter-lang"})
    assert "filter" not in body and "filter-lang" not in body


def test_convert_for_get_representations():
    assert convert_for_get("bbox", [1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)
    assert convert_for_get("filter", {"op": "isNull"}) == '{"op":"isNull"}'
    assert convert_for_get(
        "sortby", [{"field": "datetime", "direction": "desc"}, {"field": "gsd"}]
    ) == ["-datetime", "gsd"]


def test_apply_to_get_request_merges_and_strips_prompt():
    data = FakeGetRequest(prompt="sentinel over germany")
    applied = apply_to_get_request(data, AI_BODY, set())

    assert data.prompt is None
    assert data.bbox == (1.0, 2.0, 3.0, 4.0)
    assert data.limit == 42
    assert data.sortby == ["-datetime"]
    assert data.q == ["flood damage"]
    assert data.filter_lang == "cql2-json"
    assert '"eo:cloud_cover"' in data.filter_expr
    assert applied == AI_BODY


def test_apply_to_get_request_user_precedence():
    data = FakeGetRequest(prompt="x", bbox=(9.0, 9.0, 10.0, 10.0), limit=10)
    applied = apply_to_get_request(data, AI_BODY, {"bbox", "limit"})

    assert data.bbox == (9.0, 9.0, 10.0, 10.0)
    assert data.limit == 10
    assert "bbox" not in applied and "limit" not in applied
    assert applied["datetime"] == AI_BODY["datetime"]


def test_apply_to_get_request_capability_by_model():
    @attr.s
    class MinimalGet(APIRequest):
        prompt = attr.ib(default=None)
        datetime = attr.ib(default=None)

    data = MinimalGet(prompt="x")
    applied = apply_to_get_request(data, AI_BODY, set())
    # only fields present on the model are applied or echoed
    assert applied == {"datetime": AI_BODY["datetime"]}


def test_apply_to_get_request_and_composition():
    user_filter = '{"op":"=","args":[{"property":"platform"},"s2"]}'
    data = FakeGetRequest(prompt="x", filter_expr=user_filter, filter_lang="cql2-json")
    applied = apply_to_get_request(
        data, AI_BODY, {"filter", "filter-lang"}, filter_merge_strategy="and"
    )
    assert '"op":"and"' in data.filter_expr
    assert applied["filter"] == AI_BODY["filter"]


def _post_model():
    return create_post_request_model(
        [
            FreeTextExtension(),
            SortExtension(),
            FilterExtension(),
            AISearchPromptExtension(),
        ]
    )


def test_apply_to_post_request_merges_and_strips_prompt():
    model_cls = _post_model()
    search_request = model_cls.model_validate({"prompt": "x", "limit": 5})

    merged, applied = apply_to_post_request(
        search_request, AI_BODY, post_user_wire_keys(search_request)
    )

    assert merged.prompt is None
    assert merged.limit == 5, "explicit user value must win over AI limit"
    assert "limit" not in applied
    assert merged.datetime == AI_BODY["datetime"]
    assert merged.filter_lang == "cql2-json"
    assert merged.filter_expr == AI_BODY["filter"]
    assert applied["filter"] == AI_BODY["filter"]


def test_post_user_default_value_counts_as_user_set():
    model_cls = _post_model()
    search_request = model_cls.model_validate({"prompt": "x", "limit": 10})
    keys = post_user_wire_keys(search_request)
    assert "limit" in keys, "explicitly sent default must count as user-set"

    merged, _ = apply_to_post_request(search_request, {"limit": 99}, keys)
    assert merged.limit == 10


def test_apply_to_post_request_filter_user_wins():
    model_cls = _post_model()
    user_filter = {"op": "=", "args": [{"property": "platform"}, "s2"]}
    search_request = model_cls.model_validate({"prompt": "x", "filter": user_filter})

    merged, applied = apply_to_post_request(
        search_request, AI_BODY, post_user_wire_keys(search_request)
    )
    assert merged.filter_expr == user_filter
    assert "filter" not in applied


def test_apply_to_post_request_filter_and_strategy():
    model_cls = _post_model()
    user_filter = {"op": "=", "args": [{"property": "platform"}, "s2"]}
    search_request = model_cls.model_validate({"prompt": "x", "filter": user_filter})

    merged, applied = apply_to_post_request(
        search_request,
        AI_BODY,
        post_user_wire_keys(search_request),
        filter_merge_strategy="and",
    )
    assert merged.filter_expr == {"op": "and", "args": [user_filter, AI_BODY["filter"]]}
    assert applied["filter"] == AI_BODY["filter"]


def test_apply_to_post_request_spatial_family():
    model_cls = _post_model()
    search_request = model_cls.model_validate({"prompt": "x", "bbox": [9, 9, 10, 10]})
    merged, applied = apply_to_post_request(
        search_request,
        {"intersects": {"type": "Point", "coordinates": [1, 2]}},
        post_user_wire_keys(search_request),
    )
    assert merged.intersects is None
    assert applied == {}


def test_validate_translation_drops_invalid_fields():
    result = TranslationResult(
        bbox=[999.0, 2.0, 3.0, 4.0],
        intersects={"bad": "geometry"},
        datetime="not-a-datetime",
        limit=999_999,
        sortby=[SortField(field="datetime"), SortField(field="bad field!")],
        collections=["known", "unknown"],
    )
    candidates = [CollectionCandidate(id="known")]
    validated = validate_translation(result, candidates)

    assert validated.bbox is None
    assert validated.intersects is None
    assert validated.datetime is None
    assert validated.limit == 10_000
    assert [s.field for s in validated.sortby] == ["datetime"]
    assert validated.collections == ["known"]


def test_validate_property_filters_grounding():
    result = TranslationResult(
        property_filters=[
            PropertyFilter(property="eo:cloud_cover", op="<=", value=10),
            PropertyFilter(property="s2:water_percentage", op="<", value=50),
            PropertyFilter(property="'; DROP TABLE items", op="=", value="x"),
        ]
    )

    # allowlist only: the deployment-specific property is dropped
    validated = validate_translation(result)
    assert [f.property for f in validated.property_filters] == ["eo:cloud_cover"]

    # a queryables set admits deployment-specific properties, but never
    # malformed names
    validated = validate_translation(
        result, queryables={"s2:water_percentage", "'; DROP TABLE items"}
    )
    assert [f.property for f in validated.property_filters] == [
        "eo:cloud_cover",
        "s2:water_percentage",
    ]

    # nothing valid -> None (so no filter key is projected at all)
    only_bad = TranslationResult(
        property_filters=[PropertyFilter(property="nope:nope", op="=", value=1)]
    )
    assert validate_translation(only_bad).property_filters is None


def test_property_filters_projection():
    single = TranslationResult(
        property_filters=[PropertyFilter(property="eo:cloud_cover", op="<=", value=10.0)]
    )
    body = to_search_body(single, FULL_CAPS)
    assert body["filter"] == {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 10.0]}
    assert body["filter-lang"] == "cql2-json"

    ranged = TranslationResult(
        property_filters=[
            PropertyFilter(property="eo:cloud_cover", op=">=", value=10),
            PropertyFilter(property="eo:cloud_cover", op="<=", value=20),
            PropertyFilter(property="sat:orbit_state", op="=", value="ascending"),
        ]
    )
    body = to_search_body(ranged, FULL_CAPS)
    assert body["filter"]["op"] == "and"
    assert len(body["filter"]["args"]) == 3
    assert body["filter"]["args"][2] == {
        "op": "=",
        "args": [{"property": "sat:orbit_state"}, "ascending"],
    }


def test_to_search_body_capability_gating():
    translation = TranslationResult(
        bbox=[1, 2, 3, 4],
        collections=["c1"],
        property_filters=[PropertyFilter(property="eo:cloud_cover", op="<=", value=10.0)],
        free_text="flood",
        sortby=[SortField(field="datetime", direction="desc")],
    )
    no_caps = SearchCapabilities(filter=False, item_free_text=False, sort=False)
    body = to_search_body(translation, no_caps)
    assert set(body) == {"bbox", "collections"}

    body = to_search_body(translation, FULL_CAPS, collection_id="c1")
    assert "collections" not in body, "fixed-collection context drops collections"
    assert body["filter"] == {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 10.0]}
    assert body["q"] == ["flood"]


def test_free_text_terms_tokenization():
    assert free_text_terms("aerial imagery") == ["aerial", "imagery"]
    assert free_text_terms("Sentinel-2") == ["Sentinel-2"], "hyphenated tokens survive"
    assert free_text_terms("black & white!") == ["black", "white"]
    assert free_text_terms("(urban) areas, flood/damage") == [
        "urban",
        "areas",
        "flood",
        "damage",
    ]
    assert free_text_terms("&()!") == []
    assert free_text_terms("") == []


def test_free_text_projection_is_backend_safe():
    # basic flavor: multi-word phrases become spec-conformant OR terms
    body = to_search_body(TranslationResult(free_text="flood damage (urban)"), FULL_CAPS)
    assert body["q"] == ["flood", "damage", "urban"]

    params = to_collection_search_params(
        TranslationResult(free_text="aerial imagery"), FULL_CAPS
    )
    assert params["q"] == ["aerial", "imagery"]

    # nothing tokenizable -> q omitted entirely instead of sending garbage
    assert "q" not in to_search_body(TranslationResult(free_text="&&&"), FULL_CAPS)
    assert "q" not in to_collection_search_params(
        TranslationResult(free_text="&&&"), FULL_CAPS
    )

    # advanced flavor (single string) keeps the phrase for its richer grammar
    advanced = SearchCapabilities(
        item_free_text=True, collection_free_text=True, free_text_is_list=False
    )
    assert (
        to_search_body(TranslationResult(free_text="flood damage"), advanced)["q"]
        == "flood damage"
    )


def test_to_collection_search_params_prefers_platform_keyword():
    translation = TranslationResult(
        platform_keyword="Sentinel-2", free_text="flood", limit=5
    )
    params = to_collection_search_params(translation, FULL_CAPS)
    assert params["q"] == ["Sentinel-2"]
    assert params["limit"] == 5

    advanced = SearchCapabilities(collection_free_text=True, free_text_is_list=False)
    assert to_collection_search_params(translation, advanced)["q"] == "Sentinel-2"


def test_ttl_cache_single_flight_and_expiry():
    async def scenario():
        cache = TTLCache(maxsize=4, ttl=0.05)
        calls = []

        async def factory():
            calls.append(1)
            await asyncio.sleep(0.01)
            return "value"

        results = await asyncio.gather(
            cache.get_or_create("k", factory),
            cache.get_or_create("k", factory),
            cache.get_or_create("k", factory),
        )
        assert results == ["value"] * 3
        assert len(calls) == 1, "concurrent lookups must share one factory call"

        assert await cache.get_or_create("k", factory) == "value"
        assert len(calls) == 1, "hit within TTL"

        await asyncio.sleep(0.06)
        await cache.get_or_create("k", factory)
        assert len(calls) == 2, "expired entry is rebuilt"

        failures = []

        async def failing():
            failures.append(1)
            raise RuntimeError("boom")

        for _ in range(2):
            try:
                await cache.get_or_create("fail", failing)
            except RuntimeError:
                pass
        assert len(failures) == 2, "errors are never cached"

    asyncio.run(scenario())
