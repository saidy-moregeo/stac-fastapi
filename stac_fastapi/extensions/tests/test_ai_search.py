"""Tests for the /ai-search endpoint, errors, caching and conformance."""

import asyncio
from urllib.parse import parse_qsl, urlsplit

from ai_search_helpers import DummyCoreClient, make_api
from starlette.testclient import TestClient

from stac_fastapi.api.app import StacApi
from stac_fastapi.api.models import create_get_request_model
from stac_fastapi.extensions import (
    AISearchConformanceClasses,
    AISearchExtension,
    AISearchSettings,
    SortExtension,
)
from stac_fastapi.extensions.ai_search.testing import StaticTranslator
from stac_fastapi.extensions.ai_search.types import (
    PropertyFilter,
    SortField,
    TranslationResult,
)
from stac_fastapi.types.config import ApiSettings
from stac_fastapi.types.core import AsyncBaseCoreClient

TRANSLATION = TranslationResult(
    bbox=[5.8, 50.3, 9.5, 52.5],
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01T00:00:00Z/2023-08-31T23:59:59Z",
    property_filters=[PropertyFilter(property="eo:cloud_cover", op="<=", value=10.0)],
    platform_keyword="Sentinel-2",
    limit=10,
    sortby=[SortField(field="datetime", direction="desc")],
)


def test_ai_search_get_uses_query_parameters():
    """Regression: /ai-search must take query params, not a request body."""
    api = make_api(StaticTranslator(TRANSLATION))
    with TestClient(api.app) as client:
        response = client.get("/ai-search", params={"prompt": "sentinel-2 germany"})
        assert response.is_success, response.text

        spec = client.get("/api").json()
        params = {p["name"]: p for p in spec["paths"]["/ai-search"]["get"]["parameters"]}
        assert params["prompt"]["in"] == "query"
        assert params["prompt"]["required"] is True
        assert params["target_hint"]["in"] == "query"
        assert "requestBody" not in spec["paths"]["/ai-search"]["get"]


def test_ai_search_composes_with_other_extensions():
    """Regression: fragments must compose into GET/POST models without TypeError."""
    from stac_fastapi.api.models import create_post_request_model
    from stac_fastapi.extensions import AISearchPromptExtension

    extensions = [SortExtension(), AISearchPromptExtension()]
    get_model = create_get_request_model(extensions)
    post_model = create_post_request_model(extensions)
    assert hasattr(get_model(), "prompt")
    assert "prompt" in post_model.model_fields


def test_ai_search_items_target():
    translator = StaticTranslator(TRANSLATION)
    client_impl = DummyCoreClient()
    api = make_api(translator, client=client_impl)

    with TestClient(api.app) as client:
        response = client.get(
            "/ai-search", params={"prompt": "sentinel", "target_hint": "items"}
        )
        assert response.is_success, response.text
        assert response.headers["content-type"].startswith("application/geo+json")

        body = response.json()
        assert body["type"] == "FeatureCollection"
        assert body["features"] == [{"id": "item-1"}]

        # executed through the composed POST model
        received = client_impl.record["post_search"]
        assert received["bbox"] == TRANSLATION.bbox
        assert received["collections"] == ["sentinel-2-l2a"]
        assert received["filter_expr"] == {
            "op": "<=",
            "args": [{"property": "eo:cloud_cover"}, 10.0],
        }

        # parameters echo is the AI-derived searchBody
        assert body["parameters"]["bbox"] == TRANSLATION.bbox
        assert body["parameters"]["filter-lang"] == "cql2-json"
        # links: own self/root plus a synthesized continuation against /search
        rels = [link["rel"] for link in body["links"]]
        assert rels.count("self") == 1 and "root" in rels
        (next_link,) = [link for link in body["links"] if link["rel"] == "next"]
        assert next_link["method"] == "POST"
        assert next_link["href"] == "http://testserver/search"
        assert next_link["type"] == "application/geo+json"
        # complete body: the echoed parameters plus the relayed cursor
        assert next_link["body"] == {**body["parameters"], "token": "next:abc"}
        assert "merge" not in next_link
        assert "prompt" not in next_link["body"]


def test_ai_search_collections_target():
    client_impl = DummyCoreClient()
    api = make_api(StaticTranslator(TRANSLATION), client=client_impl)

    with TestClient(api.app) as client:
        response = client.get(
            "/ai-search", params={"prompt": "sentinel", "target_hint": "collections"}
        )
        assert response.is_success, response.text
        assert response.headers["content-type"].startswith("application/json")

        body = response.json()
        assert [c["id"] for c in body["collections"]] == [
            "sentinel-2-l2a",
            "landsat-c2-l2",
        ]
        received = client_impl.record["all_collections"]
        assert received["q"] == ["Sentinel-2"], "platform keyword grounds q"
        assert list(received["bbox"]) == [5.8, 50.3, 9.5, 52.5]
        assert "parameters" in body

        # synthesized GET continuation against /collections
        (next_link,) = [link for link in body["links"] if link["rel"] == "next"]
        assert next_link["method"] == "GET"
        assert next_link["type"] == "application/json"
        split = urlsplit(next_link["href"])
        assert split.path == "/collections"
        query = dict(parse_qsl(split.query))
        assert query["offset"] == "10", "backend cursor relayed verbatim"
        assert query["bbox"] == "5.8,50.3,9.5,52.5"
        assert query["q"] == "Sentinel-2"
        assert query["limit"] == "10"
        assert "prompt" not in query and "target_hint" not in query


def test_ai_search_combined_default():
    client_impl = DummyCoreClient()
    api = make_api(StaticTranslator(TRANSLATION), client=client_impl)

    with TestClient(api.app) as client:
        response = client.get("/ai-search", params={"prompt": "sentinel"})
        assert response.is_success, response.text
        assert response.headers["content-type"].startswith("application/geo+json")

        body = response.json()
        assert body["type"] == "FeatureCollection"
        assert body["features"] == [{"id": "item-1"}]
        assert [c["id"] for c in body["collections"]] == [
            "sentinel-2-l2a",
            "landsat-c2-l2",
        ]
        assert isinstance(body["parameters"], dict)
        assert "features" not in body["parameters"]
        # combined-mode paging is a documented spec gap: no continuation links
        rels = [link["rel"] for link in body["links"]]
        assert not {"next", "prev", "previous"} & set(rels)


def test_ai_search_no_continuation_without_cursor():
    """A backend paging link with no surviving cursor synthesizes nothing."""

    class NoCursorClient(DummyCoreClient):
        def post_search(self, search_request, **kwargs):
            response = super().post_search(search_request, **kwargs)
            response["links"] = [
                {"rel": "next", "href": "http://testserver/ai-search?prompt=sentinel"}
            ]
            return response

    api = make_api(StaticTranslator(TRANSLATION), client=NoCursorClient())
    with TestClient(api.app) as client:
        response = client.get(
            "/ai-search", params={"prompt": "sentinel", "target_hint": "items"}
        )
        assert response.is_success, response.text
        rels = [link["rel"] for link in response.json()["links"]]
        assert not {"next", "prev", "previous"} & set(rels)


def test_ai_search_relays_prev_links():
    """Backend prev links are synthesized alongside next ones."""

    class PrevLinkClient(DummyCoreClient):
        def post_search(self, search_request, **kwargs):
            response = super().post_search(search_request, **kwargs)
            response["links"].append(
                {
                    "rel": "prev",
                    "href": (
                        "http://testserver/ai-search"
                        "?prompt=sentinel&target_hint=items&token=prev:xyz"
                    ),
                }
            )
            return response

    api = make_api(StaticTranslator(TRANSLATION), client=PrevLinkClient())
    with TestClient(api.app) as client:
        response = client.get(
            "/ai-search", params={"prompt": "sentinel", "target_hint": "items"}
        )
        assert response.is_success, response.text
        body = response.json()
        by_rel = {
            link["rel"]: link for link in body["links"] if link["rel"] in ("next", "prev")
        }
        assert by_rel["next"]["body"]["token"] == "next:abc"
        assert by_rel["prev"]["body"]["token"] == "prev:xyz"
        assert by_rel["prev"]["method"] == "POST"
        assert by_rel["prev"]["href"] == "http://testserver/search"


class ConcurrencyProbeClient(AsyncBaseCoreClient):
    """Deadlocks unless items and collections searches run concurrently."""

    def _events(self):
        if not hasattr(self, "_evts"):
            self._evts = (asyncio.Event(), asyncio.Event())
        return self._evts

    async def post_search(self, search_request, **kwargs):
        items_started, collections_started = self._events()
        items_started.set()
        await asyncio.wait_for(collections_started.wait(), timeout=2)
        return {"type": "FeatureCollection", "features": [], "links": []}

    async def all_collections(self, **kwargs):
        items_started, collections_started = self._events()
        collections_started.set()
        await asyncio.wait_for(items_started.wait(), timeout=2)
        return {"collections": [], "links": []}

    async def get_search(self, **kwargs):
        raise NotImplementedError

    async def get_collection(self, *args, **kwargs):
        raise NotImplementedError

    async def get_item(self, *args, **kwargs):
        raise NotImplementedError

    async def item_collection(self, *args, **kwargs):
        raise NotImplementedError


def test_ai_search_combined_runs_concurrently():
    api = make_api(
        StaticTranslator(TranslationResult()),
        client=ConcurrencyProbeClient(),
        ai_settings=AISearchSettings(enabled=True, resolve_collections=False),
    )
    with TestClient(api.app) as client:
        response = client.get("/ai-search", params={"prompt": "anything"})
        assert response.is_success, response.text


def test_queryables_grounding_admits_and_drops_filters():
    from ai_search_helpers import DummyQueryablesClient

    translator = StaticTranslator(
        TranslationResult(
            property_filters=[
                PropertyFilter(property="eo:cloud_cover", op="<=", value=10),
                PropertyFilter(property="s2:water_percentage", op="<", value=50),
                PropertyFilter(property="hallucinated:prop", op="=", value=1),
            ]
        )
    )
    client_impl = DummyCoreClient()
    queryables = DummyQueryablesClient()
    api = make_api(translator, client=client_impl, queryables_client=queryables)

    with TestClient(api.app) as client:
        response = client.get(
            "/ai-search", params={"prompt": "watery scenes", "target_hint": "items"}
        )
        assert response.is_success, response.text

    assert queryables.calls == 1
    # queryables admit the deployment-specific property; the hallucinated one
    # is dropped; the pair survives as an AND tree
    sent = client_impl.record["post_search"]["filter_expr"]
    assert sent["op"] == "and"
    assert [a["args"][0]["property"] for a in sent["args"]] == [
        "eo:cloud_cover",
        "s2:water_percentage",
    ]
    # the translator saw the queryable names for grounding
    assert "s2:water_percentage" in (translator.calls[0].queryables or [])


def test_ai_search_error_codes():
    api = make_api(StaticTranslator(TRANSLATION))
    with TestClient(api.app) as client:
        # 400: sanitizes to empty
        response = client.get("/ai-search", params={"prompt": "<b></b>"})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "AI_SEARCH_PROMPT_EMPTY"

    # 400: over the configured (lower) limit
    api = make_api(
        StaticTranslator(TRANSLATION),
        ai_settings=AISearchSettings(enabled=True, max_prompt_length=10),
    )
    with TestClient(api.app) as client:
        response = client.get("/ai-search", params={"prompt": "x" * 20})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "AI_SEARCH_PROMPT_TOO_LONG"

    # 502: translator failure
    api = make_api(StaticTranslator(raises=RuntimeError("provider exploded")))
    with TestClient(api.app) as client:
        response = client.get("/ai-search", params={"prompt": "sentinel"})
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "AI_SEARCH_TRANSLATION_FAILED"
        assert "exploded" not in response.text, "provider internals must not leak"

    # 503: missing dependency
    api = make_api(StaticTranslator(raises=ImportError("no pydantic_ai")))
    with TestClient(api.app) as client:
        response = client.get("/ai-search", params={"prompt": "sentinel"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "AI_SEARCH_NOT_INSTALLED"

    # 503: no translator configured at all
    api = make_api(None, ai_settings=AISearchSettings(enabled=True, model=None))
    with TestClient(api.app) as client:
        response = client.get("/ai-search", params={"prompt": "sentinel"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "AI_SEARCH_PROVIDER_CONFIG_INVALID"


def test_ai_search_provider_exception_mapping():
    class UsageLimitExceeded(Exception):
        pass

    class ModelHTTPError(Exception):
        status_code = 401

    api = make_api(StaticTranslator(raises=UsageLimitExceeded("limit")))
    with TestClient(api.app) as client:
        response = client.get("/ai-search", params={"prompt": "sentinel"})
        assert response.status_code == 429
        assert response.json()["detail"]["code"] == "AI_SEARCH_PROVIDER_RATE_LIMITED"

    api = make_api(StaticTranslator(raises=ModelHTTPError("auth")))
    with TestClient(api.app) as client:
        response = client.get("/ai-search", params={"prompt": "sentinel"})
        assert response.status_code == 503


def test_ai_search_rate_limited():
    try:
        import limits  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("limits not installed")

    api = make_api(
        StaticTranslator(TRANSLATION),
        ai_settings=AISearchSettings(
            enabled=True, rate_limit="1/minute", resolve_collections=False
        ),
    )
    with TestClient(api.app) as client:
        first = client.get("/ai-search", params={"prompt": "sentinel"})
        assert first.is_success, first.text
        second = client.get("/ai-search", params={"prompt": "sentinel again"})
        assert second.status_code == 429
        assert second.json()["detail"]["code"] == "AI_SEARCH_RATE_LIMITED"


def test_translation_cache_deduplicates_llm_calls():
    translator = StaticTranslator(TRANSLATION)
    api = make_api(translator)
    with TestClient(api.app) as client:
        for _ in range(3):
            response = client.get(
                "/ai-search", params={"prompt": "same prompt", "target_hint": "items"}
            )
            assert response.is_success
    assert len(translator.calls) == 1

    # different prompt misses the cache
    with TestClient(api.app) as client:
        client.get("/ai-search", params={"prompt": "other", "target_hint": "items"})
    assert len(translator.calls) == 2


def test_grounding_drops_unknown_collections():
    translator = StaticTranslator(
        TranslationResult(collections=["sentinel-2-l2a", "hallucinated-collection"])
    )
    client_impl = DummyCoreClient()
    api = make_api(translator, client=client_impl)
    with TestClient(api.app) as client:
        response = client.get(
            "/ai-search", params={"prompt": "sentinel", "target_hint": "items"}
        )
        assert response.is_success
    assert client_impl.record["post_search"]["collections"] == ["sentinel-2-l2a"]


def test_conformance_classes_exposed():
    api = make_api(StaticTranslator(TRANSLATION))
    with TestClient(api.app) as client:
        conforms = client.get("/conformance").json()["conformsTo"]
    for uri in AISearchConformanceClasses:
        assert uri.value in conforms, uri


def test_extension_constructs_without_pydantic_ai():
    """The extension must be usable (and importable) without the extra installed."""
    settings = ApiSettings()
    client = DummyCoreClient()
    extension = AISearchExtension(client=client, settings=AISearchSettings(enabled=True))
    api = StacApi(settings=settings, client=client, extensions=[extension])
    with TestClient(api.app) as http:
        assert http.get("/ai-search", params={"prompt": "x"}).status_code in (
            502,
            503,
        )
