"""Tests for prompt handling on the existing (wrapped) endpoints."""

from ai_search_helpers import DummyCoreClient, make_api
from starlette.testclient import TestClient

from stac_fastapi.api.app import StacApi
from stac_fastapi.extensions import AISearchExtension, AISearchSettings, SortExtension
from stac_fastapi.extensions.ai_search.testing import StaticTranslator
from stac_fastapi.extensions.ai_search.types import (
    PropertyFilter,
    SortField,
    TranslationResult,
)
from stac_fastapi.types.config import ApiSettings

TRANSLATION = TranslationResult(
    bbox=[5.8, 50.3, 9.5, 52.5],
    datetime="2022-01-01T00:00:00Z/2022-12-31T23:59:59Z",
    property_filters=[PropertyFilter(property="eo:cloud_cover", op="<=", value=10.0)],
    platform_keyword="Sentinel-2",
    free_text="flood damage",
    limit=99,
    sortby=[SortField(field="datetime", direction="desc")],
)

NO_GROUNDING = AISearchSettings(enabled=True, resolve_collections=False)


def test_search_get_prompt_merges_ai_parameters():
    client_impl = DummyCoreClient()
    api = make_api(
        StaticTranslator(TRANSLATION), client=client_impl, ai_settings=NO_GROUNDING
    )
    with TestClient(api.app) as client:
        response = client.get("/search", params={"prompt": "sentinel over germany"})
        assert response.is_success, response.text

        received = response.json()["received"]
        assert received["bbox"] == [5.8, 50.3, 9.5, 52.5]
        assert received["datetime"] == TRANSLATION.datetime
        assert received["limit"] == 99, "GET default must not mask AI limit"
        assert received["sortby"] == ["-datetime"]
        assert received["q"] == ["flood", "damage"], "free text is tokenized"
        assert received["filter_lang"] == "cql2-json"
        assert "eo:cloud_cover" in received["filter_expr"]
        assert "prompt" not in received, "prompt must never reach the client"

        parameters = response.json()["parameters"]
        assert parameters["bbox"] == TRANSLATION.bbox
        assert parameters["filter-lang"] == "cql2-json"


def test_search_get_user_parameters_win():
    client_impl = DummyCoreClient()
    api = make_api(
        StaticTranslator(TRANSLATION), client=client_impl, ai_settings=NO_GROUNDING
    )
    with TestClient(api.app) as client:
        response = client.get(
            "/search",
            params={"prompt": "sentinel", "bbox": "9,9,10,10", "limit": 5},
        )
        assert response.is_success, response.text

        received = response.json()["received"]
        assert received["bbox"] == [9, 9, 10, 10]
        assert received["limit"] == 5
        assert received["datetime"] == TRANSLATION.datetime

        parameters = response.json()["parameters"]
        assert "bbox" not in parameters and "limit" not in parameters
        assert parameters["datetime"] == TRANSLATION.datetime


def test_search_get_without_prompt_is_untouched():
    client_impl = DummyCoreClient()
    api = make_api(
        StaticTranslator(TRANSLATION), client=client_impl, ai_settings=NO_GROUNDING
    )
    with TestClient(api.app) as client:
        response = client.get("/search", params={"bbox": "1,2,3,4"})
        assert response.is_success, response.text
        body = response.json()
        assert "parameters" not in body
        received = body["received"]
        assert received["bbox"] == [1, 2, 3, 4]
        assert received["limit"] == 10
        assert "datetime" not in received and "sortby" not in received


def test_search_post_prompt_merges_under_user_precedence():
    client_impl = DummyCoreClient()
    api = make_api(
        StaticTranslator(TRANSLATION), client=client_impl, ai_settings=NO_GROUNDING
    )
    with TestClient(api.app) as client:
        response = client.post("/search", json={"prompt": "sentinel", "limit": 5})
        assert response.is_success, response.text

        received = response.json()["received"]
        assert received["limit"] == 5, "user body value wins over AI limit"
        assert received["bbox"] == TRANSLATION.bbox
        assert received["datetime"] == TRANSLATION.datetime
        assert "prompt" not in received

        parameters = response.json()["parameters"]
        assert "limit" not in parameters
        assert parameters["bbox"] == TRANSLATION.bbox


def test_search_post_explicit_default_counts_as_user_set():
    client_impl = DummyCoreClient()
    api = make_api(
        StaticTranslator(TRANSLATION), client=client_impl, ai_settings=NO_GROUNDING
    )
    with TestClient(api.app) as client:
        response = client.post("/search", json={"prompt": "sentinel", "limit": 10})
        assert response.is_success, response.text
        assert response.json()["received"]["limit"] == 10


def test_search_post_user_filter_wins():
    user_filter = {"op": "=", "args": [{"property": "platform"}, "sentinel-2b"]}
    client_impl = DummyCoreClient()
    api = make_api(
        StaticTranslator(TRANSLATION), client=client_impl, ai_settings=NO_GROUNDING
    )
    with TestClient(api.app) as client:
        response = client.post(
            "/search", json={"prompt": "sentinel", "filter": user_filter}
        )
        assert response.is_success, response.text
        received = response.json()["received"]
        assert received["filter_expr"] == user_filter
        assert "filter" not in response.json()["parameters"]


def test_search_post_filter_and_strategy_composes():
    user_filter = {"op": "=", "args": [{"property": "platform"}, "sentinel-2b"]}
    client_impl = DummyCoreClient()
    api = make_api(
        StaticTranslator(TRANSLATION),
        client=client_impl,
        ai_settings=AISearchSettings(
            enabled=True, resolve_collections=False, filter_merge_strategy="and"
        ),
    )
    with TestClient(api.app) as client:
        response = client.post(
            "/search", json={"prompt": "sentinel", "filter": user_filter}
        )
        assert response.is_success, response.text
        received = response.json()["received"]
        assert received["filter_expr"]["op"] == "and"
        assert received["filter_expr"]["args"][0] == user_filter
        # the echo records only the AI-derived part
        assert response.json()["parameters"]["filter"] == {
            "op": "<=",
            "args": [{"property": "eo:cloud_cover"}, 10.0],
        }


def test_items_endpoint_prompt_respects_fixed_collection():
    translator = StaticTranslator(
        TranslationResult(
            bbox=[1.0, 2.0, 3.0, 4.0],
            datetime="2022-01-01T00:00:00Z/2022-12-31T23:59:59Z",
            collections=["some-other-collection"],
        )
    )
    client_impl = DummyCoreClient()
    api = make_api(translator, client=client_impl, ai_settings=NO_GROUNDING)
    with TestClient(api.app) as client:
        response = client.get(
            "/collections/sentinel-2-l2a/items", params={"prompt": "flood scenes"}
        )
        assert response.is_success, response.text

        received = response.json()["received"]
        assert received["collection_id"] == "sentinel-2-l2a"
        assert received["bbox"] == [1, 2, 3, 4]
        assert received["datetime"] == "2022-01-01T00:00:00Z/2022-12-31T23:59:59Z"
        assert "collections" not in received, "fixed collection cannot be widened"
        assert "prompt" not in received
        # the translator was told about the fixed collection
        assert translator.calls[0].collection_id == "sentinel-2-l2a"


def test_collections_endpoint_prompt():
    client_impl = DummyCoreClient()
    api = make_api(
        StaticTranslator(TRANSLATION), client=client_impl, ai_settings=NO_GROUNDING
    )
    with TestClient(api.app) as client:
        response = client.get("/collections", params={"prompt": "sentinel data"})
        assert response.is_success, response.text

        received = response.json()["received"]
        assert received["bbox"] == [5.8, 50.3, 9.5, 52.5]
        assert received["q"] == ["Sentinel-2"], "collections search grounds on platform"
        assert "prompt" not in received
        assert response.json()["parameters"]["q"] == ["Sentinel-2"]


def test_collections_endpoint_user_precedence():
    client_impl = DummyCoreClient()
    api = make_api(
        StaticTranslator(TRANSLATION), client=client_impl, ai_settings=NO_GROUNDING
    )
    with TestClient(api.app) as client:
        response = client.get(
            "/collections", params={"prompt": "sentinel", "q": "landsat"}
        )
        assert response.is_success, response.text
        assert response.json()["received"]["q"] == ["landsat"]


def test_routes_not_wrapped_without_prompt_fragments():
    """An app without the fragments still gets /ai-search but no route wrapping."""
    client_impl = DummyCoreClient()
    extension = AISearchExtension(
        client=client_impl,
        settings=AISearchSettings(enabled=True, resolve_collections=False),
        translator=StaticTranslator(TRANSLATION),
    )
    api = StacApi(
        settings=ApiSettings(),
        client=client_impl,
        extensions=[SortExtension(), extension],
    )
    for route in api.app.router.routes:
        endpoint = getattr(route, "endpoint", None)
        if getattr(route, "path", "") == "/search":
            assert not getattr(endpoint, "_ai_search_wrapped", False)

    with TestClient(api.app) as client:
        assert client.get(
            "/ai-search", params={"prompt": "x", "target_hint": "items"}
        ).is_success


def test_double_registration_is_idempotent():
    client_impl = DummyCoreClient()
    translator = StaticTranslator(TRANSLATION)
    api = make_api(translator, client=client_impl, ai_settings=NO_GROUNDING)

    ai_extension = next(
        ext for ext in api.extensions if isinstance(ext, AISearchExtension)
    )
    ai_extension._wrap_prompt_routes(api.app)  # second pass must be a no-op

    with TestClient(api.app) as client:
        response = client.get("/search", params={"prompt": "sentinel"})
        assert response.is_success, response.text
    assert len(translator.calls) == 1, "prompt must be translated exactly once"
