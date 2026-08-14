"""Shared helpers for AI-search extension tests."""

from typing import Any, Dict, Optional

from stac_fastapi.api.app import StacApi
from stac_fastapi.api.models import (
    ItemCollectionUri,
    create_get_request_model,
    create_post_request_model,
    create_request_model,
)
from stac_fastapi.extensions import (
    AISearchConformanceClasses,
    AISearchExtension,
    AISearchPromptExtension,
    AISearchSettings,
    CollectionSearchExtension,
    FreeTextExtension,
    SortExtension,
)
from stac_fastapi.extensions.collection_search.request import (
    BaseCollectionSearchGetRequest,
)
from stac_fastapi.extensions.filter import FilterExtension
from stac_fastapi.extensions.free_text import FreeTextConformanceClasses
from stac_fastapi.types.config import ApiSettings
from stac_fastapi.types.core import BaseCoreClient

CATALOG = [
    {"id": "sentinel-2-l2a", "title": "Sentinel-2 Level-2A"},
    {"id": "landsat-c2-l2", "title": "Landsat Collection 2 Level-2"},
]


class DummyCoreClient(BaseCoreClient):
    """Records the kwargs each method receives and returns STAC-shaped dicts."""

    @property
    def record(self) -> Dict[str, Any]:
        if not hasattr(self, "_record"):
            self._record = {}
        return self._record

    @staticmethod
    def _clean(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs.pop("request", None)
        return {k: v for k, v in kwargs.items() if v is not None}

    def all_collections(self, *args, **kwargs):
        received = self._clean(kwargs)
        self.record["all_collections"] = received
        # paging href as backends build it: incoming request URL + cursor
        return {
            "collections": list(CATALOG),
            "links": [
                {
                    "rel": "next",
                    "href": "http://testserver/ai-search?prompt=sentinel&offset=10",
                }
            ],
            "received": received,
        }

    def get_collection(self, *args, **kwargs):
        raise NotImplementedError

    def get_item(self, *args, **kwargs):
        raise NotImplementedError

    def get_search(self, *args, **kwargs):
        received = self._clean(kwargs)
        self.record["get_search"] = received
        return {
            "type": "FeatureCollection",
            "features": [],
            "links": [],
            "received": received,
        }

    def post_search(self, search_request, **kwargs):
        received = search_request.model_dump(mode="json", exclude_none=True)
        self.record["post_search"] = received
        # paging href as backends build it: incoming request URL + cursor
        return {
            "type": "FeatureCollection",
            "features": [{"id": "item-1"}],
            "links": [
                {
                    "rel": "next",
                    "href": (
                        "http://testserver/ai-search"
                        "?prompt=sentinel&target_hint=items&token=next:abc"
                    ),
                }
            ],
            "received": received,
        }

    def item_collection(self, collection_id, *args, **kwargs):
        received = {"collection_id": collection_id, **self._clean(kwargs)}
        self.record["item_collection"] = received
        return {
            "type": "FeatureCollection",
            "features": [],
            "links": [],
            "received": received,
        }


class DummyQueryablesClient:
    """Returns a fixed queryables JSON schema, recording calls."""

    def __init__(self, properties=("eo:cloud_cover", "s2:water_percentage")):
        self.properties = properties
        self.calls = 0

    def get_queryables(self, collection_id=None, **kwargs):
        self.calls += 1
        return {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "properties": {name: {} for name in self.properties},
        }


def make_api(
    translator: Any,
    ai_settings: Optional[AISearchSettings] = None,
    client: Optional[BaseCoreClient] = None,
    queryables_client: Any = None,
) -> StacApi:
    """Build a StacApi with prompt fragments on every scope + the orchestrator."""
    client = client or DummyCoreClient()
    ai_settings = ai_settings or AISearchSettings(enabled=True)

    prompt_search = AISearchPromptExtension(
        conformance_classes=[AISearchConformanceClasses.SEARCH]
    )
    prompt_items = AISearchPromptExtension(
        conformance_classes=[AISearchConformanceClasses.ITEMS]
    )
    prompt_collections = AISearchPromptExtension(
        conformance_classes=[AISearchConformanceClasses.COLLECTIONS]
    )

    free_text = FreeTextExtension(
        conformance_classes=[
            FreeTextConformanceClasses.SEARCH,
            FreeTextConformanceClasses.COLLECTIONS,
        ]
    )
    search_extensions = [SortExtension(), free_text, FilterExtension(), prompt_search]

    search_get_model = create_get_request_model(search_extensions)
    search_post_model = create_post_request_model(search_extensions)
    items_get_model = create_request_model(
        "ItemCollectionUri",
        base_model=ItemCollectionUri,
        extensions=[prompt_items],
        mixins=[FilterExtension().GET],
        request_type="GET",
    )
    collections_get_model = create_request_model(
        "CollectionsGetRequest",
        base_model=BaseCollectionSearchGetRequest,
        extensions=[free_text, prompt_collections],
        request_type="GET",
    )

    ai_extension = AISearchExtension(
        client=client,
        settings=ai_settings,
        translator=translator,
        queryables_client=queryables_client,
        search_post_request_model=search_post_model,
        collections_get_request_model=collections_get_model,
    )

    return StacApi(
        settings=ApiSettings(),
        client=client,
        extensions=[
            *search_extensions,
            prompt_items,
            prompt_collections,
            CollectionSearchExtension(GET=collections_get_model),
            ai_extension,
        ],
        search_get_request_model=search_get_model,
        search_post_request_model=search_post_model,
        items_get_request_model=items_get_model,
        collections_get_request_model=collections_get_model,
    )
