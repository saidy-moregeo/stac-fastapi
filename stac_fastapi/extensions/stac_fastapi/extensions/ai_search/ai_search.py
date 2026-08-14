"""AI-Assisted Search extension.

Implements https://stac-extensions.moregeo.it/v0.1.0 for stac-fastapi:

- ``AISearchPromptExtension``: a parameter-only fragment adding `prompt` to
  the composed request models of existing endpoints (one instance per
  conformance scope, like `FreeTextExtension`).
- ``AISearchExtension``: registers ``GET /ai-search`` and wraps the existing
  search routes so a supplied `prompt` is translated into STAC parameters and
  merged under user precedence — with zero changes required in backend
  clients.
"""

import asyncio
import functools
import hashlib
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit

import attr
from fastapi import APIRouter, FastAPI
from fastapi.datastructures import DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_dependant
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute, request_response
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

from stac_fastapi.api.models import GeoJSONResponse
from stac_fastapi.api.routes import create_async_endpoint
from stac_fastapi.types.core import AsyncBaseCoreClient, BaseCoreClient
from stac_fastapi.types.extension import ApiExtension
from stac_fastapi.types.search import APIRequest, BaseSearchPostRequest

from .cache import TTLCache
from .catalog import CollectionCatalog, call_client_method
from .errors import (
    AISearchError,
    ProviderConfigError,
    TranslationFailedError,
    map_translation_exception,
)
from .limiter import AISearchRateLimiter
from .merge import (
    WIRE_TO_FIELD,
    apply_to_get_request,
    apply_to_post_request,
    convert_for_get,
    get_user_wire_keys,
    post_user_wire_keys,
)
from .request import (
    AISearchGetRequest,
    AISearchPromptGetRequest,
    AISearchPromptPostRequest,
    validate_prompt,
)
from .settings import AISearchSettings
from .translator import Translator, validate_translation
from .types import (
    AISearchTarget,
    EndpointKind,
    SearchCapabilities,
    TranslationRequest,
    TranslationResult,
    to_collection_search_params,
    to_search_body,
)

logger = logging.getLogger(__name__)


class AISearchConformanceClasses(str, Enum):
    """Conformance classes for the AI Search extension.

    See https://github.com/saidy-moregeo/stac-api-ai-search-extension
    """

    ENDPOINT = "https://stac-extensions.moregeo.it/v0.1.0/extensions/ai-search"
    SEARCH = "https://stac-extensions.moregeo.it/v0.1.0/item-search#ai-search"
    ITEMS = "https://stac-extensions.moregeo.it/v0.1.0/ogcapi-features#ai-search"
    COLLECTIONS = "https://stac-extensions.moregeo.it/v0.1.0/collection-search#ai-search"


@attr.s
class AISearchPromptExtension(ApiExtension):
    """`prompt` parameter fragment for existing endpoints.

    Compose one instance per endpoint scope into the corresponding request
    model builders (like `FreeTextExtension`), with that scope's conformance
    class::

        AISearchPromptExtension(
            conformance_classes=[AISearchConformanceClasses.SEARCH]
        )

    Prompt handling on the wrapped routes is done by `AISearchExtension`.
    """

    GET: Type[APIRequest] = AISearchPromptGetRequest
    POST: Type[BaseModel] = AISearchPromptPostRequest

    conformance_classes: List[str] = attr.ib(
        factory=lambda: [AISearchConformanceClasses.SEARCH.value]
    )
    schema_href: Optional[str] = attr.ib(default=None)

    def register(self, app: FastAPI) -> None:
        """No routes of its own; parameters are added via request models."""
        pass


_PROMPT_ROUTE_TARGETS = [
    ("/search", "GET", EndpointKind.ITEM_SEARCH),
    ("/search", "POST", EndpointKind.ITEM_SEARCH),
    ("/collections", "GET", EndpointKind.COLLECTION_SEARCH),
    ("/collections", "POST", EndpointKind.COLLECTION_SEARCH),
    ("/collections/{collection_id}/items", "GET", EndpointKind.FEATURES_ITEMS),
]

#: sub-search paging rels continued via synthesized links
_CONTINUATION_RELS = {"next", "prev", "previous"}

#: AI-search parameters that must never appear in continuation links
_AI_PARAMS = {"prompt", "target_hint"}

_ERROR_RESPONSES = {
    400: {"description": "Invalid request input."},
    429: {"description": "Rate limit exceeded."},
    502: {"description": "AI provider translation failure."},
    503: {"description": "AI provider configuration missing or invalid."},
}


def _relay_params(link: Dict[str, Any]) -> Dict[str, Any]:
    """Cursor parameters carried in a backend paging link, relayed verbatim.

    Backends build paging hrefs from the incoming request URL — inside
    `/ai-search` that URL is the AI endpoint itself, so the href query string
    holds the opaque cursor (`token`, `offset`, ...) next to the AI
    parameters, which are stripped. The cursor is never parsed or rebuilt.
    """
    relay = {
        key: value
        for key, value in parse_qsl(urlsplit(link.get("href", "")).query)
        if key not in _AI_PARAMS
    }
    body = link.get("body")
    if isinstance(body, dict):
        relay.update({k: v for k, v in body.items() if k not in _AI_PARAMS | {"merge"}})
    return relay


def _query_value(value: Any) -> str:
    """GET query-string form of a collection-search wire value."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _iter_api_routes(routes: Any):
    """Yield APIRoutes, descending into included/mounted routers.

    Mirrors the recursion in `stac_fastapi.api.routes.add_route_dependencies`
    (FastAPI can nest included routers instead of flattening them).
    """
    for route in routes:
        if hasattr(route, "original_router"):
            yield from _iter_api_routes(route.original_router.routes)
            continue
        if hasattr(route, "routes") and route.routes:
            yield from _iter_api_routes(route.routes)
            continue
        if isinstance(route, APIRoute):
            yield route


def _dependant_has_query_prompt(dependant: Dependant) -> bool:
    for field in dependant.query_params:
        if field.name == "prompt" or field.alias == "prompt":
            return True
    return any(_dependant_has_query_prompt(sub) for sub in dependant.dependencies)


def _dependant_has_body_prompt(dependant: Dependant) -> bool:
    for field in dependant.body_params:
        annotation = getattr(field.field_info, "annotation", None)
        if (
            isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
            and "prompt" in annotation.model_fields
        ):
            return True
    return False


@attr.s
class AISearchExtension(ApiExtension):
    """AI-Assisted Search extension (orchestrator).

    Registers ``GET /ai-search`` and, when `prompt` fragments are composed
    into the application's request models, transparently handles `prompt` on
    ``/search`` (GET+POST), ``/collections`` (GET, and POST when a
    collection-search POST endpoint exists) and
    ``/collections/{collection_id}/items`` (GET).

    Should be listed **last** in ``StacApi(extensions=[...])`` so routes
    registered by other extensions exist when it wraps them.
    """

    client: Union[AsyncBaseCoreClient, BaseCoreClient] = attr.ib()
    settings: AISearchSettings = attr.ib(factory=AISearchSettings)
    #: the backend's composed POST /search model; item searches run through it
    search_post_request_model: Type[BaseModel] = attr.ib(default=BaseSearchPostRequest)
    #: the backend's composed GET /collections model (gates collection params)
    collections_get_request_model: Optional[Type[APIRequest]] = attr.ib(default=None)
    #: custom `Translator`; None builds the pydantic-ai default from settings
    translator: Optional[Translator] = attr.ib(default=None)
    #: the backend's filters client (duck-typed `get_queryables(...)`); when
    #: set, AI-derived property filters are grounded against /queryables
    queryables_client: Any = attr.ib(default=None)
    GET: Type[APIRequest] = attr.ib(default=AISearchGetRequest)
    router: APIRouter = attr.ib(factory=APIRouter)

    conformance_classes: List[str] = attr.ib(
        factory=lambda: [AISearchConformanceClasses.ENDPOINT.value]
    )
    schema_href: Optional[str] = attr.ib(default=None)

    def __attrs_post_init__(self) -> None:
        """Initialize runtime state (caches, limiter, capability memo)."""
        self._translation_cache = TTLCache(
            maxsize=self.settings.translation_cache_size,
            ttl=self.settings.translation_cache_ttl,
        )
        self._catalog = CollectionCatalog(
            self.client,
            ttl=self.settings.catalog_ttl,
            max_candidates=self.settings.catalog_max_candidates,
        )
        self._limiter = AISearchRateLimiter(
            rate=self.settings.rate_limit,
            storage_uri=self.settings.rate_limit_storage_uri,
        )
        self._queryables_cache = TTLCache(maxsize=2, ttl=self.settings.catalog_ttl)
        self._capabilities_memo: Optional[SearchCapabilities] = None
        self._default_translator: Optional[Translator] = None

    # --- registration -----------------------------------------------------

    def register(self, app: FastAPI) -> None:
        """Register /ai-search and wrap prompt-capable routes."""
        self.router.prefix = app.state.router_prefix

        self.router.add_api_route(
            name="AI Search",
            path="/ai-search",
            methods=["GET"],
            endpoint=create_async_endpoint(self._ai_search_endpoint, self.GET),
            response_model=None,
            responses={
                200: {
                    "description": (
                        "Native STAC payload: ItemCollection, Collections, or a "
                        "combined response, depending on target_hint."
                    ),
                    "content": {"application/geo+json": {}, "application/json": {}},
                },
                **_ERROR_RESPONSES,
            },
        )
        app.include_router(self.router, tags=["AI Search Extension"])

        if self.settings.enable_prompt_on_endpoints:
            self._wrap_prompt_routes(app)

    def _wrap_prompt_routes(self, app: FastAPI) -> None:
        prefix = app.state.router_prefix or ""
        wrapped = []
        for route in _iter_api_routes(app.router.routes):
            for path, method, kind in _PROMPT_ROUTE_TARGETS:
                if route.path != f"{prefix}{path}" or method not in (
                    route.methods or set()
                ):
                    continue
                if method == "GET" and not _dependant_has_query_prompt(route.dependant):
                    continue
                if method == "POST" and not _dependant_has_body_prompt(route.dependant):
                    continue
                if self._wrap_route(route, kind):
                    wrapped.append(f"{method} {path}")

        if wrapped:
            logger.info("AI search prompt handling enabled on: %s", ", ".join(wrapped))
        else:
            logger.info(
                "No prompt-capable routes found to wrap; compose "
                "AISearchPromptExtension into the request models to enable "
                "prompt on existing endpoints."
            )

    def _wrap_route(self, route: APIRoute, kind: EndpointKind) -> bool:
        original = route.endpoint
        if getattr(original, "_ai_search_wrapped", False):
            return False

        response_class = route.response_class
        if isinstance(response_class, DefaultPlaceholder):
            response_class = response_class.value

        @functools.wraps(original)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request: Optional[Request] = kwargs.get("request")
            request_data = kwargs.get("request_data")
            prompt = getattr(request_data, "prompt", None)
            if not prompt or request is None:
                return await original(*args, **kwargs)
            try:
                return await self._handle_prompt(
                    original, kind, request, request_data, kwargs, response_class
                )
            except AISearchError as exc:
                raise exc.to_http() from exc

        wrapper._ai_search_wrapped = True  # type: ignore[attr-defined]
        route.endpoint = wrapper
        route.dependant = get_dependant(path=route.path_format, call=wrapper)
        route.app = request_response(route.get_route_handler())
        return True

    # --- prompt handling on wrapped routes ---------------------------------

    async def _handle_prompt(
        self,
        original: Callable,
        kind: EndpointKind,
        request: Request,
        request_data: Any,
        kwargs: Dict[str, Any],
        response_class: Type[Response],
    ) -> Any:
        prompt = validate_prompt(
            getattr(request_data, "prompt"), self.settings.max_prompt_length
        )
        await self._limiter.check(request)
        capabilities = self._capabilities()

        collection_id = (
            request.path_params.get("collection_id")
            if kind is EndpointKind.FEATURES_ITEMS
            else None
        )
        target = (
            AISearchTarget.COLLECTIONS
            if kind is EndpointKind.COLLECTION_SEARCH
            else AISearchTarget.ITEMS
        )
        translation = await self._translate(prompt, target, collection_id, request)

        if kind is EndpointKind.COLLECTION_SEARCH:
            ai_body = to_collection_search_params(translation, capabilities)
        else:
            ai_body = to_search_body(
                translation, capabilities, collection_id=collection_id
            )

        strategy = self.settings.filter_merge_strategy
        if isinstance(request_data, BaseModel):
            request_data, applied = apply_to_post_request(
                request_data,
                ai_body,
                post_user_wire_keys(request_data),
                filter_merge_strategy=strategy,
            )
            kwargs = {**kwargs, "request_data": request_data}
        else:
            applied = apply_to_get_request(
                request_data,
                ai_body,
                get_user_wire_keys(request),
                filter_merge_strategy=strategy,
            )

        result = await original(**kwargs)

        if applied and self.settings.echo_parameters and isinstance(result, dict):
            content = dict(result)
            content["parameters"] = applied
            # Returned as a Response so a route-level response_model can't
            # strip the extension member.
            return response_class(content)
        return result

    # --- /ai-search --------------------------------------------------------

    async def _ai_search_endpoint(
        self,
        request: Request = None,  # type: ignore[assignment]
        prompt: str = "",
        target_hint: Optional[str] = None,
        **kwargs: Any,
    ) -> Response:
        """GET /ai-search handler."""
        try:
            return await self._run_ai_search(request, prompt, target_hint)
        except AISearchError as exc:
            raise exc.to_http() from exc

    async def _run_ai_search(
        self, request: Request, prompt: str, target_hint: Optional[str]
    ) -> Response:
        clean = validate_prompt(prompt, self.settings.max_prompt_length)
        await self._limiter.check(request)
        capabilities = self._capabilities()
        target = AISearchTarget(target_hint) if target_hint else AISearchTarget.COMBINED

        translation = await self._translate(clean, target, None, request)
        parameters = to_search_body(translation, capabilities)

        if target is AISearchTarget.ITEMS:
            items = await self._items_search(translation, capabilities, request)
            return GeoJSONResponse(
                self._assemble(
                    request,
                    items,
                    parameters,
                    geojson=True,
                    extra_links=self._continuation_links(
                        items.get("links", []),
                        target,
                        parameters,
                        translation,
                        capabilities,
                        request,
                    ),
                )
            )

        if target is AISearchTarget.COLLECTIONS:
            collections = await self._collections_search(
                translation, capabilities, request
            )
            return JSONResponse(
                self._assemble(
                    request,
                    collections,
                    parameters,
                    geojson=False,
                    extra_links=self._continuation_links(
                        collections.get("links", []),
                        target,
                        parameters,
                        translation,
                        capabilities,
                        request,
                    ),
                )
            )

        items, collections = await asyncio.gather(
            self._items_search(translation, capabilities, request),
            self._collections_search(translation, capabilities, request),
        )
        combined = {
            "type": "FeatureCollection",
            "collections": collections.get("collections", []),
            "features": items.get("features", []),
        }
        # Combined-mode paging is an open spec question: no continuation links.
        return GeoJSONResponse(
            self._assemble(request, combined, parameters, geojson=True)
        )

    def _continuation_links(
        self,
        sub_links: List[Any],
        target: AISearchTarget,
        parameters: Dict[str, Any],
        translation: TranslationResult,
        capabilities: SearchCapabilities,
        request: Request,
    ) -> List[Dict[str, Any]]:
        """Synthesize standard STAC continuation links for `/ai-search`.

        The backend's own paging links point back at `/ai-search` (where the
        cursor parameter would be dropped), so continuation is rebuilt against
        the normal search endpoints: the translated parameters plus the
        relayed opaque cursor form a complete, prompt-free request that pages
        with no further AI involvement.
        """
        links: List[Dict[str, Any]] = []
        base = str(request.base_url)
        for link in sub_links:
            if not isinstance(link, dict) or link.get("rel") not in _CONTINUATION_RELS:
                continue
            relay = _relay_params(link)
            if not relay:  # no cursor survived — nothing to continue
                continue
            if target is AISearchTarget.ITEMS:
                links.append(
                    {
                        "rel": link["rel"],
                        "type": "application/geo+json",
                        "method": "POST",
                        "href": urljoin(base, "search"),
                        "body": {**parameters, **relay},
                    }
                )
            else:
                params = {
                    **to_collection_search_params(translation, capabilities),
                    **relay,
                }
                query = urlencode({k: _query_value(v) for k, v in params.items()})
                links.append(
                    {
                        "rel": link["rel"],
                        "type": "application/json",
                        "method": "GET",
                        "href": urljoin(base, "collections") + "?" + query,
                    }
                )
        return links

    def _assemble(
        self,
        request: Request,
        content: Dict[str, Any],
        parameters: Dict[str, Any],
        *,
        geojson: bool,
        extra_links: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        content = dict(content)
        media_type = "application/geo+json" if geojson else "application/json"
        content["links"] = [
            {"rel": "self", "type": media_type, "href": str(request.url)},
            {"rel": "root", "type": "application/json", "href": str(request.base_url)},
            *(extra_links or []),
        ]
        if self.settings.echo_parameters:
            content["parameters"] = parameters
        return content

    async def _items_search(
        self,
        translation: TranslationResult,
        capabilities: SearchCapabilities,
        request: Request,
    ) -> Dict[str, Any]:
        body = to_search_body(translation, capabilities)
        try:
            search_request = self.search_post_request_model.model_validate(body)
        except Exception as exc:
            raise TranslationFailedError(
                "AI-derived parameters did not form a valid search."
            ) from exc
        return await call_client_method(
            self.client.post_search, search_request, request=request
        )

    def _collections_param_fields(self) -> Set[str]:
        if self.collections_get_request_model is not None:
            return {
                field.name for field in attr.fields(self.collections_get_request_model)
            }
        if self._capabilities().collection_search:
            return {"bbox", "datetime", "limit", "q"}
        return set()

    async def _collections_search(
        self,
        translation: TranslationResult,
        capabilities: SearchCapabilities,
        request: Request,
    ) -> Dict[str, Any]:
        params = to_collection_search_params(translation, capabilities)
        allowed_fields = self._collections_param_fields()
        call_kwargs = {}
        for wire_key, value in params.items():
            field = WIRE_TO_FIELD.get(wire_key, wire_key)
            if field in allowed_fields:
                call_kwargs[field] = convert_for_get(wire_key, value)
        return await call_client_method(
            self.client.all_collections, request=request, **call_kwargs
        )

    # --- translation -------------------------------------------------------

    async def _get_queryables(self, request: Request) -> Optional[Set[str]]:
        """Queryable property names from the backend's filters client.

        Fail-soft: None (allowlist-only validation) when no client is
        configured or the fetch fails.
        """
        if self.queryables_client is None:
            return None

        async def fetch() -> Set[str]:
            schema = await call_client_method(
                self.queryables_client.get_queryables, request=request
            )
            return set((schema or {}).get("properties", {}).keys())

        try:
            return await self._queryables_cache.get_or_create("queryables", fetch)
        except Exception:
            logger.warning(
                "Queryables fetch failed; property filters validate against the "
                "built-in allowlist only",
                exc_info=True,
            )
            return None

    def _capabilities(self) -> SearchCapabilities:
        if self._capabilities_memo is None:
            try:
                # conformance lists may mix str and str-Enum members; use values
                classes = [
                    getattr(c, "value", c) for c in self.client.conformance_classes()
                ]
            except Exception:
                logger.warning(
                    "Could not read conformance classes from the client",
                    exc_info=True,
                )
                classes = []
            self._capabilities_memo = SearchCapabilities.from_conformance(classes)
        return self._capabilities_memo

    def _get_translator(self) -> Translator:
        if self.translator is not None:
            return self.translator
        if self._default_translator is not None:
            return self._default_translator

        if not self.settings.model:
            raise ProviderConfigError(
                "AI provider settings are missing on the server "
                "(set STAC_FASTAPI_AI_SEARCH_MODEL)."
            )
        try:
            from .pydantic_ai_translator import PydanticAiTranslator
        except ImportError as exc:
            raise map_translation_exception(exc) from exc

        api_key = (
            self.settings.api_key.get_secret_value() if self.settings.api_key else None
        )
        try:
            self._default_translator = PydanticAiTranslator(
                model=self.settings.model,
                api_key=api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.request_timeout,
            )
        except Exception as exc:
            raise ProviderConfigError("AI provider configuration is invalid.") from exc
        return self._default_translator

    async def _translate(
        self,
        prompt: str,
        target: AISearchTarget,
        collection_id: Optional[str],
        request: Request,
    ) -> TranslationResult:
        capabilities = self._capabilities()

        candidates = None
        if (
            self.settings.resolve_collections
            and collection_id is None
            and target is not AISearchTarget.COLLECTIONS
        ):
            candidates = await self._catalog.get(request)

        queryables = await self._get_queryables(request) if capabilities.filter else None

        cache_key = (
            hashlib.sha256(prompt.encode()).hexdigest(),
            target.value,
            collection_id or "",
            capabilities.fingerprint(),
            self._catalog.version() if candidates is not None else "-",
            hashlib.sha256("\n".join(sorted(queryables)).encode()).hexdigest()[:16]
            if queryables
            else "-",
        )

        async def factory() -> TranslationResult:
            translator = self._get_translator()
            translation_request = TranslationRequest(
                prompt=prompt,
                target=target,
                collection_id=collection_id,
                candidates=(
                    candidates
                    if candidates
                    and len(candidates) <= self.settings.inline_candidates_max
                    else None
                ),
                queryables=sorted(queryables)[:100] if queryables else None,
                capabilities=capabilities,
            )
            try:
                result = await translator.translate(translation_request)
            except Exception as exc:
                raise map_translation_exception(exc) from exc

            result = validate_translation(result, candidates or None, queryables)

            if (
                result.platform_keyword
                and not result.collections
                and candidates
                and len(candidates) > self.settings.inline_candidates_max
            ):
                selected = await translator.select_collections(prompt, candidates)
                if selected:
                    result = result.model_copy(update={"collections": selected})
            return result

        return await self._translation_cache.get_or_create(cache_key, factory)
