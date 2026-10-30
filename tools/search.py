"""Tavily-backed web search tool with normalized source evidence."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from tavily import TavilyClient

from rag.retriever import SourceType
from tools.registry import (
    StructuredTool,
    ToolContext,
    ToolEvidence,
    ToolResult,
    ToolRisk,
    ToolSpec,
)


class SearchInput(BaseModel):
    """Validated arguments accepted by the web search tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=20)
    topic: Literal["general", "news", "finance"] = "general"
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "advanced"
    include_domains: tuple[str, ...] = Field(default=(), max_length=20)
    exclude_domains: tuple[str, ...] = Field(default=(), max_length=20)
    include_url_prefixes: tuple[str, ...] = Field(default=(), max_length=20)


class SearchHit(BaseModel):
    """One normalized Tavily result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    url: str
    content: str
    score: float = Field(ge=0, le=1)


class SearchClient(Protocol):
    def search(self, **kwargs: Any) -> dict[str, Any]: ...


class TavilySearchTool(StructuredTool):
    """Search public web sources through Tavily/Search API."""

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        *,
        client: SearchClient | None = None,
    ) -> None:
        super().__init__(
            search_tool_spec(),
            SearchInput,
        )
        resolved_key = (
            api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        )
        if client is None and not resolved_key:
            raise ValueError("TAVILY_API_KEY is required to construct TavilySearchTool")
        self._client = client or TavilyClient(api_key=resolved_key)

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        request = SearchInput.model_validate(arguments)
        response = self._client.search(
            query=request.query,
            max_results=request.max_results,
            topic=request.topic,
            search_depth=request.search_depth,
            include_domains=list(request.include_domains),
            exclude_domains=list(request.exclude_domains),
            include_answer=False,
            include_raw_content=False,
        )
        hits = tuple(_normalize_hit(item) for item in response.get("results", []))
        if request.include_url_prefixes:
            hits = tuple(
                hit
                for hit in hits
                if _matches_url_prefix(hit.url, request.include_url_prefixes)
            )
        evidences = tuple(
            ToolEvidence(
                evidence_id=_evidence_id(hit.url, hit.content),
                source_type=SourceType.WEB,
                source_uri=hit.url,
                title=hit.title,
                content=hit.content,
                confidence=hit.score,
            )
            for hit in hits
        )
        return ToolResult(
            tool_name=self.spec.name,
            data={"query": request.query, "results": [hit.model_dump() for hit in hits]},
            evidences=evidences,
            metadata={
                "result_count": len(hits),
                "response_time": response.get("response_time"),
            },
        )


def _normalize_hit(item: dict[str, Any]) -> SearchHit:
    raw_score = float(item.get("score") or 0)
    return SearchHit(
        title=str(item.get("title") or "Untitled source"),
        url=str(item.get("url") or "unknown://source"),
        content=str(item.get("content") or item.get("raw_content") or ""),
        score=min(1.0, max(0.0, raw_score)),
    )


def _evidence_id(url: str, content: str) -> str:
    digest = sha256(f"{url}\n{content}".encode()).hexdigest()[:24]
    return f"evidence:web:{digest}"


def _matches_url_prefix(url: str, prefixes: tuple[str, ...]) -> bool:
    normalized_url = url.casefold().rstrip("/")
    for value in prefixes:
        prefix = value.strip().casefold().rstrip("/")
        if normalized_url == prefix or normalized_url.startswith(f"{prefix}/"):
            return True
    return False


def search_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="web.search",
        description=(
            "Search public web pages and return source URLs, snippets, and relevance scores."
        ),
        input_schema=SearchInput.model_json_schema(),
        capabilities=frozenset({"search", "web"}),
        required_permission="search:execute",
        risk=ToolRisk.LOW,
    )
