"""Local MCP server exposing guarded search, knowledge-base, and database tools."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from core.config import Settings, get_settings
from rag.pipeline import create_rag_pipeline
from tools.factory import create_tool_runtime
from tools.registry import ToolContext


def create_mcp_server(settings: Settings | None = None) -> FastMCP:
    """Build a stdio/HTTP-capable MCP server from application configuration."""

    runtime_settings = settings or get_settings()
    rag_pipeline = (
        create_rag_pipeline(runtime_settings) if runtime_settings.rag_enabled else None
    )
    runtime = create_tool_runtime(runtime_settings, rag_pipeline=rag_pipeline)
    permissions = frozenset(
        permission.strip()
        for permission in runtime_settings.tool_permissions.split(",")
        if permission.strip()
    )
    server = FastMCP(
        "research-agent-tools",
        instructions=(
            "Guarded research tools. Database access is read-only and web search is quota limited."
        ),
    )

    def invoke(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = runtime.invoke(
            name,
            arguments,
            ToolContext(
                run_id="mcp-session",
                user_id="mcp-client",
                task_id=name,
                permissions=permissions,
            ),
        )
        return result.model_dump(mode="json")

    if runtime.registry.contains("web.search"):

        @server.tool(name="web.search")
        def web_search(
            query: str,
            max_results: int = 5,
            topic: Literal["general", "news", "finance"] = "general",
            search_depth: Literal[
                "basic", "advanced", "fast", "ultra-fast"
            ] = "advanced",
            include_domains: tuple[str, ...] = (),
            exclude_domains: tuple[str, ...] = (),
        ) -> dict[str, Any]:
            """Search public web pages through Tavily."""

            return invoke(
                "web.search",
                {
                    "query": query,
                    "max_results": max_results,
                    "topic": topic,
                    "search_depth": search_depth,
                    "include_domains": include_domains,
                    "exclude_domains": exclude_domains,
                },
            )

    if runtime.registry.contains("knowledge_base.search"):

        @server.tool(name="knowledge_base.search")
        def knowledge_base_search(query: str, top_k: int = 5) -> dict[str, Any]:
            """Search indexed knowledge-base chunks through hybrid RAG."""

            return invoke("knowledge_base.search", {"query": query, "top_k": top_k})

    @server.tool(name="database.query")
    def database_query(
        sql: str,
        parameters: tuple[Any, ...] = (),
        max_rows: int = 100,
    ) -> dict[str, Any]:
        """Execute one bounded read-only PostgreSQL query."""

        return invoke(
            "database.query",
            {"sql": sql, "parameters": parameters, "max_rows": max_rows},
        )

    return server


def main() -> None:
    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
