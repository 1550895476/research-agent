"""Application factory for lazily loaded, guarded local tools."""

from __future__ import annotations

from core.config import Settings
from guardrail.middleware import GuardedToolRuntime, GuardrailMiddleware
from rag.pipeline import RAGPipeline
from tools.database import PostgresQueryTool, database_tool_spec
from tools.knowledge_base import KnowledgeBaseSearchTool, knowledge_base_tool_spec
from tools.registry import ToolRegistry
from tools.search import TavilySearchTool, search_tool_spec


def create_tool_registry(
    settings: Settings,
    *,
    rag_pipeline: RAGPipeline | None = None,
) -> ToolRegistry:
    """Register configured tools while deferring construction until selected."""

    registry = ToolRegistry()
    if settings.tavily_api_key is not None:
        registry.register_lazy(
            search_tool_spec(),
            lambda: TavilySearchTool(settings.tavily_api_key),
        )
    if rag_pipeline is not None:
        registry.register_lazy(
            knowledge_base_tool_spec(),
            lambda: KnowledgeBaseSearchTool(rag_pipeline),
        )
    if settings.database_tool_enabled:
        registry.register_lazy(
            database_tool_spec(),
            lambda: PostgresQueryTool(
                settings.database_url,
                connect_timeout_seconds=settings.database_connect_timeout_seconds,
                statement_timeout_ms=settings.database_statement_timeout_ms,
                max_rows=settings.database_max_rows,
            ),
        )
    return registry


def create_tool_runtime(
    settings: Settings,
    *,
    rag_pipeline: RAGPipeline | None = None,
) -> GuardedToolRuntime:
    registry = create_tool_registry(settings, rag_pipeline=rag_pipeline)
    guardrail = GuardrailMiddleware(
        max_web_search_calls_per_run=settings.search_max_calls_per_run,
        max_web_search_calls_per_task=settings.search_max_calls_per_task,
    )
    return GuardedToolRuntime(registry, guardrail)
