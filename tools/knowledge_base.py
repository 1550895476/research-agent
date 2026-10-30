"""Knowledge-base retrieval tool backed by the Step 5 RAG pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag.pipeline import RAGPipeline
from rag.retriever import SourceType
from tools.registry import (
    StructuredTool,
    ToolContext,
    ToolEvidence,
    ToolResult,
    ToolRisk,
    ToolSpec,
)


class KnowledgeBaseSearchInput(BaseModel):
    """Validated semantic/lexical knowledge-base search arguments."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeBaseSearchTool(StructuredTool):
    """Retrieve already indexed knowledge-base chunks with citations."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        super().__init__(
            knowledge_base_tool_spec(),
            KnowledgeBaseSearchInput,
        )
        self._pipeline = pipeline

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        request = KnowledgeBaseSearchInput.model_validate(arguments)
        retrieved = self._pipeline.retrieve(
            request.query,
            source_types=(SourceType.KNOWLEDGE_BASE,),
        )
        selected = retrieved.documents[: request.top_k]
        evidences = tuple(
            ToolEvidence(
                evidence_id=item.evidence_id,
                source_type=item.document.source_type,
                source_uri=item.document.source_uri,
                title=item.document.title,
                content=item.document.content,
                confidence=min(1.0, max(0.0, item.rerank_score)),
            )
            for item in selected
        )
        return ToolResult(
            tool_name=self.spec.name,
            data={
                "query": retrieved.rewritten_query,
                "results": [item.model_dump(mode="json") for item in selected],
            },
            evidences=evidences,
            metadata={"result_count": len(selected)},
        )


def knowledge_base_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="knowledge_base.search",
        description=(
            "Run hybrid semantic and lexical retrieval over indexed knowledge-base chunks."
        ),
        input_schema=KnowledgeBaseSearchInput.model_json_schema(),
        capabilities=frozenset({"search", "knowledge_base"}),
        required_permission="knowledge_base:read",
        risk=ToolRisk.LOW,
    )
