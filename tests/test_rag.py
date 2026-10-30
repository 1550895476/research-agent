"""Offline tests for the Step 5 hybrid RAG pipeline and Agent integration."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agents.researcher import ResearchOutput
from agents.runtime import AgentRuntime
from graph.state import AgentName, ResearchTask
from graph.workflow import TaskExecutionRequest
from rag.embedding import HybridEmbedding
from rag.pipeline import RAGPipeline
from rag.query_rewriter import HeuristicQueryRewriter, RewrittenQuery
from rag.reranker import (
    FusionScoreReranker,
    RerankCandidate,
    RerankScore,
)
from rag.retriever import (
    IndexedDocument,
    InMemoryHybridRetriever,
    RetrievalDocument,
    RetrievalHit,
    SourceDocument,
    SourceType,
    TextChunker,
    reciprocal_rank_fusion,
)


class FakeHybridEncoder:
    """Tiny deterministic encoder that avoids downloading BGE model weights."""

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        results = []
        for text in texts:
            normalized = text.lower()
            alpha = float("alpha" in normalized)
            beta = float("beta" in normalized)
            dense = (alpha or 0.1, beta or 0.1)
            sparse = {
                token: weight
                for token, weight in {"alpha": alpha, "beta": beta}.items()
                if weight
            }
            results.append(HybridEmbedding(dense=dense, sparse=sparse))
        return results


class FixedRewriter:
    def rewrite(self, query: str) -> RewrittenQuery:
        return RewrittenQuery(
            original_query=query,
            rewritten_query="alpha",
            search_queries=("alpha",),
            keywords=("alpha",),
        )


class PreferBetaReranker:
    def score(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankScore]:
        del query
        return [
            RerankScore(
                chunk_id=candidate.chunk_id,
                score=10.0 if "beta" in candidate.text.lower() else 0.0,
            )
            for candidate in candidates
        ]


class ResearchOnlyModel:
    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        assert messages
        if schema is ResearchOutput:
            return ResearchOutput(summary="retrieval analysis")
        raise AssertionError(f"unexpected schema: {schema.__name__}")


def _document(
    chunk_id: str,
    content: str,
    source_type: SourceType = SourceType.KNOWLEDGE_BASE,
) -> RetrievalDocument:
    return RetrievalDocument(
        chunk_id=chunk_id,
        document_id=f"doc:{chunk_id}",
        source_type=source_type,
        source_uri=f"https://example.test/{chunk_id}",
        title=chunk_id,
        content=content,
    )


def _indexed(
    chunk_id: str,
    content: str,
    embedding: HybridEmbedding,
    source_type: SourceType = SourceType.KNOWLEDGE_BASE,
) -> IndexedDocument:
    return IndexedDocument(
        document=_document(chunk_id, content, source_type),
        embedding=embedding,
    )


def test_query_rewrite_normalizes_input_and_extracts_keywords() -> None:
    result = HeuristicQueryRewriter().rewrite("  Agent   platform  趋势  ")

    assert result.rewritten_query == "Agent platform 趋势"
    assert result.search_queries == ("Agent platform 趋势",)
    assert {"agent", "platform", "趋势"}.issubset(result.keywords)


def test_chunking_has_stable_ids_and_preserves_source_metadata() -> None:
    source = SourceDocument(
        document_id="market-report",
        source_type=SourceType.WEB,
        source_uri="https://example.test/report",
        title="Market report",
        content="A" * 450,
        metadata={"year": 2026},
    )
    chunker = TextChunker(chunk_chars=250, overlap_chars=50)

    first = chunker.chunk(source)
    second = chunker.chunk(source)

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert len(first) == 2
    assert all(item.source_type == SourceType.WEB for item in first)
    assert first[0].metadata == {"year": 2026, "chunk_index": 0}


def test_dense_and_sparse_retrieval_can_rank_different_documents() -> None:
    backend = InMemoryHybridRetriever()
    backend.upsert(
        [
            _indexed("dense-first", "semantic", HybridEmbedding(dense=(1, 0))),
            _indexed(
                "sparse-first",
                "lexical",
                HybridEmbedding(dense=(0, 1), sparse={"rare": 2}),
            ),
        ]
    )

    dense = backend.dense_search(
        (1, 0), source_types=tuple(SourceType), limit=2
    )
    sparse = backend.sparse_search(
        {"rare": 1}, query_text="rare", source_types=tuple(SourceType), limit=2
    )

    assert dense[0].document.chunk_id == "dense-first"
    assert sparse[0].document.chunk_id == "sparse-first"


def test_rrf_prioritizes_a_document_found_by_both_channels() -> None:
    shared = _document("shared", "shared")
    dense_only = _document("dense-only", "dense")
    sparse_only = _document("sparse-only", "sparse")

    fused = reciprocal_rank_fusion(
        [
            RetrievalHit(document=shared, score=0.9),
            RetrievalHit(document=dense_only, score=0.8),
        ],
        [
            RetrievalHit(document=shared, score=2.0),
            RetrievalHit(document=sparse_only, score=1.0),
        ],
    )

    assert fused[0].document.chunk_id == "shared"
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 1


def test_pipeline_reranks_candidates_and_honors_context_budget() -> None:
    backend = InMemoryHybridRetriever()
    backend.upsert(
        [
            _indexed(
                "alpha",
                "alpha semantic result",
                HybridEmbedding(dense=(1, 0), sparse={"alpha": 1}),
            ),
            _indexed(
                "beta",
                "beta cross encoder winner",
                HybridEmbedding(dense=(0, 1), sparse={"beta": 1}),
            ),
        ]
    )
    pipeline = RAGPipeline(
        rewriter=FixedRewriter(),
        embedding_model=FakeHybridEncoder(),
        backend=backend,
        reranker=PreferBetaReranker(),
        candidate_k=2,
        top_k=2,
        max_context_chars=100,
    )

    result = pipeline.retrieve("original question")

    assert result.documents[0].document.chunk_id == "beta"
    assert len(result.documents) <= 2
    assert len(result.context) <= 100
    assert result.citations[0].evidence_id == "evidence:beta"


def test_index_sources_supports_web_knowledge_base_and_database() -> None:
    backend = InMemoryHybridRetriever()
    pipeline = RAGPipeline(
        rewriter=FixedRewriter(),
        embedding_model=FakeHybridEncoder(),
        backend=backend,
        reranker=FusionScoreReranker(),
    )
    sources = [
        SourceDocument(
            document_id=source_type.value,
            source_type=source_type,
            source_uri=f"source://{source_type.value}",
            content=f"alpha content from {source_type.value}",
        )
        for source_type in SourceType
    ]

    chunk_ids = pipeline.index_sources(sources)
    hits = backend.dense_search(
        (1, 0.1), source_types=tuple(SourceType), limit=10
    )

    assert len(chunk_ids) == 3
    assert {hit.document.source_type for hit in hits} == set(SourceType)


def test_researcher_stores_rag_context_and_evidence_references() -> None:
    backend = InMemoryHybridRetriever()
    pipeline = RAGPipeline(
        rewriter=FixedRewriter(),
        embedding_model=FakeHybridEncoder(),
        backend=backend,
        reranker=FusionScoreReranker(),
        top_k=1,
    )
    pipeline.index_sources(
        [
            SourceDocument(
                document_id="source-1",
                source_type=SourceType.WEB,
                source_uri="https://example.test/source-1",
                title="Source one",
                content="alpha evidence",
            )
        ]
    )
    model = ResearchOnlyModel()
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        rag_pipeline=pipeline,
    )
    task = ResearchTask(
        task_id="retrieve",
        title="Retrieve alpha",
        description="Find alpha evidence",
        assigned_agent=AgentName.RESEARCHER,
    )

    result = runtime.executors()[AgentName.RESEARCHER](
        TaskExecutionRequest(
            user_goal="Research alpha",
            run_id="run-1",
            user_id="user-1",
            task=task,
        )
    )
    payload = runtime.blackboard.get_by_uri(result.payload_ref).payload

    assert result.evidence_ids == (payload.sources[0].evidence_id,)
    assert payload.sources[0].source_uri == "https://example.test/source-1"
    assert "alpha evidence" in payload.retrieved_context
