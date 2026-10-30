"""End-to-end query rewrite, hybrid retrieval, RRF, reranking, and context assembly."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, ConfigDict, Field

from agents.model import StructuredModel
from core.config import Settings
from rag.embedding import BGEM3EmbeddingModel, HybridEmbedding, HybridEmbeddingModel
from rag.query_rewriter import HeuristicQueryRewriter, LLMQueryRewriter, QueryRewriter
from rag.reranker import BGEReranker, RerankCandidate, Reranker
from rag.retriever import (
    FusedDocument,
    HybridRetrievalBackend,
    IndexedDocument,
    InMemoryHybridRetriever,
    PostgresHybridRetriever,
    RetrievalDocument,
    SourceDocument,
    SourceType,
    TextChunker,
    reciprocal_rank_fusion,
)


class RankedContextDocument(BaseModel):
    """Final reranked document plus stable evidence identifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    document: RetrievalDocument
    rrf_score: float = Field(ge=0)
    rerank_score: float


class RAGCitation(BaseModel):
    """Compact provenance record handed to downstream Agents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    chunk_id: str
    source_type: SourceType
    source_uri: str
    title: str


class RAGContext(BaseModel):
    """Top-K context and citations produced for one original query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_query: str
    rewritten_query: str
    context: str
    documents: tuple[RankedContextDocument, ...] = ()
    citations: tuple[RAGCitation, ...] = ()


class RAGPipeline:
    """Coordinate the complete multi-stage retrieval path."""

    def __init__(
        self,
        *,
        rewriter: QueryRewriter,
        embedding_model: HybridEmbeddingModel,
        backend: HybridRetrievalBackend,
        reranker: Reranker,
        chunker: TextChunker | None = None,
        candidate_k: int = 30,
        top_k: int = 8,
        rrf_k: int = 60,
        max_context_chars: int = 12000,
    ) -> None:
        self._rewriter = rewriter
        self._embedding_model = embedding_model
        self._backend = backend
        self._reranker = reranker
        self._chunker = chunker or TextChunker()
        self._candidate_k = candidate_k
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._max_context_chars = max_context_chars

    def index_sources(self, sources: Sequence[SourceDocument]) -> list[str]:
        """Chunk, BGE-M3 encode, and upsert multi-source documents."""

        chunks = [chunk for source in sources for chunk in self._chunker.chunk(source)]
        if not chunks:
            return []
        embeddings = self._embedding_model.encode([chunk.content for chunk in chunks])
        self._backend.upsert(
            [
                IndexedDocument(document=chunk, embedding=embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
        )
        return [chunk.chunk_id for chunk in chunks]

    def retrieve(
        self,
        query: str,
        *,
        source_types: tuple[SourceType, ...] = tuple(SourceType),
    ) -> RAGContext:
        """Run rewrite → hybrid retrieval → RRF → rerank → Top-K context."""

        rewritten = self._rewriter.rewrite(query)
        query_embeddings = self._embedding_model.encode(rewritten.search_queries)
        merged_embedding = _merge_embeddings(query_embeddings)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-retrieval") as executor:
            dense_future = executor.submit(
                self._backend.dense_search,
                merged_embedding.dense,
                source_types=source_types,
                limit=self._candidate_k,
            )
            sparse_future = executor.submit(
                self._backend.sparse_search,
                merged_embedding.sparse,
                query_text=rewritten.rewritten_query,
                source_types=source_types,
                limit=self._candidate_k,
            )
            dense_hits = dense_future.result()
            sparse_hits = sparse_future.result()

        fused = reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            rrf_k=self._rrf_k,
            limit=self._candidate_k,
        )
        rerank_scores = {
            score.chunk_id: score.score
            for score in self._reranker.score(
                rewritten.rewritten_query,
                [
                    RerankCandidate(
                        chunk_id=item.document.chunk_id,
                        text=item.document.content,
                        fusion_score=item.rrf_score,
                    )
                    for item in fused
                ],
            )
        }
        reranked = sorted(
            fused,
            key=lambda item: (
                rerank_scores.get(item.document.chunk_id, float("-inf")),
                item.rrf_score,
                item.document.chunk_id,
            ),
            reverse=True,
        )
        ranked = [
            _to_ranked(item, rerank_scores.get(item.document.chunk_id, item.rrf_score))
            for item in reranked[: self._top_k]
        ]
        selected, context = _build_context(ranked, self._max_context_chars)
        citations = tuple(
            RAGCitation(
                evidence_id=item.evidence_id,
                chunk_id=item.document.chunk_id,
                source_type=item.document.source_type,
                source_uri=item.document.source_uri,
                title=item.document.title,
            )
            for item in selected
        )
        return RAGContext(
            original_query=rewritten.original_query,
            rewritten_query=rewritten.rewritten_query,
            context=context,
            documents=tuple(selected),
            citations=citations,
        )


def create_rag_pipeline(
    settings: Settings,
    *,
    rewrite_model: StructuredModel | None = None,
) -> RAGPipeline:
    """Create a lazily loaded BGE pipeline using configured storage."""

    backend: HybridRetrievalBackend
    if settings.rag_backend == "postgres":
        backend = PostgresHybridRetriever(
            settings.database_url,
            embedding_dimension=settings.embedding_dimension,
            auto_setup=True,
        )
    else:
        backend = InMemoryHybridRetriever()
    rewriter: QueryRewriter = (
        LLMQueryRewriter(rewrite_model) if rewrite_model else HeuristicQueryRewriter()
    )
    return RAGPipeline(
        rewriter=rewriter,
        embedding_model=BGEM3EmbeddingModel(
            settings.bge_m3_model,
            device=settings.bge_device,
            use_fp16=settings.bge_use_fp16,
            expected_dimension=settings.embedding_dimension,
        ),
        backend=backend,
        reranker=BGEReranker(
            settings.bge_reranker_model,
            device=settings.bge_device,
            use_fp16=settings.bge_use_fp16,
        ),
        candidate_k=settings.rag_candidate_k,
        top_k=settings.rag_top_k,
        rrf_k=settings.rag_rrf_k,
        max_context_chars=settings.rag_context_chars,
    )


def _merge_embeddings(embeddings: Sequence[HybridEmbedding]) -> HybridEmbedding:
    if not embeddings:
        raise ValueError("query rewrite produced no embeddings")
    dimension = len(embeddings[0].dense)
    if any(len(embedding.dense) != dimension for embedding in embeddings):
        raise ValueError("query embeddings have inconsistent dimensions")
    count = len(embeddings)
    dense = tuple(
        sum(embedding.dense[index] for embedding in embeddings) / count
        for index in range(dimension)
    )
    sparse: dict[str, float] = {}
    for embedding in embeddings:
        for token, weight in embedding.sparse.items():
            sparse[token] = max(sparse.get(token, 0), weight)
    return HybridEmbedding(dense=dense, sparse=sparse)


def _to_ranked(item: FusedDocument, rerank_score: float) -> RankedContextDocument:
    return RankedContextDocument(
        evidence_id=f"evidence:{item.document.chunk_id}",
        document=item.document,
        rrf_score=item.rrf_score,
        rerank_score=rerank_score,
    )


def _build_context(
    ranked: Sequence[RankedContextDocument],
    max_chars: int,
) -> tuple[list[RankedContextDocument], str]:
    selected: list[RankedContextDocument] = []
    sections: list[str] = []
    used = 0
    for item in ranked:
        header = (
            f"[{item.evidence_id}] {item.document.title or item.document.document_id}\n"
            f"Source: {item.document.source_uri}\n"
        )
        remaining = max_chars - used - len(header)
        if remaining <= 0:
            break
        content = item.document.content[:remaining]
        section = f"{header}{content}"
        sections.append(section)
        selected.append(item)
        used += len(section) + 2
        if used >= max_chars:
            break
    return selected, "\n\n".join(sections)[:max_chars]
