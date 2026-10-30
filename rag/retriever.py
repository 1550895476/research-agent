"""Dense/sparse retrieval backends, source models, chunking, and RRF fusion."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from math import sqrt
from threading import RLock
from typing import Any, Protocol

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from rag.embedding import HybridEmbedding


class SourceType(StrEnum):
    """Origin class for retrieved content."""

    WEB = "web"
    KNOWLEDGE_BASE = "knowledge_base"
    DATABASE = "database"


class SourceDocument(BaseModel):
    """Unchunked source content accepted by the indexing pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    source_type: SourceType
    source_uri: str = Field(min_length=1)
    title: str = ""
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalDocument(BaseModel):
    """One retrievable chunk with provenance metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_type: SourceType
    source_uri: str = Field(min_length=1)
    title: str = ""
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexedDocument(BaseModel):
    """Chunk plus BGE-M3 dense/sparse representations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: RetrievalDocument
    embedding: HybridEmbedding


class RetrievalHit(BaseModel):
    """One result from an individual dense or sparse retriever."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: RetrievalDocument
    score: float


class FusedDocument(BaseModel):
    """Dense/sparse candidate merged by reciprocal-rank fusion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: RetrievalDocument
    rrf_score: float = Field(ge=0)
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


class HybridRetrievalBackend(Protocol):
    """Storage-neutral hybrid retrieval contract."""

    def upsert(self, documents: Sequence[IndexedDocument]) -> None: ...

    def dense_search(
        self,
        dense: tuple[float, ...],
        *,
        source_types: tuple[SourceType, ...],
        limit: int,
    ) -> list[RetrievalHit]: ...

    def sparse_search(
        self,
        sparse: dict[str, float],
        *,
        query_text: str,
        source_types: tuple[SourceType, ...],
        limit: int,
    ) -> list[RetrievalHit]: ...


class TextChunker:
    """Deterministic character chunker with overlap and stable identifiers."""

    def __init__(self, chunk_chars: int = 1200, overlap_chars: int = 150) -> None:
        if chunk_chars < 200:
            raise ValueError("chunk_chars must be at least 200")
        if overlap_chars < 0 or overlap_chars >= chunk_chars:
            raise ValueError("overlap_chars must be in [0, chunk_chars)")
        self._chunk_chars = chunk_chars
        self._overlap_chars = overlap_chars

    def chunk(self, source: SourceDocument) -> list[RetrievalDocument]:
        """Split source content while retaining source provenance on every chunk."""

        normalized = "\n".join(line.strip() for line in source.content.splitlines() if line.strip())
        chunks: list[RetrievalDocument] = []
        start = 0
        index = 0
        while start < len(normalized):
            end = min(len(normalized), start + self._chunk_chars)
            content = normalized[start:end]
            digest = sha256(
                f"{source.document_id}:{index}:{content}".encode()
            ).hexdigest()[:20]
            chunks.append(
                RetrievalDocument(
                    chunk_id=f"chunk:{digest}",
                    document_id=source.document_id,
                    source_type=source.source_type,
                    source_uri=source.source_uri,
                    title=source.title,
                    content=content,
                    metadata={**source.metadata, "chunk_index": index},
                )
            )
            if end == len(normalized):
                break
            start = end - self._overlap_chars
            index += 1
        return chunks


class InMemoryHybridRetriever:
    """Thread-safe exact dense/sparse retriever used for tests and small corpora."""

    def __init__(self) -> None:
        self._documents: dict[str, IndexedDocument] = {}
        self._lock = RLock()

    def upsert(self, documents: Sequence[IndexedDocument]) -> None:
        with self._lock:
            for document in documents:
                self._documents[document.document.chunk_id] = document

    def dense_search(
        self,
        dense: tuple[float, ...],
        *,
        source_types: tuple[SourceType, ...],
        limit: int,
    ) -> list[RetrievalHit]:
        allowed = set(source_types)
        with self._lock:
            hits = [
                RetrievalHit(
                    document=indexed.document,
                    score=_cosine_similarity(dense, indexed.embedding.dense),
                )
                for indexed in self._documents.values()
                if indexed.document.source_type in allowed
            ]
        hits.sort(key=lambda hit: (hit.score, hit.document.chunk_id), reverse=True)
        return hits[: max(0, limit)]

    def sparse_search(
        self,
        sparse: dict[str, float],
        *,
        query_text: str,
        source_types: tuple[SourceType, ...],
        limit: int,
    ) -> list[RetrievalHit]:
        del query_text
        allowed = set(source_types)
        with self._lock:
            hits = [
                RetrievalHit(
                    document=indexed.document,
                    score=sum(
                        weight * indexed.embedding.sparse.get(token, 0)
                        for token, weight in sparse.items()
                    ),
                )
                for indexed in self._documents.values()
                if indexed.document.source_type in allowed
            ]
        hits.sort(key=lambda hit: (hit.score, hit.document.chunk_id), reverse=True)
        return hits[: max(0, limit)]


class PostgresHybridRetriever:
    """pgvector dense search plus BGE lexical-weight sparse search."""

    def __init__(
        self,
        database_url: str,
        *,
        embedding_dimension: int = 1024,
        auto_setup: bool = True,
    ) -> None:
        self._database_url = database_url
        self._embedding_dimension = embedding_dimension
        if auto_setup:
            self.setup()

    def setup(self) -> None:
        dimension = int(self._embedding_dimension)
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                dense_embedding vector({dimension}) NOT NULL,
                sparse_embedding JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                search_vector TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('simple', COALESCE(title, '') || ' ' || content)
                ) STORED,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_source
            ON knowledge_chunks(source_type)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_dense
            ON knowledge_chunks USING hnsw (dense_embedding vector_cosine_ops)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_sparse
            ON knowledge_chunks USING gin (sparse_embedding)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_search
            ON knowledge_chunks USING gin (search_vector)
            """,
        ]
        with self._connect(register_vector_types=False) as connection:
            for statement in statements:
                connection.execute(statement)

    def upsert(self, documents: Sequence[IndexedDocument]) -> None:
        if not documents:
            return
        now = datetime.now(UTC)
        rows = []
        for indexed in documents:
            if len(indexed.embedding.dense) != self._embedding_dimension:
                raise ValueError("dense embedding dimension does not match PostgreSQL schema")
            document = indexed.document
            rows.append(
                (
                    document.chunk_id,
                    document.document_id,
                    document.source_type.value,
                    document.source_uri,
                    document.title,
                    document.content,
                    Jsonb(document.metadata),
                    Vector(indexed.embedding.dense),
                    Jsonb(indexed.embedding.sparse),
                    now,
                    now,
                )
            )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO knowledge_chunks (
                        chunk_id, document_id, source_type, source_uri, title, content,
                        metadata, dense_embedding, sparse_embedding, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source_type = EXCLUDED.source_type,
                        source_uri = EXCLUDED.source_uri,
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        dense_embedding = EXCLUDED.dense_embedding,
                        sparse_embedding = EXCLUDED.sparse_embedding,
                        updated_at = EXCLUDED.updated_at
                    """,
                    rows,
                )

    def dense_search(
        self,
        dense: tuple[float, ...],
        *,
        source_types: tuple[SourceType, ...],
        limit: int,
    ) -> list[RetrievalHit]:
        if len(dense) != self._embedding_dimension:
            raise ValueError("query embedding dimension does not match PostgreSQL schema")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *, 1 - (dense_embedding <=> %s) AS retrieval_score
                FROM knowledge_chunks
                WHERE source_type = ANY(%s)
                ORDER BY dense_embedding <=> %s
                LIMIT %s
                """,
                (
                    Vector(dense),
                    [source_type.value for source_type in source_types],
                    Vector(dense),
                    max(0, limit),
                ),
            ).fetchall()
        return [_hit_from_row(row) for row in rows]

    def sparse_search(
        self,
        sparse: dict[str, float],
        *,
        query_text: str,
        source_types: tuple[SourceType, ...],
        limit: int,
    ) -> list[RetrievalHit]:
        source_values = [source_type.value for source_type in source_types]
        with self._connect() as connection:
            if sparse:
                rows = connection.execute(
                    """
                    WITH query_sparse AS (
                        SELECT key AS token, value::DOUBLE PRECISION AS weight
                        FROM jsonb_each_text(%s::jsonb)
                    )
                    SELECT chunks.*,
                           SUM(terms.value::DOUBLE PRECISION * query_sparse.weight)
                               AS retrieval_score
                    FROM knowledge_chunks AS chunks
                    JOIN LATERAL jsonb_each_text(chunks.sparse_embedding)
                        AS terms(token, value) ON TRUE
                    JOIN query_sparse ON query_sparse.token = terms.token
                    WHERE chunks.source_type = ANY(%s)
                    GROUP BY chunks.chunk_id
                    ORDER BY retrieval_score DESC
                    LIMIT %s
                    """,
                    (Jsonb(sparse), source_values, max(0, limit)),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *,
                           ts_rank_cd(
                               search_vector,
                               websearch_to_tsquery('simple', %s)
                           ) AS retrieval_score
                    FROM knowledge_chunks
                    WHERE source_type = ANY(%s)
                      AND search_vector @@ websearch_to_tsquery('simple', %s)
                    ORDER BY retrieval_score DESC
                    LIMIT %s
                    """,
                    (query_text, source_values, query_text, max(0, limit)),
                ).fetchall()
        return [_hit_from_row(row) for row in rows]

    def _connect(
        self,
        *,
        register_vector_types: bool = True,
    ) -> psycopg.Connection[Any]:
        connection = psycopg.connect(
            self._database_url,
            autocommit=True,
            row_factory=dict_row,
        )
        if register_vector_types:
            register_vector(connection)
        return connection


def reciprocal_rank_fusion(
    dense_hits: Sequence[RetrievalHit],
    sparse_hits: Sequence[RetrievalHit],
    *,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    limit: int | None = None,
) -> list[FusedDocument]:
    """Fuse rankings by chunk_id while retaining per-channel scores and ranks."""

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    records: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        record = records.setdefault(
            hit.document.chunk_id,
            {"document": hit.document, "rrf_score": 0.0},
        )
        record["rrf_score"] += dense_weight / (rrf_k + rank)
        record["dense_score"] = hit.score
        record["dense_rank"] = rank
    for rank, hit in enumerate(sparse_hits, start=1):
        record = records.setdefault(
            hit.document.chunk_id,
            {"document": hit.document, "rrf_score": 0.0},
        )
        record["rrf_score"] += sparse_weight / (rrf_k + rank)
        record["sparse_score"] = hit.score
        record["sparse_rank"] = rank
    fused = [FusedDocument(**record) for record in records.values()]
    fused.sort(key=lambda item: (item.rrf_score, item.document.chunk_id), reverse=True)
    return fused[:limit] if limit is not None else fused


def _hit_from_row(row: dict[str, Any]) -> RetrievalHit:
    return RetrievalHit(
        document=RetrievalDocument(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            source_type=SourceType(row["source_type"]),
            source_uri=row["source_uri"],
            title=row["title"],
            content=row["content"],
            metadata=row["metadata"],
        ),
        score=float(row["retrieval_score"]),
    )


def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values) or not left_values:
        return 0
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    left_norm = sqrt(sum(value * value for value in left_values))
    right_norm = sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0
    return numerator / (left_norm * right_norm)
