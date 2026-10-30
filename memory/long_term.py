"""Long-term user, research-history, and evidence memory with pgvector support."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from math import sqrt
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from core.config import Settings


class MemoryKind(StrEnum):
    """Durable memory categories with different downstream consumers."""

    USER_PREFERENCE = "user_preference"
    RESEARCH_HISTORY = "research_history"
    HIGH_VALUE_EVIDENCE = "high_value_evidence"


class MemoryWrite(BaseModel):
    """Input for inserting or updating one long-term memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = Field(min_length=1)
    kind: MemoryKind
    content: str = Field(min_length=1)
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: tuple[float, ...] | None = None
    quality: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.5, ge=0, le=1)
    reuse_signal: float = Field(default=0, ge=0, le=1)
    explicit_importance: float = Field(default=0, ge=0, le=1)
    expires_at: datetime | None = None


class MemoryRecord(BaseModel):
    """Stored memory plus lifecycle and value metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    user_id: str
    kind: MemoryKind
    content: str
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: tuple[float, ...] | None = None
    value_score: float = Field(ge=0, le=1)
    access_count: int = Field(default=0, ge=0)
    compressed: bool = False
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None


class MemoryEvictionResult(BaseModel):
    """Identifiers removed by value/expiry/capacity lifecycle rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evicted_ids: tuple[str, ...]


class MemoryValueEvaluator:
    """Score memories using quality, confidence, reuse, and explicit importance."""

    def evaluate(self, memory: MemoryWrite) -> float:
        """Return a stable value score in the inclusive [0, 1] interval."""

        score = (
            0.35 * memory.quality
            + 0.35 * memory.confidence
            + 0.15 * memory.reuse_signal
            + 0.15 * memory.explicit_importance
        )
        return round(max(0.0, min(1.0, score)), 6)


class MemoryCompressor:
    """Deterministically compact oversized memories before persistence."""

    def __init__(self, max_chars: int = 8000) -> None:
        if max_chars < 500:
            raise ValueError("max_chars must be at least 500")
        self._max_chars = max_chars

    def compress(self, content: str, summary: str | None = None) -> tuple[str, str, bool]:
        """Normalize whitespace and retain both leading and trailing high-signal text."""

        normalized = " ".join(content.split())
        generated_summary = " ".join((summary or normalized[:500]).split())
        if len(normalized) <= self._max_chars:
            return normalized, generated_summary, False
        head_size = int(self._max_chars * 0.75)
        tail_size = self._max_chars - head_size
        compressed = f"{normalized[:head_size]} …[compressed]… {normalized[-tail_size:]}"
        return compressed, generated_summary, True


class LongTermMemory(Protocol):
    """Storage-neutral contract consumed by Planner and workflow finalization."""

    def upsert(self, memory: MemoryWrite) -> MemoryRecord: ...

    def search(
        self,
        user_id: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        query_embedding: tuple[float, ...] | None = None,
        limit: int = 10,
        min_value_score: float = 0,
    ) -> list[MemoryRecord]: ...

    def record_access(self, memory_id: str) -> None: ...

    def delete(self, memory_id: str) -> None: ...

    def evict(self) -> MemoryEvictionResult: ...


class InMemoryLongTermMemory:
    """Thread-safe long-term memory used in tests and single-process development."""

    def __init__(
        self,
        *,
        compressor: MemoryCompressor | None = None,
        evaluator: MemoryValueEvaluator | None = None,
        min_value_score: float = 0.2,
        max_records_per_user: int = 1000,
    ) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._compressor = compressor or MemoryCompressor()
        self._evaluator = evaluator or MemoryValueEvaluator()
        self._min_value_score = min_value_score
        self._max_records_per_user = max_records_per_user
        self._lock = RLock()

    def upsert(self, memory: MemoryWrite) -> MemoryRecord:
        content, summary, compressed = self._compressor.compress(
            memory.content,
            memory.summary,
        )
        now = datetime.now(UTC)
        with self._lock:
            previous = self._records.get(memory.memory_id)
            record = MemoryRecord(
                memory_id=memory.memory_id,
                user_id=memory.user_id,
                kind=memory.kind,
                content=content,
                summary=summary,
                metadata=memory.metadata,
                embedding=memory.embedding,
                value_score=self._evaluator.evaluate(memory),
                access_count=previous.access_count if previous else 0,
                compressed=compressed,
                created_at=previous.created_at if previous else now,
                updated_at=now,
                expires_at=memory.expires_at,
            )
            self._records[memory.memory_id] = record
            return record

    def search(
        self,
        user_id: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        query_embedding: tuple[float, ...] | None = None,
        limit: int = 10,
        min_value_score: float = 0,
    ) -> list[MemoryRecord]:
        allowed_kinds = set(kinds or MemoryKind)
        now = datetime.now(UTC)
        with self._lock:
            candidates = [
                record
                for record in self._records.values()
                if record.user_id == user_id
                and record.kind in allowed_kinds
                and record.value_score >= min_value_score
                and (record.expires_at is None or record.expires_at > now)
            ]
            if query_embedding is None:
                candidates.sort(key=lambda item: (item.value_score, item.updated_at), reverse=True)
            else:
                candidates.sort(
                    key=lambda item: (
                        0.65 * _cosine_similarity(query_embedding, item.embedding)
                        + 0.35 * item.value_score
                    ),
                    reverse=True,
                )
            selected = candidates[: max(0, limit)]
            for record in selected:
                self.record_access(record.memory_id)
            return [self._records[record.memory_id] for record in selected]

    def record_access(self, memory_id: str) -> None:
        with self._lock:
            record = self._records.get(memory_id)
            if record is None:
                return
            self._records[memory_id] = record.model_copy(
                update={
                    "access_count": record.access_count + 1,
                    "value_score": min(1.0, record.value_score + 0.02),
                    "updated_at": datetime.now(UTC),
                }
            )

    def delete(self, memory_id: str) -> None:
        """Idempotently remove one memory for SAGA compensation."""

        with self._lock:
            self._records.pop(memory_id, None)

    def evict(self) -> MemoryEvictionResult:
        now = datetime.now(UTC)
        evicted: list[str] = []
        with self._lock:
            for memory_id, record in list(self._records.items()):
                expired = record.expires_at is not None and record.expires_at <= now
                if expired or record.value_score < self._min_value_score:
                    evicted.append(memory_id)
                    del self._records[memory_id]

            by_user: dict[str, list[MemoryRecord]] = defaultdict(list)
            for record in self._records.values():
                by_user[record.user_id].append(record)
            for records in by_user.values():
                records.sort(key=lambda item: (item.value_score, item.updated_at), reverse=True)
                for record in records[self._max_records_per_user :]:
                    evicted.append(record.memory_id)
                    del self._records[record.memory_id]
        return MemoryEvictionResult(evicted_ids=tuple(evicted))


class PostgresLongTermMemory:
    """PostgreSQL + pgvector implementation of LongTermMemory."""

    def __init__(
        self,
        database_url: str,
        *,
        embedding_dimension: int = 1024,
        compressor: MemoryCompressor | None = None,
        evaluator: MemoryValueEvaluator | None = None,
        min_value_score: float = 0.2,
        max_records_per_user: int = 1000,
        auto_setup: bool = True,
    ) -> None:
        self._database_url = database_url
        self._embedding_dimension = embedding_dimension
        self._compressor = compressor or MemoryCompressor()
        self._evaluator = evaluator or MemoryValueEvaluator()
        self._min_value_score = min_value_score
        self._max_records_per_user = max_records_per_user
        if auto_setup:
            self.setup()

    def setup(self) -> None:
        """Create the extension, table, and indexes idempotently."""

        dimension = int(self._embedding_dimension)
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
            CREATE TABLE IF NOT EXISTS research_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dimension}),
                value_score DOUBLE PRECISION NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                compressed BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                expires_at TIMESTAMPTZ
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memories_user_type
            ON research_memories(user_id, memory_type)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memories_value
            ON research_memories(user_id, value_score DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memories_embedding
            ON research_memories USING hnsw (embedding vector_cosine_ops)
            """,
        ]
        with self._connect(register_vector_types=False) as connection:
            for statement in statements:
                connection.execute(statement)

    def upsert(self, memory: MemoryWrite) -> MemoryRecord:
        content, summary, compressed = self._compressor.compress(
            memory.content,
            memory.summary,
        )
        now = datetime.now(UTC)
        embedding = Vector(memory.embedding) if memory.embedding is not None else None
        with self._connect() as connection:
            row = connection.execute(
                """
                INSERT INTO research_memories (
                    memory_id, user_id, memory_type, content, summary, metadata, embedding,
                    value_score, compressed, created_at, updated_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (memory_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    memory_type = EXCLUDED.memory_type,
                    content = EXCLUDED.content,
                    summary = EXCLUDED.summary,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    value_score = EXCLUDED.value_score,
                    compressed = EXCLUDED.compressed,
                    updated_at = EXCLUDED.updated_at,
                    expires_at = EXCLUDED.expires_at
                RETURNING *
                """,
                (
                    memory.memory_id,
                    memory.user_id,
                    memory.kind.value,
                    content,
                    summary,
                    Jsonb(memory.metadata),
                    embedding,
                    self._evaluator.evaluate(memory),
                    compressed,
                    now,
                    now,
                    memory.expires_at,
                ),
            ).fetchone()
        return _record_from_row(row)

    def search(
        self,
        user_id: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        query_embedding: tuple[float, ...] | None = None,
        limit: int = 10,
        min_value_score: float = 0,
    ) -> list[MemoryRecord]:
        kind_values = [kind.value for kind in (kinds or MemoryKind)]
        params: list[Any] = [user_id, kind_values, min_value_score]
        order = "value_score DESC, updated_at DESC"
        if query_embedding is not None:
            order = "embedding <=> %s, value_score DESC"
            params.append(Vector(query_embedding))
        params.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM research_memories
                WHERE user_id = %s
                  AND memory_type = ANY(%s)
                  AND value_score >= %s
                  AND (expires_at IS NULL OR expires_at > NOW())
                  {"AND embedding IS NOT NULL" if query_embedding is not None else ""}
                ORDER BY {order}
                LIMIT %s
                """,
                params,
            ).fetchall()
            records = [_record_from_row(row) for row in rows]
            for record in records:
                self.record_access(record.memory_id, connection=connection)
        return records

    def record_access(
        self,
        memory_id: str,
        *,
        connection: psycopg.Connection[Any] | None = None,
    ) -> None:
        if connection is not None:
            connection.execute(
                """
                UPDATE research_memories
                SET access_count = access_count + 1,
                    value_score = LEAST(1.0, value_score + 0.02),
                    updated_at = NOW()
                WHERE memory_id = %s
                """,
                (memory_id,),
            )
            return
        with self._connect() as active_connection:
            self.record_access(memory_id, connection=active_connection)

    def delete(self, memory_id: str) -> None:
        """Idempotently remove one memory for SAGA compensation."""

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM research_memories WHERE memory_id = %s",
                (memory_id,),
            )

    def evict(self) -> MemoryEvictionResult:
        with self._connect() as connection:
            low_value_rows = connection.execute(
                """
                DELETE FROM research_memories
                WHERE (expires_at IS NOT NULL AND expires_at <= NOW()) OR value_score < %s
                RETURNING memory_id
                """,
                (self._min_value_score,),
            ).fetchall()
            overflow_rows = connection.execute(
                """
                WITH ranked AS (
                    SELECT memory_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY user_id ORDER BY value_score DESC, updated_at DESC
                           ) AS rank
                    FROM research_memories
                )
                DELETE FROM research_memories
                WHERE memory_id IN (SELECT memory_id FROM ranked WHERE rank > %s)
                RETURNING memory_id
                """,
                (self._max_records_per_user,),
            ).fetchall()
        ids = [row["memory_id"] for row in [*low_value_rows, *overflow_rows]]
        return MemoryEvictionResult(evicted_ids=tuple(ids))

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


def create_long_term_memory(settings: Settings) -> LongTermMemory:
    """Create the configured long-term backend."""

    compressor = MemoryCompressor(settings.memory_compression_chars)
    common = {
        "compressor": compressor,
        "min_value_score": settings.memory_min_value_score,
        "max_records_per_user": settings.memory_max_records_per_user,
    }
    if settings.long_term_memory_backend == "postgres":
        return PostgresLongTermMemory(
            settings.database_url,
            embedding_dimension=settings.embedding_dimension,
            auto_setup=True,
            **common,
        )
    return InMemoryLongTermMemory(**common)


def _record_from_row(row: dict[str, Any] | None) -> MemoryRecord:
    if row is None:
        raise RuntimeError("memory write returned no row")
    stored_embedding = row.get("embedding")
    embedding = (
        tuple(stored_embedding.to_list())
        if isinstance(stored_embedding, Vector)
        else tuple(stored_embedding)
        if stored_embedding is not None
        else None
    )
    return MemoryRecord(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        kind=MemoryKind(row["memory_type"]),
        content=row["content"],
        summary=row["summary"],
        metadata=row["metadata"],
        embedding=embedding,
        value_score=row["value_score"],
        access_count=row["access_count"],
        compressed=row["compressed"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
    )


def _cosine_similarity(
    query: tuple[float, ...],
    candidate: tuple[float, ...] | None,
) -> float:
    if candidate is None or len(query) != len(candidate) or not query:
        return 0
    numerator = sum(left * right for left, right in zip(query, candidate, strict=True))
    left_norm = sqrt(sum(value * value for value in query))
    right_norm = sqrt(sum(value * value for value in candidate))
    if left_norm == 0 or right_norm == 0:
        return 0
    return numerator / (left_norm * right_norm)
