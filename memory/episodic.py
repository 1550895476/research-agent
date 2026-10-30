"""Decision-path and failure-attribution memory for Planner feedback."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from core.config import Settings


class EpisodeType(StrEnum):
    """Material events retained for future planning and failure analysis."""

    REQUIREMENT_PARSED = "requirement_parsed"
    PLAN_CREATED = "plan_created"
    SUPERVISOR_DECISION = "supervisor_decision"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"


class EpisodeWrite(BaseModel):
    """Input for one episodic event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    event_type: EpisodeType
    stage: str = Field(min_length=1)
    task_id: str | None = None
    decision: str | None = None
    failure_reason: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class EpisodicEvent(EpisodeWrite):
    """Persisted event with its occurrence time."""

    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EpisodicMemory(Protocol):
    """Storage-neutral event log used by AgentRuntime."""

    def record(self, event: EpisodeWrite) -> EpisodicEvent: ...

    def list_events(
        self,
        *,
        user_id: str,
        run_id: str | None = None,
        event_type: EpisodeType | None = None,
        limit: int = 100,
    ) -> list[EpisodicEvent]: ...

    def recent_failures(self, user_id: str, *, limit: int = 5) -> list[EpisodicEvent]: ...


class InMemoryEpisodicMemory:
    """Thread-safe episodic log for local runs and tests."""

    def __init__(self) -> None:
        self._events: list[EpisodicEvent] = []
        self._lock = RLock()

    def record(self, event: EpisodeWrite) -> EpisodicEvent:
        stored = EpisodicEvent(**event.model_dump())
        with self._lock:
            self._events.append(stored)
        return stored

    def list_events(
        self,
        *,
        user_id: str,
        run_id: str | None = None,
        event_type: EpisodeType | None = None,
        limit: int = 100,
    ) -> list[EpisodicEvent]:
        with self._lock:
            matches = [
                event
                for event in reversed(self._events)
                if event.user_id == user_id
                and (run_id is None or event.run_id == run_id)
                and (event_type is None or event.event_type == event_type)
            ]
            return matches[: max(0, limit)]

    def recent_failures(self, user_id: str, *, limit: int = 5) -> list[EpisodicEvent]:
        with self._lock:
            matches = [
                event
                for event in reversed(self._events)
                if event.user_id == user_id and event.failure_reason
            ]
            return matches[: max(0, limit)]


class PostgresEpisodicMemory:
    """PostgreSQL-backed append-only execution history."""

    def __init__(self, database_url: str, *, auto_setup: bool = True) -> None:
        self._database_url = database_url
        if auto_setup:
            self.setup()

    def setup(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS episodic_memories (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                stage TEXT NOT NULL,
                task_id TEXT,
                decision TEXT,
                failure_reason TEXT,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                occurred_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_episodes_user_time
            ON episodic_memories(user_id, occurred_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_episodes_run_time
            ON episodic_memories(run_id, occurred_at)
            """,
        ]
        with self._connect() as connection:
            for statement in statements:
                connection.execute(statement)

    def record(self, event: EpisodeWrite) -> EpisodicEvent:
        stored = EpisodicEvent(**event.model_dump())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO episodic_memories (
                    event_id, run_id, user_id, event_type, stage, task_id,
                    decision, failure_reason, payload, occurred_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (
                    stored.event_id,
                    stored.run_id,
                    stored.user_id,
                    stored.event_type.value,
                    stored.stage,
                    stored.task_id,
                    stored.decision,
                    stored.failure_reason,
                    Jsonb(stored.payload),
                    stored.occurred_at,
                ),
            )
        return stored

    def list_events(
        self,
        *,
        user_id: str,
        run_id: str | None = None,
        event_type: EpisodeType | None = None,
        limit: int = 100,
    ) -> list[EpisodicEvent]:
        clauses = ["user_id = %s"]
        params: list[Any] = [user_id]
        if run_id is not None:
            clauses.append("run_id = %s")
            params.append(run_id)
        if event_type is not None:
            clauses.append("event_type = %s")
            params.append(event_type.value)
        params.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM episodic_memories
                WHERE {" AND ".join(clauses)}
                ORDER BY occurred_at DESC
                LIMIT %s
                """,
                params,
            ).fetchall()
        return [_event_from_row(row) for row in rows]

    def recent_failures(self, user_id: str, *, limit: int = 5) -> list[EpisodicEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM episodic_memories
                WHERE user_id = %s AND failure_reason IS NOT NULL
                ORDER BY occurred_at DESC
                LIMIT %s
                """,
                (user_id, max(0, limit)),
            ).fetchall()
        return [_event_from_row(row) for row in rows]

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(
            self._database_url,
            autocommit=True,
            row_factory=dict_row,
        )


def create_episodic_memory(settings: Settings) -> EpisodicMemory:
    """Create the configured episodic backend."""

    if settings.episodic_memory_backend == "postgres":
        return PostgresEpisodicMemory(settings.database_url, auto_setup=True)
    return InMemoryEpisodicMemory()


def _event_from_row(row: dict[str, Any]) -> EpisodicEvent:
    return EpisodicEvent(
        event_id=row["event_id"],
        run_id=row["run_id"],
        user_id=row["user_id"],
        event_type=EpisodeType(row["event_type"]),
        stage=row["stage"],
        task_id=row["task_id"],
        decision=row["decision"],
        failure_reason=row["failure_reason"],
        payload=row["payload"],
        occurred_at=row["occurred_at"],
    )
