"""Thread-safe out-of-state storage for Agent payloads."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from core.config import Settings


class BlackboardRecord(BaseModel):
    """One payload stored outside LangGraph state and referenced by URI."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    result_id: str
    task_id: str
    payload: Any
    payload_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Blackboard(Protocol):
    """Reference store required by AgentRuntime."""

    def put(self, result_id: str, task_id: str, payload: Any) -> str: ...

    def get_by_uri(self, payload_ref: str) -> BlackboardRecord: ...

    def payloads_for_tasks(self, task_ids: tuple[str, ...]) -> list[Any]: ...


class InMemoryBlackboard:
    """Process-local Blackboard used by the MVP and safe for parallel workers."""

    uri_prefix = "blackboard://results/"

    def __init__(self) -> None:
        self._records: dict[str, BlackboardRecord] = {}
        self._lock = RLock()

    def put(self, result_id: str, task_id: str, payload: Any) -> str:
        """Store or replace a result payload and return its stable reference URI."""

        record = BlackboardRecord(
            result_id=result_id,
            task_id=task_id,
            payload=payload,
            payload_type=type(payload).__name__,
        )
        with self._lock:
            self._records[result_id] = record
        return f"{self.uri_prefix}{result_id}"

    def get(self, result_id: str) -> BlackboardRecord:
        """Read a record by result identifier."""

        with self._lock:
            try:
                return self._records[result_id]
            except KeyError as exc:
                raise KeyError(f"unknown Blackboard result: {result_id}") from exc

    def get_by_uri(self, payload_ref: str) -> BlackboardRecord:
        """Resolve a Blackboard URI returned through AgentState."""

        if not payload_ref.startswith(self.uri_prefix):
            raise ValueError(f"unsupported Blackboard reference: {payload_ref}")
        return self.get(payload_ref.removeprefix(self.uri_prefix))

    def payloads_for_tasks(self, task_ids: tuple[str, ...]) -> list[Any]:
        """Return dependency payloads in task order without exposing storage internals."""

        with self._lock:
            by_task = {record.task_id: record.payload for record in self._records.values()}
            return [by_task[task_id] for task_id in task_ids if task_id in by_task]

    def snapshot(self) -> dict[str, BlackboardRecord]:
        """Return a shallow diagnostic snapshot for tests and observability."""

        with self._lock:
            return dict(self._records)


class PostgresBlackboard:
    """Durable Blackboard so checkpoint recovery can resolve prior result references."""

    uri_prefix = InMemoryBlackboard.uri_prefix

    def __init__(self, database_url: str, *, auto_setup: bool = True) -> None:
        self._database_url = database_url
        if auto_setup:
            self.setup()

    def setup(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS blackboard_results (
                result_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                payload_type TEXT NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_blackboard_task
            ON blackboard_results(task_id, created_at DESC)
            """,
        ]
        with self._connect() as connection:
            for statement in statements:
                connection.execute(statement)

    def put(self, result_id: str, task_id: str, payload: Any) -> str:
        serialized = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        payload_type = type(payload).__name__
        created_at = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO blackboard_results (
                    result_id, task_id, payload_type, payload, created_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (result_id) DO UPDATE SET
                    task_id = EXCLUDED.task_id,
                    payload_type = EXCLUDED.payload_type,
                    payload = EXCLUDED.payload,
                    created_at = EXCLUDED.created_at
                """,
                (result_id, task_id, payload_type, Jsonb(serialized), created_at),
            )
        return f"{self.uri_prefix}{result_id}"

    def get(self, result_id: str) -> BlackboardRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM blackboard_results WHERE result_id = %s",
                (result_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown Blackboard result: {result_id}")
        return _blackboard_record_from_row(row)

    def get_by_uri(self, payload_ref: str) -> BlackboardRecord:
        if not payload_ref.startswith(self.uri_prefix):
            raise ValueError(f"unsupported Blackboard reference: {payload_ref}")
        return self.get(payload_ref.removeprefix(self.uri_prefix))

    def payloads_for_tasks(self, task_ids: tuple[str, ...]) -> list[Any]:
        if not task_ids:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT ON (task_id) *
                FROM blackboard_results
                WHERE task_id = ANY(%s)
                ORDER BY task_id, created_at DESC
                """,
                (list(task_ids),),
            ).fetchall()
        by_task = {row["task_id"]: row["payload"] for row in rows}
        return [by_task[task_id] for task_id in task_ids if task_id in by_task]

    def snapshot(self) -> dict[str, BlackboardRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM blackboard_results").fetchall()
        return {
            row["result_id"]: _blackboard_record_from_row(row)
            for row in rows
        }

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(
            self._database_url,
            autocommit=True,
            row_factory=dict_row,
        )


def create_blackboard(settings: Settings) -> Blackboard:
    """Create a process-local or PostgreSQL Blackboard backend."""

    if settings.blackboard_backend == "postgres":
        return PostgresBlackboard(settings.database_url, auto_setup=True)
    return InMemoryBlackboard()


def _blackboard_record_from_row(row: dict[str, Any]) -> BlackboardRecord:
    return BlackboardRecord(
        result_id=row["result_id"],
        task_id=row["task_id"],
        payload=row["payload"],
        payload_type=row["payload_type"],
        created_at=row["created_at"],
    )
