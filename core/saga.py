"""Durable SAGA journal and reverse-order compensation coordinator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict

from core.config import Settings


class SagaStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


class SagaRecord(BaseModel):
    """Durable state of one SAGA step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    saga_id: str
    run_id: str
    step_index: int
    step_name: str
    status: SagaStatus
    error: str | None = None
    updated_at: datetime


@dataclass(frozen=True)
class SagaStep:
    """An idempotent side effect and its idempotent compensation."""

    name: str
    action: Callable[[], Any]
    compensate: Callable[[], Any]


class SagaJournal(Protocol):
    def list(self, saga_id: str) -> list[SagaRecord]: ...

    def put(self, record: SagaRecord) -> None: ...


class InMemorySagaJournal:
    def __init__(self) -> None:
        self._records: dict[tuple[str, int], SagaRecord] = {}
        self._lock = RLock()

    def list(self, saga_id: str) -> list[SagaRecord]:
        with self._lock:
            return sorted(
                (record for key, record in self._records.items() if key[0] == saga_id),
                key=lambda record: record.step_index,
            )

    def put(self, record: SagaRecord) -> None:
        with self._lock:
            self._records[(record.saga_id, record.step_index)] = record


class PostgresSagaJournal:
    def __init__(self, database_url: str, *, auto_setup: bool = True) -> None:
        self._database_url = database_url
        if auto_setup:
            self.setup()

    def setup(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS saga_journal (
                    saga_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    updated_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (saga_id, step_index)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_saga_run
                ON saga_journal(run_id, updated_at DESC)
                """
            )

    def list(self, saga_id: str) -> list[SagaRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM saga_journal WHERE saga_id = %s ORDER BY step_index",
                (saga_id,),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def put(self, record: SagaRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO saga_journal (
                    saga_id, run_id, step_index, step_name, status, error, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (saga_id, step_index) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    step_name = EXCLUDED.step_name,
                    status = EXCLUDED.status,
                    error = EXCLUDED.error,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    record.saga_id,
                    record.run_id,
                    record.step_index,
                    record.step_name,
                    record.status.value,
                    record.error,
                    record.updated_at,
                ),
            )

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(
            self._database_url,
            autocommit=True,
            row_factory=dict_row,
        )


class SagaExecutionError(RuntimeError):
    """Raised after failed side effects have been compensated as far as possible."""


class SagaCoordinator:
    """Resume completed steps and compensate started steps in reverse order on failure."""

    def __init__(self, journal: SagaJournal | None = None) -> None:
        self.journal = journal or InMemorySagaJournal()

    def execute(
        self,
        saga_id: str,
        run_id: str,
        steps: Sequence[SagaStep],
    ) -> None:
        if len({step.name for step in steps}) != len(steps):
            raise ValueError("SAGA step names must be unique")
        existing = {record.step_index: record for record in self.journal.list(saga_id)}
        started: list[int] = []
        try:
            for index, step in enumerate(steps):
                previous = existing.get(index)
                if previous is not None and previous.status == SagaStatus.COMPLETED:
                    started.append(index)
                    continue
                started.append(index)
                self._put(saga_id, run_id, index, step.name, SagaStatus.RUNNING)
                step.action()
                self._put(saga_id, run_id, index, step.name, SagaStatus.COMPLETED)
        except Exception as exc:
            self._put(
                saga_id,
                run_id,
                started[-1],
                steps[started[-1]].name,
                SagaStatus.FAILED,
                str(exc),
            )
            compensation_errors = []
            for index in reversed(started):
                step = steps[index]
                try:
                    step.compensate()
                    self._put(
                        saga_id,
                        run_id,
                        index,
                        step.name,
                        SagaStatus.COMPENSATED,
                    )
                except Exception as compensation_exc:  # noqa: BLE001
                    compensation_errors.append(str(compensation_exc))
                    self._put(
                        saga_id,
                        run_id,
                        index,
                        step.name,
                        SagaStatus.COMPENSATION_FAILED,
                        str(compensation_exc),
                    )
            suffix = (
                f"; compensation errors: {compensation_errors}" if compensation_errors else ""
            )
            raise SagaExecutionError(f"SAGA {saga_id} failed: {exc}{suffix}") from exc

    def _put(
        self,
        saga_id: str,
        run_id: str,
        index: int,
        name: str,
        status: SagaStatus,
        error: str | None = None,
    ) -> None:
        self.journal.put(
            SagaRecord(
                saga_id=saga_id,
                run_id=run_id,
                step_index=index,
                step_name=name,
                status=status,
                error=error,
                updated_at=datetime.now(UTC),
            )
        )


def create_saga_coordinator(settings: Settings) -> SagaCoordinator:
    journal: SagaJournal
    if settings.saga_backend == "postgres":
        journal = PostgresSagaJournal(
            settings.database_url,
            auto_setup=settings.saga_auto_setup,
        )
    else:
        journal = InMemorySagaJournal()
    return SagaCoordinator(journal)


def _record_from_row(row: dict[str, Any]) -> SagaRecord:
    return SagaRecord(
        saga_id=row["saga_id"],
        run_id=row["run_id"],
        step_index=row["step_index"],
        step_name=row["step_name"],
        status=SagaStatus(row["status"]),
        error=row["error"],
        updated_at=row["updated_at"],
    )
