"""Thread-safe provenance store for real tool calls and source evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from rag.retriever import SourceType
from tools.registry import ToolEvidence


class ToolCallStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EvidenceRecord(BaseModel):
    """One immutable source captured from an actual tool response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    run_id: str
    user_id: str
    task_id: str
    tool_name: str
    source_type: SourceType
    source_url: str
    title: str = ""
    content: str
    confidence: float = Field(ge=0, le=1)
    related_claim: str | None = None
    created_time: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolCallRecord(BaseModel):
    """Auditable lifecycle record proving whether a tool was really invoked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    run_id: str
    user_id: str
    task_id: str
    tool_name: str
    purpose: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ToolCallStatus
    evidence_ids: tuple[str, ...] = ()
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)


class EvidenceStore(Protocol):
    def start_tool_call(
        self,
        *,
        run_id: str,
        user_id: str,
        task_id: str,
        tool_name: str,
        purpose: str,
        arguments: dict[str, Any],
    ) -> str: ...

    def finish_tool_call(
        self,
        call_id: str,
        *,
        evidences: tuple[ToolEvidence, ...] = (),
        error: str | None = None,
    ) -> None: ...

    def list_evidences(self, run_id: str) -> tuple[EvidenceRecord, ...]: ...

    def list_tool_calls(self, run_id: str) -> tuple[ToolCallRecord, ...]: ...


class InMemoryEvidenceStore:
    """Process-local store used by the web runtime without polluting graph state."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._evidences: dict[tuple[str, str], EvidenceRecord] = {}
        self._tool_calls: dict[str, ToolCallRecord] = {}
        self._started: dict[str, float] = {}

    def start_tool_call(
        self,
        *,
        run_id: str,
        user_id: str,
        task_id: str,
        tool_name: str,
        purpose: str,
        arguments: dict[str, Any],
    ) -> str:
        call_id = f"tool-call:{uuid4().hex}"
        record = ToolCallRecord(
            call_id=call_id,
            run_id=run_id,
            user_id=user_id,
            task_id=task_id,
            tool_name=tool_name,
            purpose=purpose,
            arguments=arguments,
            status=ToolCallStatus.RUNNING,
            started_at=datetime.now(UTC),
        )
        with self._lock:
            self._tool_calls[call_id] = record
            self._started[call_id] = perf_counter()
        return call_id

    def finish_tool_call(
        self,
        call_id: str,
        *,
        evidences: tuple[ToolEvidence, ...] = (),
        error: str | None = None,
    ) -> None:
        with self._lock:
            record = self._tool_calls[call_id]
            elapsed = max(0.0, (perf_counter() - self._started.pop(call_id)) * 1000)
            stored_ids: list[str] = []
            for evidence in evidences:
                stored = EvidenceRecord(
                    evidence_id=evidence.evidence_id,
                    run_id=record.run_id,
                    user_id=record.user_id,
                    task_id=record.task_id,
                    tool_name=record.tool_name,
                    source_type=evidence.source_type,
                    source_url=evidence.source_uri,
                    title=evidence.title,
                    content=evidence.content,
                    confidence=evidence.confidence,
                )
                self._evidences[(record.run_id, evidence.evidence_id)] = stored
                stored_ids.append(evidence.evidence_id)
            self._tool_calls[call_id] = record.model_copy(
                update={
                    "status": (
                        ToolCallStatus.FAILED if error is not None else ToolCallStatus.SUCCEEDED
                    ),
                    "evidence_ids": tuple(stored_ids),
                    "error": error,
                    "finished_at": datetime.now(UTC),
                    "duration_ms": elapsed,
                }
            )

    def list_evidences(self, run_id: str) -> tuple[EvidenceRecord, ...]:
        with self._lock:
            return tuple(
                record
                for (record_run_id, _), record in self._evidences.items()
                if record_run_id == run_id
            )

    def list_tool_calls(self, run_id: str) -> tuple[ToolCallRecord, ...]:
        with self._lock:
            return tuple(
                record for record in self._tool_calls.values() if record.run_id == run_id
            )
