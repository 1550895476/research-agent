"""Agent trace collection with optional LangSmith export and token estimation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import uuid4

from langsmith import Client, trace
from pydantic import BaseModel, ConfigDict, Field

from core.config import Settings


@dataclass(frozen=True)
class ObservationContext:
    run_id: str = "unscoped"
    user_id: str = "unknown"
    task_id: str | None = None
    agent: str | None = None


_current_context: ContextVar[ObservationContext | None] = ContextVar(
    "agent_observation_context",
    default=None,
)


class TraceRecord(BaseModel):
    """One completed Supervisor, Agent, model, or tool span."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    run_id: str
    user_id: str
    task_id: str | None = None
    agent: str | None = None
    name: str
    kind: str
    status: str
    started_at: datetime
    duration_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


@dataclass
class ActiveSpan:
    metadata: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


class AgentObservability:
    """Thread-safe local trace store; subclasses may additionally export spans."""

    def __init__(self) -> None:
        self._records: list[TraceRecord] = []
        self._lock = RLock()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[ActiveSpan]:
        context = _current_context.get() or ObservationContext()
        active = ActiveSpan(metadata=dict(metadata or {}))
        started_at = datetime.now(UTC)
        started = perf_counter()
        error: str | None = None
        status = "completed"
        try:
            with self._external_span(name, kind, context, active):
                yield active
        except Exception as exc:
            status = "failed"
            error = str(exc)
            raise
        finally:
            record = TraceRecord(
                trace_id=str(uuid4()),
                run_id=context.run_id,
                user_id=context.user_id,
                task_id=context.task_id,
                agent=context.agent,
                name=name,
                kind=kind,
                status=status,
                started_at=started_at,
                duration_ms=(perf_counter() - started) * 1000,
                input_tokens=active.input_tokens,
                output_tokens=active.output_tokens,
                metadata=active.metadata,
                error=error,
            )
            with self._lock:
                self._records.append(record)

    @contextmanager
    def _external_span(
        self,
        name: str,
        kind: str,
        context: ObservationContext,
        active: ActiveSpan,
    ) -> Iterator[None]:
        del name, kind, context, active
        yield

    def snapshot(self, *, run_id: str | None = None) -> list[TraceRecord]:
        with self._lock:
            records = list(self._records)
        return [record for record in records if run_id is None or record.run_id == run_id]


class LangSmithObservability(AgentObservability):
    """Local trace store plus explicit LangSmith span export."""

    def __init__(self, *, api_key: str, project: str) -> None:
        super().__init__()
        self._client = Client(api_key=api_key)
        self._project = project

    @contextmanager
    def _external_span(
        self,
        name: str,
        kind: str,
        context: ObservationContext,
        active: ActiveSpan,
    ) -> Iterator[None]:
        with trace(
            name,
            run_type=_langsmith_run_type(kind),
            inputs={"task_id": context.task_id, "agent": context.agent},
            project_name=self._project,
            client=self._client,
            metadata={"run_id": context.run_id, **active.metadata},
        ) as run:
            try:
                yield
            finally:
                run.add_metadata(
                    {
                        **active.metadata,
                        "input_tokens": active.input_tokens,
                        "output_tokens": active.output_tokens,
                        "token_counting": "estimated",
                    }
                )


@contextmanager
def observation_scope(
    *,
    run_id: str,
    user_id: str,
    task_id: str | None = None,
    agent: str | None = None,
) -> Iterator[None]:
    token = _current_context.set(
        ObservationContext(
            run_id=run_id,
            user_id=user_id,
            task_id=task_id,
            agent=agent,
        )
    )
    try:
        yield
    finally:
        _current_context.reset(token)


def current_observation_context() -> ObservationContext | None:
    """Return the active execution identity for metering and policy enforcement."""

    return _current_context.get()


def estimate_tokens(text: str) -> int:
    """Cheap deterministic estimate when an OpenAI-compatible endpoint omits usage."""

    return max(1, (len(text) + 3) // 4) if text else 0


def create_observability(settings: Settings) -> AgentObservability:
    if settings.langsmith_tracing and settings.langsmith_api_key:
        return LangSmithObservability(
            api_key=settings.langsmith_api_key.get_secret_value(),
            project=settings.langsmith_project,
        )
    return AgentObservability()


def _langsmith_run_type(kind: str) -> str:
    return {
        "model": "llm",
        "tool": "tool",
        "agent": "chain",
        "supervisor": "chain",
    }.get(kind, "chain")
