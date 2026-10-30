"""Failure classification and bounded recovery decisions for worker execution."""

from __future__ import annotations

from enum import StrEnum

import psycopg
from pydantic import BaseModel, ConfigDict

from graph.state import AgentName, ResearchTask
from guardrail.middleware import (
    HumanApprovalRequired,
    InvalidToolArguments,
    ToolPermissionDenied,
    ToolQuotaExceeded,
)


class EmptyRetrievalError(RuntimeError):
    """A retrieval operation completed without any usable evidence."""


class ToolExecutionError(RuntimeError):
    """A tool returned an explicit error result."""


class RecoveryAction(StrEnum):
    RETRY = "retry"
    REPLAN = "replan"
    HUMAN = "human"
    FAIL = "fail"


class RecoveryDecision(BaseModel):
    """Supervisor-readable recovery action with an auditable reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: RecoveryAction
    reason: str
    retry_delay_seconds: float = 0


class RecoveryPolicy:
    """Map failures to bounded retry, local re-planning, or human intervention."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_tool_retries: int = 2,
        max_task_replans: int = 1,
        retry_backoff_seconds: float = 0,
    ) -> None:
        self.enabled = enabled
        self.max_tool_retries = max(0, max_tool_retries)
        self.max_task_replans = max(0, max_task_replans)
        self.retry_backoff_seconds = max(0, retry_backoff_seconds)

    @classmethod
    def disabled(cls) -> RecoveryPolicy:
        return cls(enabled=False, max_tool_retries=0, max_task_replans=0)

    def decide(
        self,
        error: Exception,
        task: ResearchTask,
        *,
        failure_count: int,
    ) -> RecoveryDecision:
        if not self.enabled:
            return RecoveryDecision(action=RecoveryAction.FAIL, reason=str(error))
        if isinstance(
            error,
            HumanApprovalRequired | ToolPermissionDenied | ToolQuotaExceeded,
        ):
            return RecoveryDecision(action=RecoveryAction.HUMAN, reason=str(error))
        if isinstance(error, InvalidToolArguments):
            if task.attempts <= self.max_task_replans:
                return RecoveryDecision(action=RecoveryAction.REPLAN, reason=str(error))
            return RecoveryDecision(action=RecoveryAction.HUMAN, reason=str(error))
        if isinstance(error, EmptyRetrievalError):
            if task.attempts <= self.max_task_replans:
                return RecoveryDecision(action=RecoveryAction.REPLAN, reason=str(error))
            return RecoveryDecision(
                action=RecoveryAction.HUMAN,
                reason="retrieval evidence remained insufficient after local re-planning",
            )
        retryable = isinstance(
            error,
            (
                ConnectionError,
                TimeoutError,
                ToolExecutionError,
                psycopg.OperationalError,
            ),
        ) or task.assigned_agent in {AgentName.RETRIEVER, AgentName.DATABASE}
        if retryable and failure_count <= self.max_tool_retries:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                reason=str(error),
                retry_delay_seconds=self.retry_backoff_seconds * (2 ** (failure_count - 1)),
            )
        return RecoveryDecision(action=RecoveryAction.FAIL, reason=str(error))
