"""Verifier Agent for claim consistency checks and uncertainty scoring."""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.model import StructuredModel
from agents.researcher import _render_context, _task_snapshot
from core.context_compression import ContextCompressor
from graph.state import ResearchTask


class VerifiedClaim(BaseModel):
    """Verification decision for one material claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    statement: str = Field(min_length=1)
    verified: bool
    confidence: float = Field(ge=0, le=1)
    supporting_evidence_ids: tuple[str, ...] = ()
    supporting_result_ids: tuple[str, ...] = ()
    conflict: str | None = None


class VerificationOutput(BaseModel):
    """Structured result of cross-source consistency checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verified: bool
    confidence: float = Field(ge=0, le=1)
    claims: tuple[VerifiedClaim, ...] = ()
    conflicts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class CriterionAssessment(BaseModel):
    """Verifier decision for one Planner-defined task completion criterion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion: str = Field(min_length=1)
    satisfied: bool
    rationale: str = ""
    supporting_evidence_ids: tuple[str, ...] = ()


class TaskQualityAssessment(BaseModel):
    """Quality-gate score for one Researcher task before DAG fan-in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    score: int = Field(ge=0, le=100)
    complete: bool
    criteria: tuple[CriterionAssessment, ...] = ()
    claims: tuple[VerifiedClaim, ...] = ()
    missing_aspects: tuple[str, ...] = ()
    unresolved_conflicts: tuple[str, ...] = ()
    followup_queries: tuple[str, ...] = Field(default=(), max_length=10)


class VerifierAgent:
    """Verify claims only against payloads actually provided by dependencies."""

    def __init__(
        self,
        model: StructuredModel,
        *,
        max_context_chars: int = 16000,
        context_compressor: ContextCompressor | None = None,
    ) -> None:
        self._model = model
        self._max_context_chars = max_context_chars
        self._context_compressor = context_compressor or ContextCompressor(
            max_chars=max_context_chars
        )

    def verify(
        self,
        user_goal: str,
        task: ResearchTask,
        dependency_payloads: Sequence[Any],
        *,
        context_id: str | None = None,
    ) -> VerificationOutput:
        """Check support, disagreement, and uncertainty for material claims."""

        context = _render_context(
            dependency_payloads,
            self._max_context_chars,
            compressor=self._context_compressor,
            context_id=context_id or f"task:{task.task_id}:verifier",
            snapshot=_task_snapshot(user_goal, task),
        )
        messages = [
            SystemMessage(
                content=(
                    "You are the Verifier Agent. Validate each material claim only against the "
                    "provided dependency context. A claim without source evidence must not receive "
                    "high confidence. Detect contradictions, distinguish independent support from "
                    "repetition, and never fabricate evidence or result IDs."
                )
            ),
            HumanMessage(
                content=(
                    f"User goal:\n{user_goal}\n\nVerification task:\n"
                    f"{task.model_dump_json(indent=2)}\n\nCandidate findings:\n{context}"
                )
            ),
        ]
        return self._model.invoke(VerificationOutput, messages)

    def verify_task(
        self,
        user_goal: str,
        task: ResearchTask,
        candidate_payload: Any,
        evidence_payloads: Sequence[Any],
        *,
        context_id: str | None = None,
    ) -> TaskQualityAssessment:
        """Score one research task and propose only in-scope evidence-gap queries."""

        context = _render_quality_context(
            candidate_payload,
            evidence_payloads,
            self._max_context_chars,
            compressor=self._context_compressor,
            context_id=context_id or f"task:{task.task_id}:quality_gate",
            user_goal=user_goal,
            task=task,
        )
        messages = [
            SystemMessage(
                content=(
                    "You are the task-level Verifier quality gate. Score the Researcher's work "
                    "from 0 to 100 using only supplied evidence. Evaluate every Planner-defined "
                    "completion criterion, verify material claims, detect unresolved conflicts "
                    "and missing aspects, and distinguish independent sources from repeated "
                    "reporting. Mark "
                    "complete only when the task scope is adequately covered. If incomplete, "
                    "return focused followup_queries that stay within the existing task scope; "
                    "never create a new research module or invent citations. Copy task_id and "
                    "each completion criterion verbatim into the response. List only unresolved "
                    "conflicts."
                )
            ),
            HumanMessage(
                content=(
                    f"User goal:\n{user_goal}\n\nResearch task and completion criteria:\n"
                    f"{task.model_dump_json(indent=2)}\n\nCandidate findings and evidence:\n"
                    f"{context}"
                )
            ),
        ]
        return self._model.invoke(TaskQualityAssessment, messages)


def _render_quality_context(
    candidate_payload: Any,
    evidence_payloads: Sequence[Any],
    max_chars: int,
    *,
    compressor: ContextCompressor | None = None,
    context_id: str = "agent:quality_gate",
    user_goal: str = "Verify the current research task.",
    task: ResearchTask | None = None,
) -> str:
    """Preserve a compact sample from every source for criterion-level verification."""
    active_compressor = compressor or ContextCompressor(max_chars=max_chars)
    snapshot = _task_snapshot(user_goal, task) if task is not None else None
    return active_compressor.compress_payloads(
        [candidate_payload, *evidence_payloads],
        context_id=context_id,
        snapshot=snapshot,
        max_chars=max_chars,
    ).text
