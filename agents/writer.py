"""Report Writer Agent for evidence-aware structured Markdown reports."""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.model import StructuredModel
from agents.researcher import _render_context
from graph.state import ResearchRequirement, ResearchTask


class ReportSection(BaseModel):
    """One structured report section and its Blackboard support references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    heading: str = Field(min_length=1)
    content: str = Field(min_length=1)
    supporting_result_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()


class ReportDraft(BaseModel):
    """Writer output including a directly deliverable Markdown representation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    sections: tuple[ReportSection, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()
    markdown_report: str = Field(min_length=1)


class ReportWriterAgent:
    """Synthesize verified context without silently upgrading uncertain claims."""

    def __init__(self, model: StructuredModel, *, max_context_chars: int = 16000) -> None:
        self._model = model
        self._max_context_chars = max_context_chars

    def write(
        self,
        user_goal: str,
        requirement: ResearchRequirement | None,
        task: ResearchTask,
        dependency_payloads: Sequence[Any],
    ) -> ReportDraft:
        """Generate a structured report and Markdown final answer."""

        context = _render_context(dependency_payloads, self._max_context_chars)
        requirement_text = requirement.model_dump_json(indent=2) if requirement else user_goal
        messages = [
            SystemMessage(
                content=(
                    "You are the Report Writer Agent. Produce a clear research report from "
                    "verified inputs only. Preserve uncertainty and conflicts. Cite references in "
                    "Markdown using the exact IDs, for example [result:task:uuid] or "
                    "[evidence:web:abc123]. Never invent citations. Include "
                    "an executive summary, findings, comparison where relevant, limitations, and "
                    "future directions."
                )
            ),
            HumanMessage(
                content=(
                    f"Requirement:\n{requirement_text}\n\nWriter task:\n"
                    f"{task.model_dump_json(indent=2)}\n\nVerified context:\n{context}"
                )
            ),
        ]
        return self._model.invoke(ReportDraft, messages)
