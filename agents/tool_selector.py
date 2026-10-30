"""LLM-based selection of discovered tools and schema-constrained arguments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.model import StructuredModel
from graph.state import ResearchTask
from tools.registry import ToolSpec


class PlannedToolCall(BaseModel):
    """One proposed invocation referencing a discovered tool by name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""


class ToolCallPlan(BaseModel):
    """Bounded tool plan produced for one DAG task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    calls: tuple[PlannedToolCall, ...] = Field(default=(), max_length=20)


class ToolSelector(Protocol):
    def select(
        self,
        user_goal: str,
        task: ResearchTask,
        available_tools: Sequence[ToolSpec],
    ) -> ToolCallPlan: ...


class LLMToolSelector:
    """Use a small model to choose only relevant tools and valid arguments."""

    def __init__(self, model: StructuredModel) -> None:
        self._model = model

    def select(
        self,
        user_goal: str,
        task: ResearchTask,
        available_tools: Sequence[ToolSpec],
    ) -> ToolCallPlan:
        if not available_tools:
            return ToolCallPlan()
        catalog = [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "risk": spec.risk.value,
            }
            for spec in available_tools
        ]
        messages = [
            SystemMessage(
                content=(
                    "Select the minimum set of tools needed for this research task. Use only tool "
                    "names from the supplied catalog and make every argument conform to its JSON "
                    "schema. Never propose mutating SQL; database queries must be read-only and "
                    "bounded. Return no calls when the catalog cannot safely answer the task."
                )
            ),
            HumanMessage(
                content=(
                    f"User goal:\n{user_goal}\n\nTask:\n{task.model_dump_json(indent=2)}"
                    f"\n\nAvailable tools:\n{catalog}"
                )
            ),
        ]
        plan = self._model.invoke(ToolCallPlan, messages)
        available_names = {spec.name for spec in available_tools}
        unknown = {call.tool_name for call in plan.calls} - available_names
        if unknown:
            raise ValueError(f"tool selector chose undiscovered tools: {sorted(unknown)}")
        return plan
