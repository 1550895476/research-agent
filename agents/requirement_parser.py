"""Requirement Parser Agent: natural-language goal to structured research intent."""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.model import StructuredModel
from graph.state import ResearchRequirement


class RequirementParserAgent:
    """Parse user intent with a small schema-constrained model."""

    def __init__(self, model: StructuredModel) -> None:
        self._model = model

    def parse(self, user_goal: str) -> ResearchRequirement:
        """Convert one user goal into normalized scope, questions, and deliverables."""

        messages = [
            SystemMessage(
                content=(
                    "You are a research requirement analyst. Convert the request into a precise "
                    "research specification. Preserve explicit time, geography, industry, vendor, "
                    "format, and language constraints. Do not invent constraints. key_questions "
                    "must contain concrete questions that a research plan can answer."
                )
            ),
            HumanMessage(content=user_goal),
        ]
        return self._model.invoke(ResearchRequirement, messages)

