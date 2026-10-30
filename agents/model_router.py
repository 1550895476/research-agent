"""Dynamic small/large model routing with cost and trace metadata."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agents.model import StructuredModel, StructuredOutput
from core.observability import AgentObservability, estimate_tokens


class RoutedTier(StrEnum):
    SMALL = "small"
    LARGE = "large"


class DynamicModelRouter:
    """Route extraction/rewrite to small models and reasoning/generation to large models."""

    _small_schemas = {
        "PlanImpact",
        "ResearchRequirement",
        "RewrittenQuery",
        "ToolCallPlan",
    }
    _large_schemas = {
        "ReportDraft",
        "ResearchOutput",
        "ResearchPlan",
        "TaskQualityAssessment",
        "TaskRevision",
        "VerificationOutput",
    }

    def __init__(
        self,
        small_model: StructuredModel,
        large_model: StructuredModel,
        *,
        observability: AgentObservability,
        complexity_chars: int = 4000,
    ) -> None:
        self._small_model = small_model
        self._large_model = large_model
        self._observability = observability
        self._complexity_chars = max(500, complexity_chars)

    def select(self, schema_name: str, prompt_chars: int) -> RoutedTier:
        if schema_name in self._small_schemas:
            return RoutedTier.SMALL
        if schema_name in self._large_schemas:
            return RoutedTier.LARGE
        return RoutedTier.SMALL if prompt_chars < self._complexity_chars else RoutedTier.LARGE

    def invoke(
        self,
        schema: type[StructuredOutput],
        messages: Sequence[BaseMessage],
    ) -> StructuredOutput:
        prompt = "\n".join(str(message.content) for message in messages)
        tier = self.select(schema.__name__, len(prompt))
        model = self._small_model if tier == RoutedTier.SMALL else self._large_model
        with self._observability.span(
            "model.invoke",
            kind="model",
            metadata={"tier": tier.value, "schema": schema.__name__},
        ) as span:
            span.input_tokens = estimate_tokens(prompt)
            response = model.invoke(schema, messages)
            if isinstance(response, BaseModel):
                span.output_tokens = estimate_tokens(response.model_dump_json())
            return response
