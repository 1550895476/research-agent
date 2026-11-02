"""Dynamic small/large model routing with cost and trace metadata."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agents.model import StructuredModel, StructuredOutput
from core.budget import BudgetLedger
from core.observability import (
    AgentObservability,
    current_observation_context,
    estimate_tokens,
)


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
        budget_ledger: BudgetLedger | None = None,
        complexity_chars: int = 4000,
    ) -> None:
        self._small_model = small_model
        self._large_model = large_model
        self._observability = observability
        self._budget_ledger = budget_ledger
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
            self._reserve_input(tier, span.input_tokens)
            response = model.invoke(schema, messages)
            if isinstance(response, BaseModel):
                span.output_tokens = estimate_tokens(response.model_dump_json())
                self._record_output(tier, span.output_tokens)
            return response

    def _reserve_input(self, tier: RoutedTier, tokens: int) -> None:
        context = current_observation_context()
        if self._budget_ledger is None or context is None:
            return
        self._budget_ledger.reserve_model_input(
            context.run_id,
            context.task_id or context.agent or "workflow",
            tier=tier.value,
            tokens=tokens,
        )

    def _record_output(self, tier: RoutedTier, tokens: int) -> None:
        context = current_observation_context()
        if self._budget_ledger is None or context is None:
            return
        self._budget_ledger.record_model_output(
            context.run_id,
            context.task_id or context.agent or "workflow",
            tier=tier.value,
            tokens=tokens,
        )


class MeteredStructuredModel:
    """Add observability and budget enforcement to a fixed-tier model."""

    def __init__(
        self,
        model: StructuredModel,
        *,
        tier: RoutedTier,
        observability: AgentObservability,
        budget_ledger: BudgetLedger,
    ) -> None:
        self._model = model
        self._tier = tier
        self._observability = observability
        self._budget_ledger = budget_ledger

    def invoke(
        self,
        schema: type[StructuredOutput],
        messages: Sequence[BaseMessage],
    ) -> StructuredOutput:
        prompt = "\n".join(str(message.content) for message in messages)
        context = current_observation_context()
        task_id = (
            context.task_id or context.agent or "workflow"
            if context is not None
            else None
        )
        with self._observability.span(
            "model.invoke",
            kind="model",
            metadata={"tier": self._tier.value, "schema": schema.__name__},
        ) as span:
            span.input_tokens = estimate_tokens(prompt)
            if context is not None and task_id is not None:
                self._budget_ledger.reserve_model_input(
                    context.run_id,
                    task_id,
                    tier=self._tier.value,
                    tokens=span.input_tokens,
                )
            response = self._model.invoke(schema, messages)
            if isinstance(response, BaseModel):
                span.output_tokens = estimate_tokens(response.model_dump_json())
                if context is not None and task_id is not None:
                    self._budget_ledger.record_model_output(
                        context.run_id,
                        task_id,
                        tier=self._tier.value,
                        tokens=span.output_tokens,
                    )
            return response
