"""Deterministic Lead routing and pre-flight plan cost estimation."""

from __future__ import annotations

from typing import Literal

from core.budget import RunBudget
from graph.state import (
    AgentName,
    LeadDecision,
    PlanEstimate,
    ResearchRequirement,
    ResearchTask,
    ResearchType,
)


class LeadRouter:
    """Use stable rules for orchestration decisions without another model call."""

    _complex_markers = (
        "比较",
        "对比",
        "竞品",
        "行业",
        "全景",
        "趋势",
        "compare",
        "versus",
        "landscape",
        "market",
    )

    def __init__(
        self,
        *,
        run_budget: RunBudget,
        approval_mode: Literal["off", "auto", "always"] = "auto",
        approval_task_threshold: int = 8,
        approval_token_threshold: int = 200000,
        approval_cost_threshold_usd: float = 5,
    ) -> None:
        self._run_budget = run_budget
        self._approval_mode = approval_mode
        self._approval_task_threshold = approval_task_threshold
        self._approval_token_threshold = approval_token_threshold
        self._approval_cost_threshold_usd = approval_cost_threshold_usd

    def decide(self, requirement: ResearchRequirement) -> LeadDecision:
        text = " ".join((requirement.objective, *requirement.key_questions)).casefold()
        complex_request = (
            requirement.research_type != ResearchType.GENERAL
            or len(requirement.key_questions) > 1
            or any(marker in text for marker in self._complex_markers)
        )
        workers = 1
        if complex_request:
            workers = min(
                self._run_budget.max_concurrent_workers,
                max(2, min(5, len(requirement.key_questions) + 1)),
                max(1, self._run_budget.max_tasks - 2),
            )
        return LeadDecision(
            mode="multi" if workers > 1 else "single",
            recommended_workers=workers,
            reason=(
                "需求包含多个可独立研究的子题，适合并行覆盖"
                if workers > 1
                else "需求边界集中，单个研究 Worker 足以完成，避免无效并行"
            ),
        )

    def estimate(self, tasks: list[ResearchTask]) -> PlanEstimate:
        research_tasks = [
            task
            for task in tasks
            if task.assigned_agent
            in {AgentName.RESEARCHER, AgentName.RETRIEVER, AgentName.DATABASE}
        ]
        estimated_searches = sum(task.budget.max_searches for task in research_tasks)
        estimated_input_tokens = sum(
            min(task.budget.max_input_tokens, 15000)
            if task in research_tasks
            else min(task.budget.max_input_tokens, 10000)
            for task in tasks
        )
        estimated_output_tokens = sum(
            min(task.budget.max_output_tokens, 4000)
            if task in research_tasks
            else min(task.budget.max_output_tokens, 6000)
            for task in tasks
        )
        policy = self._run_budget
        estimated_cost = (
            estimated_input_tokens * policy.large_input_cost_per_million
            + estimated_output_tokens * policy.large_output_cost_per_million
        ) / 1_000_000
        longest_worker_seconds = max(
            (task.budget.max_seconds for task in research_tasks),
            default=0,
        )
        coordination_seconds = sum(
            task.budget.max_seconds
            for task in tasks
            if task not in research_tasks
        )
        estimate = PlanEstimate(
            task_count=len(tasks),
            research_workers=len(research_tasks),
            estimated_searches=estimated_searches,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            estimated_cost_usd=round(estimated_cost, 6),
            estimated_seconds=min(
                policy.max_seconds,
                longest_worker_seconds + coordination_seconds,
            ),
        )
        return estimate

    def approval_required(self, estimate: PlanEstimate) -> bool:
        if self._approval_mode == "off":
            return False
        if self._approval_mode == "always":
            return True
        return (
            estimate.task_count >= self._approval_task_threshold
            or estimate.estimated_input_tokens + estimate.estimated_output_tokens
            >= self._approval_token_threshold
            or (
                self._approval_cost_threshold_usd > 0
                and estimate.estimated_cost_usd >= self._approval_cost_threshold_usd
            )
        )

    @staticmethod
    def with_approval(
        decision: LeadDecision,
        *,
        run_id: str,
        required: bool,
    ) -> LeadDecision:
        return decision.model_copy(
            update={
                "requires_approval": required,
                "approval_token": f"plan:{run_id}" if required else None,
            }
        )
