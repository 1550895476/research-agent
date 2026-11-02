"""Atomic global and per-task budgets for multi-agent research runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BudgetExceeded(RuntimeError):
    """Raised before a model, search, or worker can exceed its configured budget."""


class TaskBudget(BaseModel):
    """Resource contract attached to every planned task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_searches: int = Field(default=10, ge=0, le=100)
    max_sources: int = Field(default=12, ge=1, le=200)
    max_input_tokens: int = Field(default=60000, ge=100, le=10_000_000)
    max_output_tokens: int = Field(default=20000, ge=100, le=10_000_000)
    max_seconds: float = Field(default=300, gt=0, le=86400)
    max_retries: int = Field(default=2, ge=0, le=20)


class RunBudget(BaseModel):
    """Global circuit breakers shared by all workers in one run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_concurrent_workers: int = Field(default=5, ge=1, le=100)
    max_tasks: int = Field(default=12, ge=3, le=100)
    max_searches: int = Field(default=100, ge=1, le=10000)
    max_input_tokens: int = Field(default=500000, ge=1000, le=100_000_000)
    max_output_tokens: int = Field(default=200000, ge=1000, le=100_000_000)
    max_cost_usd: float = Field(default=0, ge=0, le=1_000_000)
    max_seconds: float = Field(default=1200, gt=0, le=604800)
    max_conflict_tasks: int = Field(default=2, ge=0, le=20)
    small_input_cost_per_million: float = Field(default=0, ge=0)
    small_output_cost_per_million: float = Field(default=0, ge=0)
    large_input_cost_per_million: float = Field(default=0, ge=0)
    large_output_cost_per_million: float = Field(default=0, ge=0)


class TaskBudgetUsage(BaseModel):
    """Public resource usage for one worker task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    searches: int = 0
    sources: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    elapsed_ms: float = 0


class BudgetSnapshot(BaseModel):
    """Non-sensitive run budget and current usage returned by the API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    status: Literal["active", "exhausted", "completed"] = "active"
    limits: RunBudget
    task_count: int = 0
    active_workers: int = 0
    peak_workers: int = 0
    searches: int = 0
    sources: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    estimated_cost_usd: float = 0
    elapsed_ms: float = 0
    exhausted_reason: str | None = None
    tasks: tuple[TaskBudgetUsage, ...] = ()
    started_at: datetime


@dataclass
class _TaskUsage:
    started: float | None = None
    elapsed_seconds: float = 0
    searches: int = 0
    sources: set[str] = field(default_factory=set)
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0


@dataclass
class _RunUsage:
    policy: RunBudget
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started: float = field(default_factory=monotonic)
    task_count: int = 0
    active_workers: int = 0
    peak_workers: int = 0
    searches: int = 0
    sources: set[str] = field(default_factory=set)
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    estimated_cost_usd: float = 0
    exhausted_reason: str | None = None
    completed: bool = False
    tasks: dict[str, _TaskUsage] = field(default_factory=dict)
    task_budgets: dict[str, TaskBudget] = field(default_factory=dict)


class BudgetLedger:
    """Thread-safe budget accounting used by parallel model and tool calls."""

    def __init__(self, default_policy: RunBudget | None = None) -> None:
        self.default_policy = default_policy or RunBudget()
        self._runs: dict[str, _RunUsage] = {}
        self._lock = RLock()

    def start_run(self, run_id: str, policy: RunBudget | None = None) -> None:
        with self._lock:
            self._runs.setdefault(run_id, _RunUsage(policy=policy or self.default_policy))

    def register_plan(
        self,
        run_id: str,
        task_count: int,
        task_budgets: Mapping[str, TaskBudget] | None = None,
    ) -> None:
        with self._lock:
            run = self._run(run_id)
            self._check_time(run, run_id)
            if task_count > run.policy.max_tasks:
                self._exhaust(
                    run,
                    f"plan contains {task_count} tasks; limit is {run.policy.max_tasks}",
                )
            run.task_count = task_count
            if task_budgets is not None:
                run.task_budgets = dict(task_budgets)

    def start_worker(self, run_id: str, task_id: str) -> None:
        with self._lock:
            run = self._run(run_id)
            self._check_time(run, run_id)
            if run.active_workers >= run.policy.max_concurrent_workers:
                raise BudgetExceeded(
                    f"run {run_id} concurrency limit is {run.policy.max_concurrent_workers}"
                )
            usage = run.tasks.setdefault(task_id, _TaskUsage())
            usage.started = monotonic()
            run.active_workers += 1
            run.peak_workers = max(run.peak_workers, run.active_workers)

    def finish_worker(self, run_id: str, task_id: str) -> None:
        with self._lock:
            run = self._run(run_id)
            usage = run.tasks.setdefault(task_id, _TaskUsage())
            if usage.started is not None:
                usage.elapsed_seconds += monotonic() - usage.started
                usage.started = None
            run.active_workers = max(0, run.active_workers - 1)

    def reserve_model_input(
        self,
        run_id: str,
        task_id: str,
        *,
        tier: Literal["small", "large"],
        tokens: int,
        task_budget: TaskBudget | None = None,
    ) -> None:
        budget = task_budget or self.task_budget(run_id, task_id)
        with self._lock:
            run = self._run(run_id)
            usage = run.tasks.setdefault(task_id, _TaskUsage())
            self._check_time(run, run_id)
            self._check_task_time(run, usage, budget, task_id)
            if usage.input_tokens + tokens > budget.max_input_tokens:
                self._exhaust(run, f"task {task_id} input token budget exceeded")
            if run.input_tokens + tokens > run.policy.max_input_tokens:
                self._exhaust(run, f"run {run_id} input token budget exceeded")
            usage.input_tokens += tokens
            usage.model_calls += 1
            run.input_tokens += tokens
            run.model_calls += 1
            run.estimated_cost_usd += self._token_cost(
                run.policy,
                tier,
                input_tokens=tokens,
            )
            self._check_cost(run, run_id)

    def record_model_output(
        self,
        run_id: str,
        task_id: str,
        *,
        tier: Literal["small", "large"],
        tokens: int,
        task_budget: TaskBudget | None = None,
    ) -> None:
        budget = task_budget or self.task_budget(run_id, task_id)
        with self._lock:
            run = self._run(run_id)
            usage = run.tasks.setdefault(task_id, _TaskUsage())
            if usage.output_tokens + tokens > budget.max_output_tokens:
                self._exhaust(run, f"task {task_id} output token budget exceeded")
            if run.output_tokens + tokens > run.policy.max_output_tokens:
                self._exhaust(run, f"run {run_id} output token budget exceeded")
            usage.output_tokens += tokens
            run.output_tokens += tokens
            run.estimated_cost_usd += self._token_cost(
                run.policy,
                tier,
                output_tokens=tokens,
            )
            self._check_cost(run, run_id)

    def reserve_search(
        self,
        run_id: str,
        task_id: str,
        task_budget: TaskBudget,
    ) -> None:
        with self._lock:
            run = self._run(run_id)
            usage = run.tasks.setdefault(task_id, _TaskUsage())
            self._check_time(run, run_id)
            self._check_task_time(run, usage, task_budget, task_id)
            if usage.searches >= task_budget.max_searches:
                self._exhaust(run, f"task {task_id} search budget exceeded")
            if run.searches >= run.policy.max_searches:
                self._exhaust(run, f"run {run_id} search budget exceeded")
            usage.searches += 1
            run.searches += 1

    def record_sources(
        self,
        run_id: str,
        task_id: str,
        source_uris: tuple[str, ...],
        task_budget: TaskBudget,
    ) -> None:
        with self._lock:
            run = self._run(run_id)
            usage = run.tasks.setdefault(task_id, _TaskUsage())
            additions = [uri for uri in source_uris if uri not in usage.sources]
            if len(usage.sources) + len(additions) > task_budget.max_sources:
                self._exhaust(run, f"task {task_id} source budget exceeded")
            usage.sources.update(additions)
            run.sources.update(additions)

    def complete_run(self, run_id: str) -> None:
        with self._lock:
            self._get_run(run_id).completed = True

    def task_budget(self, run_id: str, task_id: str) -> TaskBudget:
        """Return the registered contract or the conservative default contract."""

        with self._lock:
            run = self._get_run(run_id)
            return run.task_budgets.get(task_id, TaskBudget())

    def remaining_sources(self, run_id: str, task_id: str) -> int:
        """Return how many additional distinct sources a task may accept."""

        with self._lock:
            run = self._run(run_id)
            usage = run.tasks.setdefault(task_id, _TaskUsage())
            budget = run.task_budgets.get(task_id, TaskBudget())
            return max(0, budget.max_sources - len(usage.sources))

    def snapshot(self, run_id: str) -> BudgetSnapshot:
        with self._lock:
            run = self._get_run(run_id)
            elapsed = monotonic() - run.started
            tasks = tuple(
                TaskBudgetUsage(
                    task_id=task_id,
                    searches=usage.searches,
                    sources=len(usage.sources),
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    model_calls=usage.model_calls,
                    elapsed_ms=round(
                        (usage.elapsed_seconds + self._active_elapsed(usage)) * 1000,
                        3,
                    ),
                )
                for task_id, usage in sorted(run.tasks.items())
            )
            status: Literal["active", "exhausted", "completed"] = "active"
            if run.exhausted_reason:
                status = "exhausted"
            elif run.completed:
                status = "completed"
            return BudgetSnapshot(
                run_id=run_id,
                status=status,
                limits=run.policy,
                task_count=run.task_count,
                active_workers=run.active_workers,
                peak_workers=run.peak_workers,
                searches=run.searches,
                sources=len(run.sources),
                input_tokens=run.input_tokens,
                output_tokens=run.output_tokens,
                model_calls=run.model_calls,
                estimated_cost_usd=round(run.estimated_cost_usd, 8),
                elapsed_ms=round(elapsed * 1000, 3),
                exhausted_reason=run.exhausted_reason,
                tasks=tasks,
                started_at=run.started_at,
            )

    def _run(self, run_id: str) -> _RunUsage:
        run = self._get_run(run_id)
        if run.exhausted_reason:
            raise BudgetExceeded(run.exhausted_reason)
        return run

    def _get_run(self, run_id: str) -> _RunUsage:
        run = self._runs.get(run_id)
        if run is None:
            run = _RunUsage(policy=self.default_policy)
            self._runs[run_id] = run
        return run

    @staticmethod
    def _active_elapsed(usage: _TaskUsage) -> float:
        return monotonic() - usage.started if usage.started is not None else 0

    @staticmethod
    def _check_task_time(
        run: _RunUsage,
        usage: _TaskUsage,
        budget: TaskBudget,
        task_id: str,
    ) -> None:
        elapsed = usage.elapsed_seconds + BudgetLedger._active_elapsed(usage)
        if elapsed > budget.max_seconds:
            BudgetLedger._exhaust(run, f"task {task_id} time budget exceeded")

    @staticmethod
    def _check_time(run: _RunUsage, run_id: str) -> None:
        if monotonic() - run.started > run.policy.max_seconds:
            BudgetLedger._exhaust(run, f"run {run_id} time budget exceeded")

    @staticmethod
    def _check_cost(run: _RunUsage, run_id: str) -> None:
        if run.policy.max_cost_usd and run.estimated_cost_usd > run.policy.max_cost_usd:
            BudgetLedger._exhaust(run, f"run {run_id} estimated cost budget exceeded")

    @staticmethod
    def _token_cost(
        policy: RunBudget,
        tier: Literal["small", "large"],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        input_rate = (
            policy.small_input_cost_per_million
            if tier == "small"
            else policy.large_input_cost_per_million
        )
        output_rate = (
            policy.small_output_cost_per_million
            if tier == "small"
            else policy.large_output_cost_per_million
        )
        return (
            input_tokens * input_rate + output_tokens * output_rate
        ) / 1_000_000

    @staticmethod
    def _exhaust(run: _RunUsage, reason: str) -> None:
        run.exhausted_reason = reason
        raise BudgetExceeded(reason)
