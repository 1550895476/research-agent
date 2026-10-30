"""Typed state, task DAG models, and merge reducers for the research graph."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentName(StrEnum):
    """Worker nodes that can receive a planned task."""

    RESEARCHER = "researcher"
    RETRIEVER = "retriever"
    DATABASE = "database"
    VERIFIER = "verifier"
    WRITER = "writer"


class TaskStatus(StrEnum):
    """Lifecycle of one task in the dynamic research DAG."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    WAITING_HUMAN = "waiting_human"


class WorkflowStatus(StrEnum):
    """Lifecycle of a complete research run."""

    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_HUMAN = "waiting_human"


class ResearchType(StrEnum):
    """High-level intent categories used by the Planner prompt."""

    INDUSTRY = "industry"
    COMPETITIVE = "competitive"
    TECHNOLOGY = "technology"
    GENERAL = "general"


class ResearchRequirement(BaseModel):
    """Structured interpretation of the user's natural-language research goal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: str = Field(min_length=1)
    research_type: ResearchType = ResearchType.GENERAL
    key_questions: tuple[str, ...] = Field(min_length=1)
    scope_in: tuple[str, ...] = ()
    scope_out: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    deliverables: tuple[str, ...] = ("structured research report",)
    language: str = "zh-CN"


class ResearchTask(BaseModel):
    """A schedulable unit of work and its dependency/status metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    initial_search_queries: tuple[str, ...] = Field(default=(), max_length=10)
    completion_criteria: tuple[str, ...] = Field(default=(), max_length=10)
    source_domains: tuple[str, ...] = Field(default=(), max_length=20)
    source_url_prefixes: tuple[str, ...] = Field(default=(), max_length=20)
    dependencies: tuple[str, ...] = ()
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: AgentName
    attempts: int = Field(default=0, ge=0)
    result_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    quality_score: int | None = Field(default=None, ge=0, le=100)
    quality_complete: bool | None = None
    search_count: int = Field(default=0, ge=0)
    quality_gaps: tuple[str, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def validate_dependencies(self) -> ResearchTask:
        if self.task_id in self.dependencies:
            raise ValueError(f"task {self.task_id!r} cannot depend on itself")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError(f"task {self.task_id!r} contains duplicate dependencies")
        if len(self.initial_search_queries) != len(set(self.initial_search_queries)):
            raise ValueError(f"task {self.task_id!r} contains duplicate search queries")
        if len(self.source_domains) != len(set(self.source_domains)):
            raise ValueError(f"task {self.task_id!r} contains duplicate source domains")
        if len(self.source_url_prefixes) != len(set(self.source_url_prefixes)):
            raise ValueError(f"task {self.task_id!r} contains duplicate source URL prefixes")
        return self


class EvidenceReference(BaseModel):
    """Small Blackboard reference to evidence stored outside graph messages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    task_id: str
    store_key: str


class ResultReference(BaseModel):
    """Small Blackboard reference to an intermediate result payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result_id: str
    task_id: str
    agent: AgentName
    payload_ref: str
    evidence_ids: tuple[str, ...] = ()


class WorkflowError(BaseModel):
    """Structured failure record suitable for later retry/re-planning logic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    stage: str
    task_id: str | None = None
    retryable: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkerStatusEvent(BaseModel):
    """Append-only task outcome emitted by a parallel worker branch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    agent: AgentName
    outcome: Literal["completed", "replan", "waiting_human", "failed"]
    reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CheckpointReference(BaseModel):
    """Logical execution marker; durable LangGraph checkpoints arrive in Step 4."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str
    task_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReplanRequest(BaseModel):
    """A recoverable task failure that should be locally revised by Planner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    reason: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HumanRequest(BaseModel):
    """Checkpointed approval or clarification request surfaced to the user."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    reason: str
    approval_token: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HumanFeedback(BaseModel):
    """Incremental user update applied to a paused/checkpointed workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revised_goal: str | None = Field(default=None, min_length=1, max_length=10000)
    additional_constraints: tuple[str, ...] = ()
    affected_task_ids: tuple[str, ...] = ()
    approvals: frozenset[str] = frozenset()


def merge_tasks(
    current: list[ResearchTask] | None,
    updates: list[ResearchTask] | None,
) -> list[ResearchTask]:
    """Merge concurrent task updates by task_id while preserving plan order."""

    merged = {task.task_id: task for task in current or []}
    order = [task.task_id for task in current or []]
    for task in updates or []:
        if task.task_id not in merged:
            order.append(task.task_id)
        merged[task.task_id] = task
    return [merged[task_id] for task_id in order]


def merge_references(
    current: dict[str, Any] | None,
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge Blackboard references emitted by parallel nodes."""

    return {**(current or {}), **(updates or {})}


def append_items(current: list[Any] | None, updates: list[Any] | None) -> list[Any]:
    """Append-only reducer for errors and logical checkpoint markers."""

    return [*(current or []), *(updates or [])]


def union_items(current: set[str] | None, updates: set[str] | None) -> set[str]:
    """Union reducer for approvals and handled recovery request identifiers."""

    return set(current or set()).union(updates or set())


class AgentState(TypedDict):
    """Unified state shared by every LangGraph node.

    Large source documents and generated payloads are intentionally absent. The graph
    carries only task metadata and Blackboard identifiers.
    """

    user_goal: str
    # Conversation metadata is deliberately separate from ``user_goal``. Follow-up
    # turns enrich the internal goal with the previous answer, while API clients
    # should continue to see only the user's latest message.
    conversation_id: str
    user_message: str
    parent_run_id: str | None
    selected_model: str | None
    run_id: str
    user_id: str
    requirements: ResearchRequirement | None
    tasks: Annotated[list[ResearchTask], merge_tasks]
    current_task: str | None
    evidences: Annotated[dict[str, EvidenceReference], merge_references]
    intermediate_results: Annotated[dict[str, ResultReference], merge_references]
    final_report: str | None
    errors: Annotated[list[WorkflowError], append_items]
    checkpoints: Annotated[list[CheckpointReference], append_items]
    replan_requests: Annotated[list[ReplanRequest], append_items]
    handled_replan_ids: Annotated[set[str], union_items]
    human_requests: Annotated[list[HumanRequest], append_items]
    status_events: Annotated[list[WorkerStatusEvent], append_items]
    approvals: Annotated[set[str], union_items]
    workflow_status: WorkflowStatus
    next_agent: str | None
    dispatched_task_ids: list[str]


class TaskDAG:
    """Validated view of tasks with dependency-aware scheduling helpers."""

    def __init__(self, tasks: Iterable[ResearchTask]) -> None:
        self._tasks = tuple(tasks)
        self._by_id = {task.task_id: task for task in self._tasks}
        if len(self._tasks) != len(self._by_id):
            raise ValueError("task_id values must be unique")
        self._validate_dependencies_exist()
        self._validate_acyclic()

    @property
    def tasks(self) -> tuple[ResearchTask, ...]:
        return self._tasks

    def get(self, task_id: str) -> ResearchTask:
        try:
            return self._by_id[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task_id: {task_id}") from exc

    def ready_tasks(self) -> list[ResearchTask]:
        """Return all pending tasks whose dependencies have completed."""

        return [
            task
            for task in self._tasks
            if task.status == TaskStatus.PENDING
            and all(
                self._by_id[dependency].status == TaskStatus.COMPLETED
                for dependency in task.dependencies
            )
        ]

    def blocked_updates(self) -> list[ResearchTask]:
        """Return every transitively blocked pending task in dependency order."""

        terminal_failure = {TaskStatus.FAILED, TaskStatus.BLOCKED}
        effective_status = {task.task_id: task.status for task in self._tasks}
        blocked: dict[str, ResearchTask] = {}
        changed = True
        while changed:
            changed = False
            for task in self._tasks:
                if effective_status[task.task_id] != TaskStatus.PENDING:
                    continue
                if any(
                    effective_status[dependency] in terminal_failure
                    for dependency in task.dependencies
                ):
                    effective_status[task.task_id] = TaskStatus.BLOCKED
                    blocked[task.task_id] = task.model_copy(
                        update={
                            "status": TaskStatus.BLOCKED,
                            "error": "dependency failed or was blocked",
                        }
                    )
                    changed = True
        return [blocked[task.task_id] for task in self._tasks if task.task_id in blocked]

    def all_completed(self) -> bool:
        return bool(self._tasks) and all(
            task.status == TaskStatus.COMPLETED for task in self._tasks
        )

    def has_terminal_failure(self) -> bool:
        return any(task.status in {TaskStatus.FAILED, TaskStatus.BLOCKED} for task in self._tasks)

    def _validate_dependencies_exist(self) -> None:
        for task in self._tasks:
            missing = set(task.dependencies) - self._by_id.keys()
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"task {task.task_id!r} has missing dependencies: {missing_list}")

    def _validate_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError(f"task DAG contains a cycle at {task_id!r}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in self._by_id[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task in self._tasks:
            visit(task.task_id)


def create_initial_state(
    user_goal: str,
    tasks: Iterable[ResearchTask] | None = None,
    *,
    run_id: str | None = None,
    user_id: str = "anonymous",
    conversation_id: str | None = None,
    user_message: str | None = None,
    parent_run_id: str | None = None,
    selected_model: str | None = None,
) -> AgentState:
    """Validate user input and create a complete graph state."""

    normalized_goal = " ".join(user_goal.split())
    if not normalized_goal:
        raise ValueError("user_goal must not be empty")
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id must not be empty")
    normalized_run_id = (run_id or str(uuid4())).strip()
    normalized_message = " ".join((user_message or user_goal).split())
    normalized_conversation_id = (conversation_id or normalized_run_id).strip()
    if not normalized_message:
        raise ValueError("user_message must not be empty")
    if not normalized_conversation_id:
        raise ValueError("conversation_id must not be empty")
    planned_tasks = list(tasks or [])
    if planned_tasks:
        TaskDAG(planned_tasks)
    return AgentState(
        user_goal=normalized_goal,
        conversation_id=normalized_conversation_id,
        user_message=normalized_message,
        parent_run_id=parent_run_id.strip() if parent_run_id else None,
        selected_model=selected_model.strip() if selected_model else None,
        run_id=normalized_run_id,
        user_id=normalized_user_id,
        requirements=None,
        tasks=planned_tasks,
        current_task=None,
        evidences={},
        intermediate_results={},
        final_report=None,
        errors=[],
        checkpoints=[],
        replan_requests=[],
        handled_replan_ids=set(),
        human_requests=[],
        status_events=[],
        approvals=set(),
        workflow_status=WorkflowStatus.PLANNING,
        next_agent=None,
        dispatched_task_ids=[],
    )
