"""Step 7 tests for recovery, SAGA, HITL, model routing, and observability."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from agents.model_router import DynamicModelRouter
from agents.researcher import ResearchOutput
from agents.runtime import AgentRuntime, run_agent_workflow
from agents.tool_selector import PlannedToolCall, ToolCallPlan
from agents.writer import ReportDraft, ReportSection
from core.config import Settings
from core.observability import AgentObservability
from core.recovery import RecoveryPolicy
from core.saga import (
    InMemorySagaJournal,
    SagaCoordinator,
    SagaExecutionError,
    SagaStatus,
    SagaStep,
)
from graph.state import (
    AgentName,
    HumanFeedback,
    ResearchRequirement,
    ResearchTask,
    ResearchType,
    TaskStatus,
    WorkflowStatus,
    create_initial_state,
)
from graph.workflow import TaskExecutionResult, build_workflow
from guardrail.middleware import GuardedToolRuntime, GuardrailMiddleware, HumanApprovalRequired
from memory.checkpoint import CheckpointManager
from rag.retriever import SourceType
from tools.registry import (
    StructuredTool,
    ToolContext,
    ToolEvidence,
    ToolRegistry,
    ToolResult,
    ToolRisk,
    ToolSpec,
)


class NeverCalledModel:
    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        raise AssertionError(f"unexpected model call for {schema.__name__}: {messages}")


class RecordingModel:
    def __init__(self) -> None:
        self.schemas: list[str] = []

    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        del messages
        self.schemas.append(schema.__name__)
        if schema is ResearchRequirement:
            return ResearchRequirement(
                objective="Research agent platforms",
                research_type=ResearchType.TECHNOLOGY,
                key_questions=("What changed?",),
            )
        if schema is ReportDraft:
            return ReportDraft(
                title="Agent report",
                executive_summary="Summary",
                sections=(ReportSection(heading="Finding", content="Content"),),
                markdown_report="# Agent report",
            )
        raise AssertionError(schema.__name__)


class LocalPlanner:
    def __init__(self) -> None:
        self.replanned: list[str] = []

    def analyze_impact(
        self,
        requirement: ResearchRequirement,
        tasks: list[ResearchTask],
        change_summary: str,
    ) -> tuple[str, ...]:
        del requirement, tasks, change_summary
        return ("source",)

    def replan_task(
        self,
        requirement: ResearchRequirement,
        task: ResearchTask,
        reason: str,
    ) -> ResearchTask:
        del requirement, reason
        self.replanned.append(task.task_id)
        return task.model_copy(update={"description": f"replanned:{task.description}"})


class ApprovalModel:
    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        assert messages
        if schema is ResearchRequirement:
            return ResearchRequirement(
                objective="Approval research",
                key_questions=("Is it approved?",),
            )
        if schema is ResearchOutput:
            return ResearchOutput(summary="approved source")
        if schema is ReportDraft:
            return ReportDraft(
                title="Approved report",
                executive_summary="Approved",
                sections=(ReportSection(heading="Result", content="Approved"),),
                markdown_report="# Approved report",
            )
        raise AssertionError(schema.__name__)


class RiskyInput(BaseModel):
    query: str


class RiskySearchTool(StructuredTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="risky.search",
                description="High risk test search.",
                input_schema=RiskyInput.model_json_schema(),
                capabilities=frozenset({"search", "web"}),
                risk=ToolRisk.HIGH,
            ),
            RiskyInput,
        )

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del arguments, context
        return ToolResult(
            tool_name=self.spec.name,
            evidences=(
                ToolEvidence(
                    evidence_id="evidence:approved",
                    source_type=SourceType.WEB,
                    source_uri="https://example.test/approved",
                    content="approved evidence",
                ),
            ),
        )


class RiskySelector:
    def select(self, user_goal: str, task: ResearchTask, available_tools: Any) -> ToolCallPlan:
        del user_goal, task, available_tools
        return ToolCallPlan(
            calls=(
                PlannedToolCall(
                    tool_name="risky.search",
                    arguments={"query": "approval"},
                ),
            )
        )


def test_transient_worker_failure_retries_and_completes() -> None:
    calls = 0

    def flaky_executor(request: Any) -> TaskExecutionResult:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("temporary timeout")
        return TaskExecutionResult(
            result_id=f"result:{request.task.task_id}",
            payload_ref=f"blackboard://results/result:{request.task.task_id}",
        )

    task = ResearchTask(
        task_id="retry_task",
        title="Retry task",
        description="Transient operation",
        assigned_agent=AgentName.RETRIEVER,
    )
    graph = build_workflow(
        {AgentName.RETRIEVER: flaky_executor},
        recovery_policy=RecoveryPolicy(max_tool_retries=2),
    )

    state = graph.invoke(create_initial_state("Research retries", [task]))

    assert calls == 3
    assert state["workflow_status"] == WorkflowStatus.COMPLETED
    assert state["tasks"][0].status == TaskStatus.COMPLETED


def test_high_risk_failure_is_checkpointed_for_human_intervention() -> None:
    def risky_executor(request: Any) -> TaskExecutionResult:
        del request
        raise HumanApprovalRequired("notification.send")

    task = ResearchTask(
        task_id="risky_task",
        title="Risky task",
        description="Needs approval",
        assigned_agent=AgentName.RESEARCHER,
    )
    graph = build_workflow(
        {AgentName.RESEARCHER: risky_executor},
        recovery_policy=RecoveryPolicy(),
    )

    state = graph.invoke(create_initial_state("Research safely", [task]))

    assert state["workflow_status"] == WorkflowStatus.WAITING_HUMAN
    assert state["tasks"][0].status == TaskStatus.WAITING_HUMAN
    assert state["human_requests"][0].approval_token == "tool:notification.send"


def test_human_feedback_invalidates_only_affected_dependency_closure() -> None:
    model = NeverCalledModel()
    runtime = AgentRuntime(small_model=model, large_model=model)
    planner = LocalPlanner()
    runtime.planner = planner  # type: ignore[assignment]
    tasks = [
        ResearchTask(
            task_id="source",
            title="Source",
            description="Collect source",
            status=TaskStatus.COMPLETED,
            assigned_agent=AgentName.RETRIEVER,
            result_ids=("result:source",),
        ),
        ResearchTask(
            task_id="verify",
            title="Verify",
            description="Verify source",
            dependencies=("source",),
            status=TaskStatus.COMPLETED,
            assigned_agent=AgentName.VERIFIER,
        ),
        ResearchTask(
            task_id="internal",
            title="Internal",
            description="Independent internal data",
            status=TaskStatus.COMPLETED,
            assigned_agent=AgentName.DATABASE,
        ),
        ResearchTask(
            task_id="write",
            title="Write",
            description="Write report",
            dependencies=("verify", "internal"),
            status=TaskStatus.COMPLETED,
            assigned_agent=AgentName.WRITER,
        ),
    ]
    state = create_initial_state("Old goal", tasks)
    state["requirements"] = ResearchRequirement(
        objective="Old goal",
        key_questions=("Question",),
    )

    update = runtime.apply_human_feedback(
        state,
        HumanFeedback(
            revised_goal="New goal",
            affected_task_ids=("source",),
            additional_constraints=("Use 2026 data",),
        ),
    )
    updated = {task.task_id: task for task in update["tasks"]}

    assert set(updated) == {"source", "verify", "write"}
    assert "internal" not in planner.replanned
    assert all(task.status == TaskStatus.PENDING for task in updated.values())
    assert updated["source"].result_ids == ()
    assert update["user_goal"] == "New goal"


def test_paused_supervisor_feedback_redispatches_unexecuted_parallel_tasks() -> None:
    model = RecordingModel()
    runtime = AgentRuntime(small_model=model, large_model=model)
    tasks = [
        ResearchTask(
            task_id="topic_a",
            title="Topic A",
            description="Research topic A",
            assigned_agent=AgentName.RESEARCHER,
        ),
        ResearchTask(
            task_id="topic_b",
            title="Topic B",
            description="Research topic B",
            assigned_agent=AgentName.DATABASE,
        ),
    ]
    manager = CheckpointManager(Settings(checkpoint_backend="memory"))
    thread_id = "feedback-at-supervisor-checkpoint"

    with manager.open() as checkpointer:
        paused = run_agent_workflow(
            "Original goal",
            runtime=runtime,
            tasks=tasks,
            checkpointer=checkpointer,
            thread_id=thread_id,
            interrupt_after=["supervisor"],
        )

    assert all(task.status == TaskStatus.RUNNING for task in paused["tasks"])

    planner = LocalPlanner()
    runtime.planner = planner  # type: ignore[assignment]
    with manager.open() as checkpointer:
        revised = run_agent_workflow(
            None,
            runtime=runtime,
            checkpointer=checkpointer,
            thread_id=thread_id,
            resume=True,
            human_feedback=HumanFeedback(
                revised_goal="Revised goal",
                affected_task_ids=("topic_a",),
            ),
            interrupt_after=["supervisor"],
        )

    dispatches = [
        checkpoint.task_ids
        for checkpoint in revised["checkpoints"]
        if checkpoint.stage == "tasks_dispatched"
    ]
    assert revised["user_goal"] == "Revised goal"
    assert planner.replanned == ["topic_a"]
    assert dispatches[-1] == ("topic_a", "topic_b")
    assert all(task.status == TaskStatus.RUNNING for task in revised["tasks"])


def test_saga_compensates_started_steps_in_reverse_order() -> None:
    events: list[str] = []
    journal = InMemorySagaJournal()
    coordinator = SagaCoordinator(journal)

    with pytest.raises(SagaExecutionError):
        coordinator.execute(
            "saga-1",
            "run-1",
            (
                SagaStep(
                    name="save",
                    action=lambda: events.append("save"),
                    compensate=lambda: events.append("undo-save"),
                ),
                SagaStep(
                    name="notify",
                    action=lambda: (_ for _ in ()).throw(RuntimeError("notify failed")),
                    compensate=lambda: events.append("undo-notify"),
                ),
            ),
        )

    assert events == ["save", "undo-notify", "undo-save"]
    assert [record.status for record in journal.list("saga-1")] == [
        SagaStatus.COMPENSATED,
        SagaStatus.COMPENSATED,
    ]


def test_saga_skips_completed_idempotent_steps_on_resume() -> None:
    calls = 0
    coordinator = SagaCoordinator()

    def action() -> None:
        nonlocal calls
        calls += 1

    step = SagaStep(name="save", action=action, compensate=lambda: None)
    coordinator.execute("saga-resume", "run-1", (step,))
    coordinator.execute("saga-resume", "run-1", (step,))

    assert calls == 1


def test_dynamic_model_router_selects_tier_and_records_token_usage() -> None:
    small = RecordingModel()
    large = RecordingModel()
    observability = AgentObservability()
    router = DynamicModelRouter(
        small,
        large,
        observability=observability,
        complexity_chars=500,
    )

    router.invoke(ResearchRequirement, [HumanMessage(content="extract requirement")])
    router.invoke(ReportDraft, [HumanMessage(content="write report")])
    traces = observability.snapshot()

    assert small.schemas == ["ResearchRequirement"]
    assert large.schemas == ["ReportDraft"]
    assert [trace.metadata["tier"] for trace in traces] == ["small", "large"]
    assert all(trace.input_tokens > 0 and trace.output_tokens > 0 for trace in traces)


def test_checkpoint_resume_applies_tool_approval_without_full_rerun() -> None:
    model = ApprovalModel()
    registry = ToolRegistry()
    registry.register(RiskySearchTool())
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        tool_runtime=GuardedToolRuntime(registry, GuardrailMiddleware()),
        tool_selector=RiskySelector(),
        recovery_policy=RecoveryPolicy(),
    )
    tasks = [
        ResearchTask(
            task_id="retrieve",
            title="Retrieve",
            description="Retrieve approved source",
            assigned_agent=AgentName.RETRIEVER,
        ),
        ResearchTask(
            task_id="write",
            title="Write",
            description="Write report",
            dependencies=("retrieve",),
            assigned_agent=AgentName.WRITER,
        ),
    ]
    manager = CheckpointManager(Settings(checkpoint_backend="memory"))

    with manager.open() as checkpointer:
        paused = run_agent_workflow(
            "Approval research",
            runtime=runtime,
            tasks=tasks,
            checkpointer=checkpointer,
            thread_id="approval-thread",
        )
    assert paused["workflow_status"] == WorkflowStatus.WAITING_HUMAN

    with manager.open() as checkpointer:
        resumed = run_agent_workflow(
            None,
            runtime=runtime,
            checkpointer=checkpointer,
            thread_id="approval-thread",
            resume=True,
            human_feedback=HumanFeedback(approvals=frozenset({"tool:risky.search"})),
        )

    assert resumed["workflow_status"] == WorkflowStatus.COMPLETED
    assert resumed["final_report"] == "# Approved report"
    assert resumed["tasks"][0].attempts == 2
