"""Integration tests for parallel LangGraph dispatch and failure propagation."""

from threading import Barrier, Lock, get_ident

from core.recovery import RecoveryPolicy
from graph.state import (
    AgentName,
    ResearchTask,
    TaskStatus,
    WorkflowStatus,
    create_initial_state,
)
from graph.workflow import (
    TaskExecutionRequest,
    TaskExecutionResult,
    build_workflow,
    run_research_workflow,
)
from guardrail.middleware import HumanApprovalRequired


def test_default_workflow_completes_dependency_dag() -> None:
    state = run_research_workflow("研究新能源汽车市场")

    assert state["workflow_status"] == WorkflowStatus.COMPLETED
    assert all(task.status == TaskStatus.COMPLETED for task in state["tasks"])
    assert len(state["intermediate_results"]) == 5
    assert state["final_report"] is not None

    dispatches = [
        checkpoint.task_ids
        for checkpoint in state["checkpoints"]
        if checkpoint.stage == "tasks_dispatched"
    ]
    assert dispatches == [
        ("research_overview", "retrieve_sources", "database_context"),
        ("verify_findings",),
        ("write_report",),
    ]


def test_first_dispatch_batch_runs_concurrently() -> None:
    barrier = Barrier(3, timeout=2)
    lock = Lock()
    thread_ids: set[int] = set()

    def synchronized_executor(request: TaskExecutionRequest) -> TaskExecutionResult:
        with lock:
            thread_ids.add(get_ident())
        barrier.wait()
        return TaskExecutionResult(
            result_id=f"result:{request.task.task_id}",
            payload_ref=f"blackboard://results/result:{request.task.task_id}",
        )

    state = run_research_workflow(
        "研究新能源汽车市场",
        executors={
            AgentName.RESEARCHER: synchronized_executor,
            AgentName.RETRIEVER: synchronized_executor,
            AgentName.DATABASE: synchronized_executor,
        },
    )

    assert state["workflow_status"] == WorkflowStatus.COMPLETED
    assert len(thread_ids) == 3


def test_parallel_tasks_for_same_agent_fan_in_once_without_state_conflict() -> None:
    barrier = Barrier(2, timeout=2)

    def synchronized_executor(request: TaskExecutionRequest) -> TaskExecutionResult:
        barrier.wait()
        return TaskExecutionResult(
            result_id=f"result:{request.task.task_id}",
            payload_ref=f"blackboard://results/result:{request.task.task_id}",
        )

    tasks = [
        ResearchTask(
            task_id=f"research_{index}",
            title=f"Research {index}",
            description="Independent research task",
            assigned_agent=AgentName.RESEARCHER,
        )
        for index in range(2)
    ]

    state = run_research_workflow(
        "Research two independent topics",
        tasks,
        executors={AgentName.RESEARCHER: synchronized_executor},
    )

    assert state["workflow_status"] == WorkflowStatus.COMPLETED
    assert {event.task_id for event in state["status_events"]} == {
        "research_0",
        "research_1",
    }
    assert all(event.outcome == "completed" for event in state["status_events"])


def test_worker_persists_research_quality_metadata_on_task() -> None:
    def quality_executor(request: TaskExecutionRequest) -> TaskExecutionResult:
        return TaskExecutionResult(
            result_id=f"result:{request.task.task_id}",
            payload_ref=f"blackboard://results/result:{request.task.task_id}",
            quality_score=88,
            quality_complete=True,
            search_count=7,
            quality_gaps=(),
        )

    task = ResearchTask(
        task_id="quality_task",
        title="Quality task",
        description="Research with a quality gate",
        assigned_agent=AgentName.RESEARCHER,
    )
    state = run_research_workflow(
        "Research quality",
        [task],
        executors={AgentName.RESEARCHER: quality_executor},
    )

    completed = state["tasks"][0]
    assert completed.quality_score == 88
    assert completed.quality_complete is True
    assert completed.search_count == 7


def test_parallel_human_requests_preserve_every_branch_and_one_global_status() -> None:
    barrier = Barrier(2, timeout=2)

    def approval_executor(request: TaskExecutionRequest) -> TaskExecutionResult:
        barrier.wait()
        raise HumanApprovalRequired(f"approval.{request.task.task_id}")

    tasks = [
        ResearchTask(
            task_id="external_write",
            title="External write",
            description="Needs approval",
            assigned_agent=AgentName.RESEARCHER,
        ),
        ResearchTask(
            task_id="database_write",
            title="Database write",
            description="Needs approval",
            assigned_agent=AgentName.DATABASE,
        ),
    ]
    graph = build_workflow(
        {
            AgentName.RESEARCHER: approval_executor,
            AgentName.DATABASE: approval_executor,
        },
        recovery_policy=RecoveryPolicy(),
    )

    state = graph.invoke(create_initial_state("Research with approvals", tasks))

    assert state["workflow_status"] == WorkflowStatus.WAITING_HUMAN
    assert all(task.status == TaskStatus.WAITING_HUMAN for task in state["tasks"])
    assert {request.task_id for request in state["human_requests"]} == {
        "external_write",
        "database_write",
    }
    waiting_events = [
        event for event in state["status_events"] if event.outcome == "waiting_human"
    ]
    assert {event.task_id for event in waiting_events} == {
        "external_write",
        "database_write",
    }


def test_worker_failure_blocks_dependent_tasks() -> None:
    def selectively_failing_researcher(
        request: TaskExecutionRequest,
    ) -> TaskExecutionResult:
        if request.task.task_id == "retrieve_sources":
            raise RuntimeError(f"search failed for {request.task.task_id}")
        return TaskExecutionResult(
            result_id=f"result:{request.task.task_id}",
            payload_ref=f"blackboard://results/result:{request.task.task_id}",
        )

    state = run_research_workflow(
        "研究新能源汽车市场",
        executors={AgentName.RESEARCHER: selectively_failing_researcher},
    )
    statuses = {task.task_id: task.status for task in state["tasks"]}

    assert state["workflow_status"] == WorkflowStatus.FAILED
    assert statuses["retrieve_sources"] == TaskStatus.FAILED
    assert statuses["verify_findings"] == TaskStatus.BLOCKED
    assert statuses["write_report"] == TaskStatus.BLOCKED
    assert any(error.code == "TASK_EXECUTION_FAILED" for error in state["errors"])
