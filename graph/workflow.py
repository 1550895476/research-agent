"""LangGraph assembly for dependency-aware, parallel research task routing."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field

from core.recovery import RecoveryAction, RecoveryPolicy
from graph.state import (
    AgentName,
    AgentState,
    CheckpointReference,
    HumanRequest,
    ReplanRequest,
    ResearchRequirement,
    ResearchTask,
    ResultReference,
    TaskDAG,
    TaskStatus,
    WorkerStatusEvent,
    WorkflowError,
    WorkflowStatus,
    create_initial_state,
)


class TaskExecutionRequest(BaseModel):
    """Input passed from a routed LangGraph worker node to an executor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_goal: str
    run_id: str
    user_id: str
    task: ResearchTask
    requirements: ResearchRequirement | None = None
    approvals: frozenset[str] = frozenset()


class TaskExecutionResult(BaseModel):
    """Reference-only output returned by a worker executor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result_id: str
    payload_ref: str
    evidence_ids: tuple[str, ...] = ()
    quality_score: int | None = Field(default=None, ge=0, le=100)
    quality_complete: bool | None = None
    search_count: int = Field(default=0, ge=0)
    quality_gaps: tuple[str, ...] = ()


TaskExecutor = Callable[[TaskExecutionRequest], TaskExecutionResult]


def build_default_research_plan(user_goal: str) -> list[ResearchTask]:
    """Build an offline deterministic DAG until the LLM Planner arrives in Step 3."""

    return [
        ResearchTask(
            task_id="research_overview",
            title="研究范围与行业概览",
            description=(
                f"自主选择网页、知识库和可用数据库工具，研究范围、关键概念与行业背景："
                f"{user_goal}"
            ),
            assigned_agent=AgentName.RESEARCHER,
        ),
        ResearchTask(
            task_id="retrieve_sources",
            title="市场与公开证据研究",
            description=f"自主检索多来源公开证据并形成可验证的市场结论：{user_goal}",
            assigned_agent=AgentName.RESEARCHER,
        ),
        ResearchTask(
            task_id="database_context",
            title="技术与内部信息研究",
            description=(
                f"结合公开资料、知识库及可用的只读数据库研究技术和内部信息：{user_goal}"
            ),
            assigned_agent=AgentName.RESEARCHER,
        ),
        ResearchTask(
            task_id="verify_findings",
            title="证据交叉验证",
            description="对阶段性结论进行多来源一致性和冲突检查",
            dependencies=("research_overview", "retrieve_sources", "database_context"),
            assigned_agent=AgentName.VERIFIER,
        ),
        ResearchTask(
            task_id="write_report",
            title="结构化报告生成",
            description="根据已验证结果生成结构化研究报告",
            dependencies=("verify_findings",),
            assigned_agent=AgentName.WRITER,
        ),
    ]


def offline_executor(request: TaskExecutionRequest) -> TaskExecutionResult:
    """Return a deterministic Blackboard reference without calling a model or tool."""

    result_id = f"result:{request.task.task_id}"
    return TaskExecutionResult(
        result_id=result_id,
        payload_ref=f"blackboard://results/{result_id}",
    )


def requirement_parser_node(state: AgentState) -> dict[str, Any]:
    """Normalize the goal before planning; semantic parsing is implemented in Step 3."""

    normalized_goal = " ".join(state["user_goal"].split())
    if not normalized_goal:
        raise ValueError("user_goal must not be empty")
    return {
        "user_goal": normalized_goal,
        "workflow_status": WorkflowStatus.PLANNING,
        "checkpoints": [CheckpointReference(stage="requirement_parsed")],
    }


def planner_node(state: AgentState) -> dict[str, Any]:
    """Validate an injected DAG or create the deterministic Step 2 plan."""

    tasks = state["tasks"] or build_default_research_plan(state["user_goal"])
    TaskDAG(tasks)
    return {
        "tasks": tasks,
        "workflow_status": WorkflowStatus.RUNNING,
        "next_agent": "supervisor",
        "checkpoints": [
            CheckpointReference(
                stage="plan_created",
                task_ids=tuple(task.task_id for task in tasks),
            )
        ],
    }


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Select every currently runnable task and prepare a parallel dispatch batch."""

    if any(task.status == TaskStatus.WAITING_HUMAN for task in state["tasks"]):
        return {
            "workflow_status": WorkflowStatus.WAITING_HUMAN,
            "next_agent": "human_intervention",
            "dispatched_task_ids": [],
        }

    dag = TaskDAG(state["tasks"])
    blocked_updates = dag.blocked_updates()
    if blocked_updates:
        return {
            "tasks": blocked_updates,
            "workflow_status": WorkflowStatus.FAILED,
            "next_agent": "workflow_failed",
            "dispatched_task_ids": [],
            "errors": [
                WorkflowError(
                    code="DEPENDENCY_FAILED",
                    message="one or more tasks were blocked by a failed dependency",
                    stage="supervisor",
                )
            ],
        }

    if dag.all_completed():
        return {
            "workflow_status": WorkflowStatus.COMPLETED,
            "next_agent": "finalize",
            "dispatched_task_ids": [],
        }

    ready = dag.ready_tasks()
    if ready:
        running = [
            task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "attempts": task.attempts + 1,
                    "error": None,
                }
            )
            for task in ready
        ]
        task_ids = [task.task_id for task in running]
        return {
            "tasks": running,
            "workflow_status": WorkflowStatus.RUNNING,
            "next_agent": "parallel_dispatch",
            "dispatched_task_ids": task_ids,
            "checkpoints": [
                CheckpointReference(stage="tasks_dispatched", task_ids=tuple(task_ids))
            ],
        }

    running_tasks = [task for task in dag.tasks if task.status == TaskStatus.RUNNING]
    if running_tasks:
        return {
            "workflow_status": WorkflowStatus.RUNNING,
            "next_agent": "supervisor",
            "dispatched_task_ids": [],
        }

    return {
        "workflow_status": WorkflowStatus.FAILED,
        "next_agent": "workflow_failed",
        "dispatched_task_ids": [],
        "errors": [
            WorkflowError(
                code="DAG_STALLED",
                message="the task DAG has no runnable or running tasks",
                stage="supervisor",
            )
        ],
    }


def route_supervisor(state: AgentState) -> str | list[Send]:
    """Route to finalization/failure or fan out runnable tasks with LangGraph Send."""

    if state["dispatched_task_ids"]:
        dag = TaskDAG(state["tasks"])
        sends: list[Send] = []
        for task_id in state["dispatched_task_ids"]:
            task = dag.get(task_id)
            sends.append(
                Send(
                    "worker",
                    {
                        "user_goal": state["user_goal"],
                        "run_id": state["run_id"],
                        "user_id": state["user_id"],
                        "requirements": state["requirements"],
                        "tasks": state["tasks"],
                        "current_task": task.task_id,
                        "approvals": state["approvals"],
                    },
                )
            )
        return sends
    if state["workflow_status"] == WorkflowStatus.COMPLETED:
        return "finalize"
    if state["workflow_status"] == WorkflowStatus.FAILED:
        return "workflow_failed"
    if state["workflow_status"] == WorkflowStatus.WAITING_HUMAN:
        return "human_intervention"
    return "supervisor"


def make_worker_node(
    executors: Mapping[AgentName, TaskExecutor],
    recovery_policy: RecoveryPolicy | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    """Dispatch any task to its executor and emit reducer-safe task-level updates."""

    def worker(state: AgentState) -> dict[str, Any]:
        task_id = state["current_task"]
        if task_id is None:
            raise ValueError("worker received no current_task")
        task = TaskDAG(state["tasks"]).get(task_id)
        agent = task.assigned_agent
        executor = executors.get(agent, offline_executor)
        policy = recovery_policy or RecoveryPolicy.disabled()
        request = TaskExecutionRequest(
            user_goal=state["user_goal"],
            run_id=state["run_id"],
            user_id=state["user_id"],
            task=task,
            requirements=state.get("requirements"),
            approvals=frozenset(state["approvals"]),
        )
        failure_count = 0
        result: TaskExecutionResult | None = None
        last_error: Exception | None = None
        last_decision = None
        while True:
            try:
                result = executor(request)
                break
            except Exception as exc:  # noqa: BLE001 - classified into graph state
                failure_count += 1
                decision = policy.decide(exc, task, failure_count=failure_count)
                last_error = exc
                last_decision = decision
                if decision.action == RecoveryAction.RETRY:
                    if decision.retry_delay_seconds:
                        time.sleep(decision.retry_delay_seconds)
                    continue
                if decision.action == RecoveryAction.REPLAN:
                    request_to_replan = ReplanRequest(
                        task_id=task_id,
                        reason=decision.reason,
                    )
                    return {
                        "tasks": [
                            task.model_copy(
                                update={"status": TaskStatus.PENDING, "error": decision.reason}
                            )
                        ],
                        "replan_requests": [request_to_replan],
                        "status_events": [
                            WorkerStatusEvent(
                                task_id=task_id,
                                agent=agent,
                                outcome="replan",
                                reason=decision.reason,
                            )
                        ],
                        "errors": [
                            WorkflowError(
                                code="REPLAN_REQUIRED",
                                message=decision.reason,
                                stage=agent.value,
                                task_id=task_id,
                                retryable=True,
                            )
                        ],
                        "checkpoints": [
                            CheckpointReference(
                                stage="task_replan_requested",
                                task_ids=(task_id,),
                            )
                        ],
                    }
                if decision.action == RecoveryAction.HUMAN:
                    human_request = HumanRequest(
                        task_id=task_id,
                        reason=decision.reason,
                        approval_token=getattr(exc, "approval_token", None),
                    )
                    return {
                        "tasks": [
                            task.model_copy(
                                update={
                                    "status": TaskStatus.WAITING_HUMAN,
                                    "error": decision.reason,
                                }
                            )
                        ],
                        "human_requests": [human_request],
                        "status_events": [
                            WorkerStatusEvent(
                                task_id=task_id,
                                agent=agent,
                                outcome="waiting_human",
                                reason=decision.reason,
                            )
                        ],
                        "errors": [
                            WorkflowError(
                                code="HUMAN_INTERVENTION_REQUIRED",
                                message=decision.reason,
                                stage=agent.value,
                                task_id=task_id,
                            )
                        ],
                        "checkpoints": [
                            CheckpointReference(
                                stage="waiting_human",
                                task_ids=(task_id,),
                            )
                        ],
                    }
                break
        if result is None:
            assert last_error is not None and last_decision is not None
            failed_task = task.model_copy(
                update={"status": TaskStatus.FAILED, "error": str(last_error)}
            )
            return {
                "tasks": [failed_task],
                "status_events": [
                    WorkerStatusEvent(
                        task_id=task_id,
                        agent=agent,
                        outcome="failed",
                        reason=last_decision.reason,
                    )
                ],
                "errors": [
                    WorkflowError(
                        code="TASK_EXECUTION_FAILED",
                        message=last_decision.reason,
                        stage=agent.value,
                        task_id=task_id,
                        retryable=True,
                    )
                ],
                "checkpoints": [
                    CheckpointReference(stage="task_failed", task_ids=(task_id,))
                ],
            }

        result_ref = ResultReference(
            result_id=result.result_id,
            task_id=task_id,
            agent=agent,
            payload_ref=result.payload_ref,
            evidence_ids=result.evidence_ids,
        )
        completed_task = task.model_copy(
            update={
                "status": TaskStatus.COMPLETED,
                "result_ids": (*task.result_ids, result.result_id),
                "evidence_ids": tuple(dict.fromkeys((*task.evidence_ids, *result.evidence_ids))),
                "quality_score": result.quality_score,
                "quality_complete": result.quality_complete,
                "search_count": result.search_count,
                "quality_gaps": result.quality_gaps,
                "error": None,
            }
        )
        return {
            "tasks": [completed_task],
            "intermediate_results": {result.result_id: result_ref},
            "status_events": [
                WorkerStatusEvent(
                    task_id=task_id,
                    agent=agent,
                    outcome="completed",
                )
            ],
            "checkpoints": [
                CheckpointReference(stage="task_completed", task_ids=(task_id,))
            ],
        }

    worker.__name__ = "worker_node"
    return worker


def _normalize_interrupt_nodes(nodes: list[str] | None) -> list[str] | None:
    """Map legacy Agent node pause targets to the unified worker boundary."""

    if nodes is None:
        return None
    agent_nodes = {agent.value for agent in AgentName}
    normalized = ["worker" if node in agent_nodes else node for node in nodes]
    return list(dict.fromkeys(normalized))


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Create a Step 2 completion summary; the real report writer arrives in Step 3."""

    result_ids = sorted(state["intermediate_results"])
    result_lines = "\n".join(f"- `{result_id}`" for result_id in result_ids)
    report = (
        "# Research Workflow Result\n\n"
        f"**Goal:** {state['user_goal']}\n\n"
        "The dependency-aware workflow completed successfully. "
        "Agent-generated report content is introduced in Step 3.\n\n"
        "## Blackboard result references\n\n"
        f"{result_lines}\n"
    )
    return {
        "final_report": report,
        "current_task": None,
        "next_agent": None,
        "workflow_status": WorkflowStatus.COMPLETED,
        "checkpoints": [CheckpointReference(stage="workflow_completed")],
    }


def workflow_failed_node(state: AgentState) -> dict[str, Any]:
    """Close a failed run while retaining structured diagnostics in state."""

    return {
        "current_task": None,
        "next_agent": None,
        "workflow_status": WorkflowStatus.FAILED,
        "checkpoints": [CheckpointReference(stage="workflow_failed")],
    }


def human_intervention_node(state: AgentState) -> dict[str, Any]:
    """Persist a stable waiting state until feedback is applied to the checkpoint."""

    return {
        "current_task": None,
        "next_agent": None,
        "workflow_status": WorkflowStatus.WAITING_HUMAN,
        "checkpoints": [CheckpointReference(stage="human_intervention")],
    }


def build_workflow(
    executors: Mapping[AgentName | str, TaskExecutor] | None = None,
    *,
    checkpointer: Any | None = None,
    requirement_parser: Callable[[AgentState], dict[str, Any]] | None = None,
    planner: Callable[[AgentState], dict[str, Any]] | None = None,
    supervisor: Callable[[AgentState], dict[str, Any]] | None = None,
    finalizer: Callable[[AgentState], dict[str, Any]] | None = None,
    recovery_policy: RecoveryPolicy | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> Any:
    """Compile the research graph with optional per-agent executor injection."""

    configured_executors = {
        AgentName(key) if isinstance(key, str) else key: value
        for key, value in (executors or {}).items()
    }
    builder = StateGraph(AgentState)
    builder.add_node("requirement_parser", requirement_parser or requirement_parser_node)
    builder.add_node("planner", planner or planner_node)
    builder.add_node("supervisor", supervisor or supervisor_node)
    builder.add_node(
        "worker",
        make_worker_node(configured_executors, recovery_policy),
    )
    builder.add_edge("worker", "supervisor")
    builder.add_node("finalize", finalizer or finalize_node)
    builder.add_node("workflow_failed", workflow_failed_node)
    builder.add_node("human_intervention", human_intervention_node)

    builder.add_edge(START, "requirement_parser")
    builder.add_edge("requirement_parser", "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges("supervisor", route_supervisor)
    builder.add_edge("finalize", END)
    builder.add_edge("workflow_failed", END)
    builder.add_edge("human_intervention", END)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=_normalize_interrupt_nodes(interrupt_before),
        interrupt_after=_normalize_interrupt_nodes(interrupt_after),
    )


def run_research_workflow(
    user_goal: str,
    tasks: list[ResearchTask] | None = None,
    *,
    executors: Mapping[AgentName | str, TaskExecutor] | None = None,
    config: dict[str, Any] | None = None,
) -> AgentState:
    """Run the offline workflow synchronously and return its complete final state."""

    graph = build_workflow(executors)
    return graph.invoke(create_initial_state(user_goal, tasks), config=config)
