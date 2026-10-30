"""Supervisor Agent: deterministic task-state and dependency decisions."""

from pydantic import BaseModel, ConfigDict

from graph.state import TaskDAG, TaskStatus, WorkflowStatus


class SupervisorDecision(BaseModel):
    """Explicit routing decision consumed by the LangGraph adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_status: WorkflowStatus
    runnable_task_ids: tuple[str, ...] = ()
    blocked_task_ids: tuple[str, ...] = ()
    next_agent: str
    reason: str


class SupervisorAgent:
    """Schedule a validated DAG without delegating state transitions to an LLM."""

    def decide(self, dag: TaskDAG) -> SupervisorDecision:
        """Return the next dispatch/finalization/failure action for current task states."""

        blocked = dag.blocked_updates()
        if blocked:
            return SupervisorDecision(
                workflow_status=WorkflowStatus.FAILED,
                blocked_task_ids=tuple(task.task_id for task in blocked),
                next_agent="workflow_failed",
                reason="one or more tasks are transitively blocked by a failed dependency",
            )
        if dag.all_completed():
            return SupervisorDecision(
                workflow_status=WorkflowStatus.COMPLETED,
                next_agent="finalize",
                reason="all planned tasks completed",
            )
        ready = dag.ready_tasks()
        if ready:
            return SupervisorDecision(
                workflow_status=WorkflowStatus.RUNNING,
                runnable_task_ids=tuple(task.task_id for task in ready),
                next_agent="parallel_dispatch",
                reason="dependency constraints are satisfied",
            )
        if any(task.status == TaskStatus.RUNNING for task in dag.tasks):
            return SupervisorDecision(
                workflow_status=WorkflowStatus.RUNNING,
                next_agent="supervisor",
                reason="waiting for running tasks",
            )
        return SupervisorDecision(
            workflow_status=WorkflowStatus.FAILED,
            next_agent="workflow_failed",
            reason="DAG stalled with no runnable tasks",
        )
