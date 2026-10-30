"""Unit tests for task DAG validation and dependency scheduling."""

import pytest

from graph.state import AgentName, ResearchTask, TaskDAG, TaskStatus, create_initial_state


def make_task(
    task_id: str,
    *,
    dependencies: tuple[str, ...] = (),
    status: TaskStatus = TaskStatus.PENDING,
) -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        title=task_id,
        description=f"execute {task_id}",
        dependencies=dependencies,
        status=status,
        assigned_agent=AgentName.RESEARCHER,
    )


def test_task_dag_rejects_cycles() -> None:
    tasks = [
        make_task("a", dependencies=("b",)),
        make_task("b", dependencies=("a",)),
    ]

    with pytest.raises(ValueError, match="cycle"):
        TaskDAG(tasks)


def test_ready_tasks_require_completed_dependencies() -> None:
    tasks = [
        make_task("a", status=TaskStatus.COMPLETED),
        make_task("b", dependencies=("a",)),
        make_task("c", dependencies=("b",)),
    ]

    assert [task.task_id for task in TaskDAG(tasks).ready_tasks()] == ["b"]


def test_initial_state_normalizes_goal_and_validates_plan() -> None:
    state = create_initial_state("  research   agent platforms  ", [make_task("a")])

    assert state["user_goal"] == "research agent platforms"
    assert state["tasks"][0].task_id == "a"

