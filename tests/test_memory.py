"""Tests for checkpoint resume, memory lifecycle, and episodic traces."""

from datetime import UTC, datetime, timedelta

from agents.runtime import AgentRuntime, run_agent_workflow
from core.config import Settings
from graph.state import WorkerStatusEvent
from memory.checkpoint import CheckpointManager
from memory.episodic import EpisodeType, EpisodeWrite, InMemoryEpisodicMemory
from memory.long_term import (
    InMemoryLongTermMemory,
    MemoryCompressor,
    MemoryKind,
    MemoryWrite,
)
from tests.test_agents import SchemaFakeModel


def test_checkpointed_agent_workflow_resumes_after_planner_interrupt() -> None:
    model = SchemaFakeModel()
    runtime = AgentRuntime(small_model=model, large_model=model)
    manager = CheckpointManager(Settings(checkpoint_backend="memory"))

    with manager.open() as checkpointer:
        first_state = run_agent_workflow(
            "调研大模型 Agent 平台",
            runtime=runtime,
            checkpointer=checkpointer,
            thread_id="thread-memory-1",
            user_id="user-1",
            interrupt_after=["planner"],
        )
    first_call_count = sum(model.calls.values())

    assert first_state["final_report"] is None
    assert first_call_count == 2
    assert manager.has_checkpoint("thread-memory-1")

    with manager.open() as checkpointer:
        resumed_state = run_agent_workflow(
            None,
            runtime=runtime,
            checkpointer=checkpointer,
            thread_id="thread-memory-1",
            user_id="user-1",
            resume=True,
        )

    assert resumed_state["final_report"] is not None
    assert resumed_state["requirements"] == first_state["requirements"]
    assert resumed_state["run_id"] == "thread-memory-1"
    assert sum(model.calls.values()) == 7


def test_checkpointed_workflow_runs_in_resumable_supervisor_segments() -> None:
    model = SchemaFakeModel()
    runtime = AgentRuntime(small_model=model, large_model=model)
    manager = CheckpointManager(Settings(checkpoint_backend="memory"))
    thread_id = "thread-segmented-1"

    with manager.open() as checkpointer:
        state = run_agent_workflow(
            "调研大模型 Agent 平台",
            runtime=runtime,
            checkpointer=checkpointer,
            thread_id=thread_id,
            user_id="user-1",
            interrupt_after=["supervisor"],
        )

    assert state["final_report"] is None
    assert sum(model.calls.values()) == 2
    completed_counts = [sum(task.status == "completed" for task in state["tasks"])]

    for _ in range(10):
        with manager.open() as checkpointer:
            state = run_agent_workflow(
                None,
                runtime=runtime,
                checkpointer=checkpointer,
                thread_id=thread_id,
                user_id="user-1",
                resume=True,
                interrupt_after=["supervisor"],
            )
        completed_counts.append(
            sum(task.status == "completed" for task in state["tasks"])
        )
        if state["final_report"] is not None:
            break

    assert state["final_report"] is not None
    assert completed_counts == sorted(completed_counts)
    assert sum(model.calls.values()) == 7
    assert state["status_events"]
    assert all(isinstance(event, WorkerStatusEvent) for event in state["status_events"])


def test_long_term_memory_compresses_scores_updates_and_evicts() -> None:
    store = InMemoryLongTermMemory(
        compressor=MemoryCompressor(max_chars=500),
        min_value_score=0.25,
        max_records_per_user=2,
    )
    high = store.upsert(
        MemoryWrite(
            memory_id="high",
            user_id="user-1",
            kind=MemoryKind.RESEARCH_HISTORY,
            content="A" * 1000,
            quality=0.9,
            confidence=0.9,
            reuse_signal=0.8,
        )
    )
    store.upsert(
        MemoryWrite(
            memory_id="expired",
            user_id="user-1",
            kind=MemoryKind.USER_PREFERENCE,
            content="old preference",
            quality=1,
            confidence=1,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    )
    store.upsert(
        MemoryWrite(
            memory_id="low",
            user_id="user-1",
            kind=MemoryKind.HIGH_VALUE_EVIDENCE,
            content="weak evidence",
            quality=0,
            confidence=0,
        )
    )

    found = store.search("user-1", limit=5)
    eviction = store.evict()

    assert high.compressed is True
    assert len(high.content) <= 520
    assert found[0].memory_id == "high"
    assert set(eviction.evicted_ids) == {"expired", "low"}


def test_episodic_memory_filters_failures_and_run_history() -> None:
    store = InMemoryEpisodicMemory()
    store.record(
        EpisodeWrite(
            run_id="run-1",
            user_id="user-1",
            event_type=EpisodeType.PLAN_CREATED,
            stage="planner",
            decision="created",
        )
    )
    store.record(
        EpisodeWrite(
            run_id="run-1",
            user_id="user-1",
            event_type=EpisodeType.TASK_FAILED,
            stage="retriever",
            task_id="search",
            failure_reason="timeout",
        )
    )

    assert len(store.list_events(user_id="user-1", run_id="run-1")) == 2
    assert store.recent_failures("user-1")[0].failure_reason == "timeout"
