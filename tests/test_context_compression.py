"""Watermark, ladder, and checkpoint tests for prompt context compression."""

from core.config import Settings
from core.context_compression import (
    ContextCompressor,
    ContextItem,
    ContextSnapshot,
)
from graph.state import AgentName, ResearchTask, create_initial_state
from graph.workflow import TaskExecutionResult, build_workflow


def test_context_below_soft_watermark_is_not_rewritten() -> None:
    compressor = ContextCompressor(max_chars=1000)

    result = compressor.compress(
        [
            ContextItem(
                item_id="active",
                kind="active",
                content="Keep this exact current goal.",
                required=True,
            )
        ],
        context_id="run:task:small",
    )

    assert result.record is None
    assert "Keep this exact current goal." in result.text
    assert compressor.snapshot() == ()


def test_level_one_replaces_regenerable_body_with_reproducible_reference() -> None:
    compressor = ContextCompressor(max_chars=1000)
    result = compressor.compress(
        [
            ContextItem(
                item_id="readme",
                kind="regenerable",
                content="README body " * 100,
                summary="Repository instructions.",
                reference="/repo/README.md",
                version="commit:4f3a8c2",
                replay_hint="read /repo/README.md at commit 4f3a8c2",
            )
        ],
        context_id="run:task:readme",
    )

    assert result.record is not None
    assert result.record.applied_level == 1
    assert result.record.trigger == "hard"
    assert result.record.checkpoint_required is True
    assert "/repo/README.md" in result.text
    assert "commit:4f3a8c2" in result.text
    assert "README body README body" not in result.text


def test_level_two_tool_truncation_keeps_error_location_and_replay_parameters() -> None:
    compressor = ContextCompressor(
        max_chars=1000,
        tool_result_max_chars=300,
    )
    content = "\n".join(
        ["build output"] * 80
        + ["ERROR authentication token expired"]
        + ["trailing output"] * 80
    )
    result = compressor.compress(
        [
            ContextItem(
                item_id="build-log",
                kind="tool_output",
                content=content,
                reference="/tmp/test-run-123.log",
                replay_hint="pytest tests/test_auth.py",
            )
        ],
        context_id="run:task:build",
    )

    assert result.record is not None
    assert result.record.applied_level == 2
    assert "authentication token expired" in result.text
    assert "/tmp/test-run-123.log" in result.text
    assert "pytest tests/test_auth.py" in result.text


def test_level_five_snapshot_preserves_required_resume_state() -> None:
    compressor = ContextCompressor(max_chars=1000)
    snapshot = ContextSnapshot(
        goal="修复登录超时问题",
        constraints=("不能修改公共 API", "不能升级依赖"),
        decisions=("服务端刷新 Token；避免多标签页竞争。",),
        pending_items=("补充并发刷新测试",),
        exact_references=("src/auth/token.ts:refreshAccessToken", "commit 4f3a8c2"),
        validation_results=("AUTH_4017 已复现",),
        next_action="运行认证回归测试",
    )
    result = compressor.compress(
        [
            ContextItem(
                item_id="active-analysis",
                kind="active",
                content="uncompressible active reasoning " * 100,
                required=True,
            )
        ],
        context_id="run:task:hard",
        snapshot=snapshot,
    )

    assert result.record is not None
    assert result.record.applied_level == 5
    assert result.record.compressed_chars <= result.record.soft_limit_chars
    assert "修复登录超时问题" in result.text
    assert "不能修改公共 API" in result.text
    assert "服务端刷新 Token" in result.text
    assert "补充并发刷新测试" in result.text
    assert "src/auth/token.ts:refreshAccessToken" in result.text
    assert "AUTH_4017" in result.text
    assert "运行认证回归测试" in result.text


def test_settings_reject_invalid_watermark_order() -> None:
    try:
        Settings(context_soft_watermark=0.9, context_hard_watermark=0.8)
    except ValueError as exc:
        assert "CONTEXT_SOFT_WATERMARK" in str(exc)
    else:
        raise AssertionError("invalid watermark order was accepted")


def test_hard_watermark_record_is_checkpointed_with_task_result() -> None:
    compressor = ContextCompressor(max_chars=1000)
    compressed = compressor.compress(
        [
            ContextItem(
                item_id="large-active-context",
                kind="active",
                content="large current context " * 100,
                required=True,
            )
        ],
        context_id="run-hard:research:prompt",
        snapshot=ContextSnapshot(
            goal="Finish research",
            next_action="Resume from the saved snapshot",
        ),
    )
    assert compressed.record is not None
    task = ResearchTask(
        task_id="research",
        title="Research",
        description="Research with hard-watermark context",
        assigned_agent=AgentName.RESEARCHER,
    )

    def executor(_: object) -> TaskExecutionResult:
        return TaskExecutionResult(
            result_id="result:research",
            payload_ref="blackboard://results/result:research",
            context_compressions=(compressed.record,),
        )

    state = build_workflow({AgentName.RESEARCHER: executor}).invoke(
        create_initial_state("Research context compression", [task])
    )

    assert state["context_compressions"] == [compressed.record]
    assert any(
        checkpoint.stage == "context_hard_compressed"
        for checkpoint in state["checkpoints"]
    )


def test_hard_watermark_record_survives_task_failure() -> None:
    compressor = ContextCompressor(max_chars=1000)
    compressed = compressor.compress(
        [
            ContextItem(
                item_id="large-failing-context",
                kind="active",
                content="failing context " * 150,
                required=True,
            )
        ],
        context_id="run-failed:research:prompt",
    )
    assert compressed.record is not None
    task = ResearchTask(
        task_id="research",
        title="Research",
        description="Fail after hard-watermark compression",
        assigned_agent=AgentName.RESEARCHER,
    )

    def executor(_: object) -> TaskExecutionResult:
        error = RuntimeError("model failed after compression")
        error.context_compressions = (compressed.record,)  # type: ignore[attr-defined]
        raise error

    state = build_workflow({AgentName.RESEARCHER: executor}).invoke(
        create_initial_state("Research context failure", [task])
    )

    assert state["context_compressions"] == [compressed.record]
    assert any(
        checkpoint.stage == "context_hard_compressed"
        for checkpoint in state["checkpoints"]
    )
