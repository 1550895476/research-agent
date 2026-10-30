"""HTTP entrypoint for the Research Agent service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, SecretStr, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from agents.model import ModelConfigurationError
from agents.runtime import AgentRuntime, run_agent_workflow
from core.config import Settings, get_settings
from evidence.store import EvidenceRecord, ToolCallRecord
from graph.state import (
    AgentName,
    HumanFeedback,
    HumanRequest,
    ResearchTask,
    WorkflowError,
    WorkflowStatus,
)
from memory.checkpoint import CheckpointManager

APP_VERSION = "0.1.0"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class HealthResponse(BaseModel):
    """Public, non-sensitive service health information."""

    status: str
    service: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    """Configuration readiness without exposing secret values."""

    status: str
    openai_configured: bool
    search_configured: bool
    langsmith_enabled: bool


class RuntimeConfigResponse(BaseModel):
    """Non-sensitive runtime configuration shown in the local web console."""

    openai_base_url: str
    small_model: str | None
    large_model: str | None
    available_models: list[str]
    structured_output_method: str
    openai_configured: bool
    tavily_configured: bool
    tools_enabled: bool
    require_external_evidence: bool
    research_quality_gate_enabled: bool
    research_initial_searches: int
    research_followup_batch_size: int
    research_max_searches_per_task: int
    research_completion_score: int
    research_min_unique_sources: int
    research_min_independent_domains: int
    dynamic_model_routing: bool
    langsmith_tracing: bool
    langsmith_configured: bool
    langsmith_project: str


class RuntimeConfigUpdate(BaseModel):
    """Mutable, process-local API settings; secret values are never returned."""

    openai_api_key: SecretStr | None = None
    clear_openai_api_key: bool = False
    openai_base_url: str | None = Field(default=None, min_length=1, max_length=2000)
    small_model: str | None = Field(default=None, min_length=1, max_length=300)
    large_model: str | None = Field(default=None, min_length=1, max_length=300)
    structured_output_method: Literal["json_schema", "function_calling", "json_mode"] | None = None
    tavily_api_key: SecretStr | None = None
    clear_tavily_api_key: bool = False
    tools_enabled: bool | None = None
    require_external_evidence: bool | None = None
    research_quality_gate_enabled: bool | None = None
    research_initial_searches: int | None = Field(default=None, ge=1, le=10)
    research_followup_batch_size: int | None = Field(default=None, ge=1, le=5)
    research_max_searches_per_task: int | None = Field(default=None, ge=1, le=100)
    research_completion_score: int | None = Field(default=None, ge=0, le=100)
    research_min_unique_sources: int | None = Field(default=None, ge=1, le=100)
    research_min_independent_domains: int | None = Field(default=None, ge=1, le=50)
    dynamic_model_routing: bool | None = None
    langsmith_tracing: bool | None = None
    langsmith_api_key: SecretStr | None = None
    clear_langsmith_api_key: bool = False
    langsmith_project: str | None = Field(default=None, min_length=1, max_length=300)


class ResearchRequest(BaseModel):
    """Input accepted by the synchronous MVP research endpoint."""

    user_goal: str | None = Field(default=None, min_length=1, max_length=10000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=200)
    thread_id: str | None = Field(default=None, min_length=1, max_length=200)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=200)
    parent_thread_id: str | None = Field(default=None, min_length=1, max_length=200)
    model: str | None = Field(default=None, min_length=1, max_length=300)
    resume: bool = False
    human_feedback: HumanFeedback | None = None
    pause_after: tuple[str, ...] = ()
    continuous: bool = False

    @model_validator(mode="after")
    def validate_start_or_resume(self) -> ResearchRequest:
        if self.resume and self.thread_id is None:
            raise ValueError("thread_id is required when resume=true")
        if not self.resume and self.user_goal is None:
            raise ValueError("user_goal is required when starting a workflow")
        if self.human_feedback is not None and not self.resume:
            raise ValueError("human_feedback requires resume=true")
        if self.resume and (self.parent_thread_id is not None or self.model is not None):
            raise ValueError("parent_thread_id and model are only valid for a new turn")
        if self.parent_thread_id is not None and self.parent_thread_id == self.thread_id:
            raise ValueError("parent_thread_id must differ from the new thread_id")
        allowed_pause_nodes = {
            "requirement_parser",
            "planner",
            "supervisor",
            "worker",
            *(agent.value for agent in AgentName),
        }
        unknown_nodes = set(self.pause_after) - allowed_pause_nodes
        if unknown_nodes:
            raise ValueError(f"unknown pause_after nodes: {sorted(unknown_nodes)}")
        return self


class ResearchResponse(BaseModel):
    """Final workflow state exposed without internal Blackboard payloads."""

    status: WorkflowStatus
    thread_id: str
    conversation_id: str
    parent_thread_id: str | None = None
    user_id: str
    user_goal: str
    selected_model: str | None = None
    tasks: list[ResearchTask]
    final_report: str | None
    errors: list[WorkflowError]
    human_requests: list[HumanRequest] = Field(default_factory=list)
    paused: bool = False
    can_resume: bool = False
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    evidences: list[EvidenceView] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStepView] = Field(default_factory=list)


class ReasoningStepView(BaseModel):
    """Safe execution trace: stages and metadata, never hidden chain-of-thought text."""

    step_id: str
    kind: str
    name: str
    status: str
    created_at: datetime
    task_id: str | None = None
    agent: str | None = None
    detail: str | None = None
    duration_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model_tier: str | None = None
    schema_name: str | None = None


class ToolCallView(BaseModel):
    """Safe tool audit information exposed to the local research console."""

    call_id: str
    task_id: str
    tool_name: str
    purpose: str
    arguments: dict[str, Any]
    status: str
    evidence_count: int
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None


class EvidenceView(BaseModel):
    """Source provenance returned without placing source text in LangGraph state."""

    evidence_id: str
    task_id: str
    tool_name: str
    source_type: str
    source_url: str
    title: str
    snippet: str
    confidence: float
    created_time: datetime


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application instance, optionally with injected test settings."""

    runtime_settings = settings or get_settings()
    active_settings = runtime_settings
    checkpoint_manager = CheckpointManager(runtime_settings)
    runtime_lock = Lock()
    shared_runtime: AgentRuntime | None = None
    run_runtimes: dict[str, AgentRuntime] = {}
    active_threads: set[str] = set()

    def get_runtime(model_name: str | None = None) -> AgentRuntime:
        nonlocal shared_runtime
        if model_name:
            model_settings = active_settings.model_copy(
                update={
                    "small_model": model_name,
                    "large_model": model_name,
                    # Keep the router enabled so model calls still emit safe tier/schema
                    # trace metadata; both tiers resolve to the explicitly selected model.
                    "dynamic_model_routing": True,
                }
            )
            return AgentRuntime.from_settings(model_settings)
        if shared_runtime is not None:
            return shared_runtime
        with runtime_lock:
            if shared_runtime is None:
                shared_runtime = AgentRuntime.from_settings(active_settings)
        return shared_runtime

    application = FastAPI(
        title="Research Agent API",
        description="Multi-agent research orchestration service.",
        version=APP_VERSION,
    )
    application.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @application.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        """Serve the interactive local research console."""

        return FileResponse(FRONTEND_DIR / "index.html")

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="research-agent",
            version=APP_VERSION,
            environment=runtime_settings.app_env,
        )

    @application.get("/ready", response_model=ReadinessResponse, tags=["system"])
    async def readiness() -> ReadinessResponse:
        return ReadinessResponse(
            status="ready",
            openai_configured=bool(
                active_settings.openai_api_key
                and active_settings.small_model
                and active_settings.large_model
            ),
            search_configured=bool(active_settings.tavily_api_key),
            langsmith_enabled=bool(
                active_settings.langsmith_tracing and active_settings.langsmith_api_key
            ),
        )

    @application.get("/config", response_model=RuntimeConfigResponse, tags=["system"])
    async def runtime_config() -> RuntimeConfigResponse:
        return _runtime_config_response(active_settings)

    @application.post("/config", response_model=RuntimeConfigResponse, tags=["system"])
    async def update_runtime_config(
        update: RuntimeConfigUpdate,
    ) -> RuntimeConfigResponse:
        nonlocal active_settings, shared_runtime
        changes = {
            key: value
            for key, value in update.model_dump(
                exclude={
                    "clear_openai_api_key",
                    "clear_tavily_api_key",
                    "clear_langsmith_api_key",
                }
            ).items()
            if value is not None
        }
        if update.clear_openai_api_key:
            changes["openai_api_key"] = None
        if update.clear_tavily_api_key:
            changes["tavily_api_key"] = None
        if update.clear_langsmith_api_key:
            changes["langsmith_api_key"] = None
        if update.research_max_searches_per_task is not None:
            changes["search_max_calls_per_task"] = max(
                active_settings.search_max_calls_per_task,
                update.research_max_searches_per_task,
            )
        try:
            candidate_settings = Settings.model_validate(
                {**active_settings.model_dump(), **changes}
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with runtime_lock:
            active_settings = candidate_settings
            shared_runtime = None
        return _runtime_config_response(active_settings)

    @application.post("/research", response_model=ResearchResponse, tags=["research"])
    async def research(request: ResearchRequest) -> ResearchResponse:
        thread_id = request.thread_id or str(uuid4())
        try:
            checkpoint_state: dict[str, Any] | None = None
            if not request.resume:
                if not (
                    active_settings.openai_api_key
                    and active_settings.small_model
                    and active_settings.large_model
                ):
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "openai_api_key, small_model and large_model must be configured"
                        ),
                    )
                if active_settings.require_external_evidence and not (
                    active_settings.tools_enabled and active_settings.tavily_api_key
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "实时调研要求启用外部工具并配置 Tavily API Key；"
                            "系统不会用模型内置知识伪装成检索结果"
                        ),
                    )
            if request.resume:
                checkpoint_state = await run_in_threadpool(
                    checkpoint_manager.load_state,
                    thread_id,
                )
                if checkpoint_state is None:
                    raise HTTPException(status_code=404, detail="checkpoint thread not found")

            parent_state: dict[str, Any] | None = None
            if request.parent_thread_id is not None:
                parent_state = await run_in_threadpool(
                    checkpoint_manager.load_state,
                    request.parent_thread_id,
                )
                if parent_state is None:
                    raise HTTPException(status_code=404, detail="parent research turn not found")
                if not parent_state.get("final_report"):
                    raise HTTPException(
                        status_code=409,
                        detail="parent research turn must be completed before a follow-up",
                    )
                parent_conversation_id = parent_state.get("conversation_id")
                if (
                    request.conversation_id is not None
                    and parent_conversation_id is not None
                    and request.conversation_id != parent_conversation_id
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="conversation_id does not match the parent research turn",
                    )

            selected_model = (
                checkpoint_state.get("selected_model") if checkpoint_state else request.model
            )
            runtime = run_runtimes.get(thread_id) or get_runtime(selected_model)
            run_runtimes[thread_id] = runtime
            with runtime_lock:
                if thread_id in active_threads:
                    raise HTTPException(
                        status_code=409,
                        detail="research thread is already executing a segment",
                    )
                active_threads.add(thread_id)

            def invoke() -> dict:
                conversation_id = (
                    request.conversation_id
                    or (parent_state or {}).get("conversation_id")
                    or thread_id
                )
                user_message = request.user_goal
                effective_goal = (
                    _follow_up_goal(
                        parent_state,
                        user_message,
                        active_settings.max_agent_context_chars,
                    )
                    if parent_state is not None
                    else user_message
                )
                with checkpoint_manager.open() as checkpointer:
                    return run_agent_workflow(
                        effective_goal,
                        runtime=runtime,
                        checkpointer=checkpointer,
                        thread_id=thread_id,
                        user_id=request.user_id,
                        conversation_id=conversation_id,
                        user_message=user_message,
                        parent_run_id=request.parent_thread_id,
                        selected_model=selected_model,
                        resume=request.resume,
                        human_feedback=request.human_feedback,
                        interrupt_after=(
                            list(request.pause_after)
                            if request.pause_after
                            else (None if request.continuous else ["supervisor"])
                        ),
                    )

            try:
                state = await run_in_threadpool(invoke)
            finally:
                with runtime_lock:
                    active_threads.discard(thread_id)
        except HTTPException:
            raise
        except ModelConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - API boundary converts internal failures
            raise HTTPException(status_code=502, detail=f"research workflow failed: {exc}") from exc
        return _research_response(state, runtime, execution_active=False)

    @application.get(
        "/research/{thread_id}",
        response_model=ResearchResponse,
        tags=["research"],
    )
    async def research_status(thread_id: str) -> ResearchResponse:
        state = await run_in_threadpool(checkpoint_manager.load_state, thread_id)
        if state is None:
            raise HTTPException(status_code=404, detail="checkpoint thread not found")
        with runtime_lock:
            execution_active = thread_id in active_threads
        return _research_response(
            state,
            run_runtimes.get(thread_id),
            execution_active=execution_active,
        )

    return application


def _runtime_config_response(settings: Settings) -> RuntimeConfigResponse:
    available_models = list(
        dict.fromkeys(
            model for model in (settings.small_model, settings.large_model) if model
        )
    )
    return RuntimeConfigResponse(
        openai_base_url=settings.openai_base_url,
        small_model=settings.small_model,
        large_model=settings.large_model,
        available_models=available_models,
        structured_output_method=settings.structured_output_method,
        openai_configured=bool(
            settings.openai_api_key and settings.small_model and settings.large_model
        ),
        tavily_configured=bool(settings.tavily_api_key),
        tools_enabled=settings.tools_enabled,
        require_external_evidence=settings.require_external_evidence,
        research_quality_gate_enabled=settings.research_quality_gate_enabled,
        research_initial_searches=settings.research_initial_searches,
        research_followup_batch_size=settings.research_followup_batch_size,
        research_max_searches_per_task=settings.research_max_searches_per_task,
        research_completion_score=settings.research_completion_score,
        research_min_unique_sources=settings.research_min_unique_sources,
        research_min_independent_domains=settings.research_min_independent_domains,
        dynamic_model_routing=settings.dynamic_model_routing,
        langsmith_tracing=settings.langsmith_tracing,
        langsmith_configured=bool(settings.langsmith_api_key),
        langsmith_project=settings.langsmith_project,
    )


def _research_response(
    state: dict[str, Any],
    runtime: AgentRuntime | None,
    *,
    execution_active: bool,
) -> ResearchResponse:
    run_id = state["run_id"]
    tool_records = runtime.evidence_store.list_tool_calls(run_id) if runtime else ()
    evidence_records = runtime.evidence_store.list_evidences(run_id) if runtime else ()
    persisted_status = state["workflow_status"]
    can_resume = bool(
        state.get("final_report") is None
        and persisted_status
        not in {WorkflowStatus.FAILED, WorkflowStatus.WAITING_HUMAN}
    )
    paused = can_resume and not execution_active
    return ResearchResponse(
        status=WorkflowStatus.PAUSED if paused else persisted_status,
        thread_id=run_id,
        conversation_id=state.get("conversation_id", run_id),
        parent_thread_id=state.get("parent_run_id"),
        user_id=state["user_id"],
        user_goal=state.get("user_message", state["user_goal"]),
        selected_model=state.get("selected_model"),
        tasks=state["tasks"],
        final_report=state.get("final_report"),
        errors=state.get("errors", []),
        human_requests=state.get("human_requests", []),
        paused=paused,
        can_resume=can_resume,
        tool_calls=[_tool_call_view(record) for record in tool_records],
        evidences=[_evidence_view(record) for record in evidence_records],
        reasoning_steps=_reasoning_step_views(state, runtime),
    )


def _follow_up_goal(
    parent_state: dict[str, Any] | None,
    user_message: str | None,
    max_context_chars: int,
) -> str | None:
    """Build bounded context for a new workflow turn linked to a completed turn."""

    if parent_state is None or user_message is None:
        return user_message
    previous_message = parent_state.get("user_message", parent_state.get("user_goal", ""))
    previous_report = str(parent_state.get("final_report") or "")
    fixed = (
        "这是同一研究会话的后续问题。请结合上一轮结论回答当前问题；"
        "若当前要求与上一轮冲突，以当前要求为准。\n\n"
        f"上一轮用户问题：\n{previous_message}\n\n"
        "上一轮研究结论：\n"
    )
    suffix = f"\n\n当前用户问题：\n{user_message}"
    report_budget = max(1000, max_context_chars - len(fixed) - len(suffix))
    bounded_report = previous_report[:report_budget]
    return f"{fixed}{bounded_report}{suffix}"


def _reasoning_step_views(
    state: dict[str, Any],
    runtime: AgentRuntime | None,
) -> list[ReasoningStepView]:
    """Return high-level observable decisions without prompts or hidden reasoning."""

    steps: list[ReasoningStepView] = []
    for index, checkpoint in enumerate(state.get("checkpoints", [])):
        task_ids = tuple(checkpoint.task_ids)
        steps.append(
            ReasoningStepView(
                step_id=f"checkpoint:{index}:{checkpoint.stage}",
                kind="checkpoint",
                name=checkpoint.stage,
                status="completed",
                created_at=checkpoint.created_at,
                detail=(f"tasks: {', '.join(task_ids)}" if task_ids else None),
            )
        )
    for event in state.get("status_events", []):
        steps.append(
            ReasoningStepView(
                step_id=event.event_id,
                kind="agent",
                name=f"worker.{event.outcome}",
                status=event.outcome,
                created_at=event.created_at,
                task_id=event.task_id,
                agent=event.agent.value,
                detail=event.reason,
            )
        )
    if runtime is not None:
        for trace in runtime.observability.snapshot(run_id=state["run_id"]):
            safe_metadata = {
                key: value
                for key, value in trace.metadata.items()
                if key in {"tier", "schema"}
            }
            steps.append(
                ReasoningStepView(
                    step_id=trace.trace_id,
                    kind=trace.kind,
                    name=trace.name,
                    status=trace.status,
                    created_at=trace.started_at,
                    task_id=trace.task_id,
                    agent=trace.agent,
                    detail=(
                        "调用失败，详情见工作流异常记录"
                        if trace.status == "failed"
                        else None
                    ),
                    duration_ms=trace.duration_ms,
                    input_tokens=trace.input_tokens,
                    output_tokens=trace.output_tokens,
                    model_tier=safe_metadata.get("tier"),
                    schema_name=safe_metadata.get("schema"),
                )
            )
    return sorted(steps, key=lambda step: step.created_at)


def _tool_call_view(record: ToolCallRecord) -> ToolCallView:
    return ToolCallView(
        call_id=record.call_id,
        task_id=record.task_id,
        tool_name=record.tool_name,
        purpose=record.purpose,
        arguments=record.arguments,
        status=record.status.value,
        evidence_count=len(record.evidence_ids),
        error=record.error,
        started_at=record.started_at,
        finished_at=record.finished_at,
        duration_ms=record.duration_ms,
    )


def _evidence_view(record: EvidenceRecord) -> EvidenceView:
    return EvidenceView(
        evidence_id=record.evidence_id,
        task_id=record.task_id,
        tool_name=record.tool_name,
        source_type=record.source_type.value,
        source_url=record.source_url,
        title=record.title,
        snippet=record.content[:1000],
        confidence=record.confidence,
        created_time=record.created_time,
    )


app = create_app()
