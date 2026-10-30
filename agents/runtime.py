"""Adapters that connect real Agent classes to the LangGraph workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
from pydantic import BaseModel

from agents.model import ModelFactory, ModelTier, StructuredModel
from agents.model_router import DynamicModelRouter
from agents.planner import PlannerAgent
from agents.requirement_parser import RequirementParserAgent
from agents.researcher import ResearcherAgent, ResearchOutput, ResearchSource
from agents.supervisor import SupervisorAgent
from agents.tool_selector import LLMToolSelector, PlannedToolCall, ToolCallPlan, ToolSelector
from agents.verifier import TaskQualityAssessment, VerifierAgent
from agents.writer import ReportDraft, ReportWriterAgent
from core.config import Settings, get_settings
from core.observability import AgentObservability, create_observability, observation_scope
from core.recovery import EmptyRetrievalError, RecoveryPolicy, ToolExecutionError
from core.saga import SagaCoordinator, SagaStep, create_saga_coordinator
from evidence.store import EvidenceRecord, EvidenceStore, InMemoryEvidenceStore
from graph.blackboard import Blackboard, InMemoryBlackboard, create_blackboard
from graph.state import (
    AgentName,
    AgentState,
    CheckpointReference,
    HumanFeedback,
    HumanRequest,
    ResearchTask,
    TaskDAG,
    TaskStatus,
    WorkflowError,
    WorkflowStatus,
    create_initial_state,
)
from graph.workflow import (
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutor,
    build_workflow,
)
from guardrail.middleware import GuardedToolRuntime, ToolQuotaExceeded
from memory.checkpoint import checkpoint_config
from memory.episodic import (
    EpisodeType,
    EpisodeWrite,
    EpisodicMemory,
    InMemoryEpisodicMemory,
    create_episodic_memory,
)
from memory.long_term import (
    InMemoryLongTermMemory,
    LongTermMemory,
    MemoryKind,
    MemoryWrite,
    create_long_term_memory,
)
from rag.pipeline import RAGPipeline, create_rag_pipeline
from rag.retriever import SourceType
from tools.factory import create_tool_runtime
from tools.registry import ToolContext, ToolEvidence, ToolResult, ToolSpec


class AgentRuntime:
    """Own Agent instances, Blackboard storage, and graph-node adapters for one run."""

    def __init__(
        self,
        *,
        small_model: StructuredModel,
        large_model: StructuredModel,
        blackboard: Blackboard | None = None,
        long_term_memory: LongTermMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
        evidence_store: EvidenceStore | None = None,
        rag_pipeline: RAGPipeline | None = None,
        tool_runtime: GuardedToolRuntime | None = None,
        tool_selector: ToolSelector | None = None,
        tool_permissions: frozenset[str] = frozenset(),
        recovery_policy: RecoveryPolicy | None = None,
        saga_coordinator: SagaCoordinator | None = None,
        model_router: StructuredModel | None = None,
        observability: AgentObservability | None = None,
        max_context_chars: int = 16000,
        require_external_evidence: bool = False,
        max_research_tool_rounds: int = 3,
        research_quality_gate_enabled: bool = False,
        research_initial_searches: int = 5,
        research_followup_batch_size: int = 2,
        research_max_searches_per_task: int = 10,
        research_completion_score: int = 80,
        research_min_unique_sources: int = 5,
        research_min_independent_domains: int = 3,
    ) -> None:
        self.blackboard = blackboard or InMemoryBlackboard()
        self.long_term_memory = long_term_memory or InMemoryLongTermMemory()
        self.episodic_memory = episodic_memory or InMemoryEpisodicMemory()
        self.evidence_store = evidence_store or InMemoryEvidenceStore()
        self.rag_pipeline = rag_pipeline
        self.tool_runtime = tool_runtime
        self.tool_selector = tool_selector
        self.tool_permissions = tool_permissions
        self.recovery_policy = recovery_policy or RecoveryPolicy.disabled()
        self.saga = saga_coordinator or SagaCoordinator()
        self.observability = observability or AgentObservability()
        self._max_context_chars = max_context_chars
        self._require_external_evidence = require_external_evidence
        self._max_research_tool_rounds = max(1, max_research_tool_rounds)
        self._research_quality_gate_enabled = research_quality_gate_enabled
        self._research_initial_searches = max(1, research_initial_searches)
        self._research_followup_batch_size = max(1, research_followup_batch_size)
        self._research_max_searches_per_task = max(1, research_max_searches_per_task)
        self._research_completion_score = max(0, min(100, research_completion_score))
        self._research_min_unique_sources = max(1, research_min_unique_sources)
        self._research_min_independent_domains = max(
            1, research_min_independent_domains
        )
        if self._research_max_searches_per_task < self._research_initial_searches:
            raise ValueError(
                "research_max_searches_per_task must be >= research_initial_searches"
            )
        small_agent_model = model_router or small_model
        large_agent_model = model_router or large_model
        self.requirement_parser = RequirementParserAgent(small_agent_model)
        self.planner = PlannerAgent(
            large_agent_model,
            require_retrieval=require_external_evidence,
        )
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent(
            large_agent_model,
            max_context_chars=max_context_chars,
        )
        self.verifier = VerifierAgent(large_agent_model, max_context_chars=max_context_chars)
        self.writer = ReportWriterAgent(large_agent_model, max_context_chars=max_context_chars)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        blackboard: Blackboard | None = None,
        long_term_memory: LongTermMemory | None = None,
        episodic_memory: EpisodicMemory | None = None,
    ) -> AgentRuntime:
        """Construct an API-backed runtime from application settings."""

        runtime_settings = settings or get_settings()
        factory = ModelFactory(runtime_settings)
        small_model = factory.create(ModelTier.SMALL)
        large_model = factory.create(ModelTier.LARGE)
        observability = create_observability(runtime_settings)
        model_router = (
            DynamicModelRouter(
                small_model,
                large_model,
                observability=observability,
                complexity_chars=runtime_settings.model_router_complexity_chars,
            )
            if runtime_settings.dynamic_model_routing
            else None
        )
        rewrite_model = model_router or small_model
        rag_pipeline = (
            create_rag_pipeline(runtime_settings, rewrite_model=rewrite_model)
            if runtime_settings.rag_enabled
            else None
        )
        return cls(
            small_model=small_model,
            large_model=large_model,
            blackboard=blackboard or create_blackboard(runtime_settings),
            long_term_memory=long_term_memory or create_long_term_memory(runtime_settings),
            episodic_memory=episodic_memory or create_episodic_memory(runtime_settings),
            evidence_store=InMemoryEvidenceStore(),
            rag_pipeline=rag_pipeline,
            tool_runtime=(
                create_tool_runtime(runtime_settings, rag_pipeline=rag_pipeline)
                if runtime_settings.tools_enabled
                else None
            ),
            tool_selector=(
                LLMToolSelector(rewrite_model) if runtime_settings.tools_enabled else None
            ),
            tool_permissions=frozenset(
                permission.strip()
                for permission in runtime_settings.tool_permissions.split(",")
                if permission.strip()
            ),
            recovery_policy=RecoveryPolicy(
                enabled=runtime_settings.recovery_enabled,
                max_tool_retries=runtime_settings.recovery_max_tool_retries,
                max_task_replans=runtime_settings.recovery_max_task_replans,
                retry_backoff_seconds=runtime_settings.recovery_backoff_seconds,
            ),
            saga_coordinator=create_saga_coordinator(runtime_settings),
            model_router=model_router,
            observability=observability,
            max_context_chars=runtime_settings.max_agent_context_chars,
            require_external_evidence=runtime_settings.require_external_evidence,
            max_research_tool_rounds=runtime_settings.research_max_tool_rounds,
            research_quality_gate_enabled=runtime_settings.research_quality_gate_enabled,
            research_initial_searches=runtime_settings.research_initial_searches,
            research_followup_batch_size=runtime_settings.research_followup_batch_size,
            research_max_searches_per_task=(
                runtime_settings.research_max_searches_per_task
            ),
            research_completion_score=runtime_settings.research_completion_score,
            research_min_unique_sources=runtime_settings.research_min_unique_sources,
            research_min_independent_domains=(
                runtime_settings.research_min_independent_domains
            ),
        )

    def requirement_parser_node(self, state: AgentState) -> dict[str, Any]:
        """LangGraph adapter for RequirementParserAgent.parse."""

        try:
            requirements = self.requirement_parser.parse(state["user_goal"])
        except Exception as exc:
            self._record_episode(
                state,
                EpisodeType.TASK_FAILED,
                stage="requirement_parser",
                failure_reason=str(exc),
            )
            raise
        self._record_episode(
            state,
            EpisodeType.REQUIREMENT_PARSED,
            stage="requirement_parser",
            decision=requirements.research_type.value,
            payload={"key_questions": list(requirements.key_questions)},
        )
        return {
            "requirements": requirements,
            "user_goal": requirements.objective,
            "workflow_status": WorkflowStatus.PLANNING,
            "checkpoints": [CheckpointReference(stage="requirement_parsed")],
        }

    def planner_node(self, state: AgentState) -> dict[str, Any]:
        """LangGraph adapter for dynamic planning with injected-plan support."""

        if state["tasks"]:
            tasks = state["tasks"]
            TaskDAG(tasks)
        else:
            requirements = state.get("requirements")
            if requirements is None:
                raise ValueError("Planner requires parsed research requirements")
            memories = self.long_term_memory.search(
                state["user_id"],
                kinds=(MemoryKind.USER_PREFERENCE, MemoryKind.RESEARCH_HISTORY),
                limit=5,
                min_value_score=0.3,
            )
            failures = self.episodic_memory.recent_failures(state["user_id"], limit=5)
            memory_context = tuple(
                [f"memory:{memory.kind.value}: {memory.summary}" for memory in memories]
                + [
                    f"failure:{event.stage}: {event.failure_reason}"
                    for event in failures
                    if event.failure_reason
                ]
            )
            try:
                tasks = self.planner.plan(requirements, memory_context=memory_context)
            except Exception as exc:
                self._record_episode(
                    state,
                    EpisodeType.TASK_FAILED,
                    stage="planner",
                    failure_reason=str(exc),
                    payload={"memory_count": len(memory_context)},
                )
                raise
        self._record_episode(
            state,
            EpisodeType.PLAN_CREATED,
            stage="planner",
            decision="dynamic_dag_created",
            payload={
                "task_ids": [task.task_id for task in tasks],
                "dependencies": {
                    task.task_id: list(task.dependencies) for task in tasks
                },
            },
        )
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

    def supervisor_node(self, state: AgentState) -> dict[str, Any]:
        """Translate a deterministic Supervisor decision into graph state updates."""

        if any(task.status == TaskStatus.WAITING_HUMAN for task in state["tasks"]):
            return {
                "workflow_status": WorkflowStatus.WAITING_HUMAN,
                "next_agent": "human_intervention",
                "dispatched_task_ids": [],
            }
        unhandled_replans = [
            request
            for request in state["replan_requests"]
            if request.request_id not in state["handled_replan_ids"]
        ]
        if unhandled_replans:
            requirements = state.get("requirements")
            if requirements is None:
                raise ValueError("local re-planning requires parsed requirements")
            dag = TaskDAG(state["tasks"])
            revised_tasks = []
            handled_ids = set()
            try:
                for request in unhandled_replans:
                    revised = self.planner.replan_task(
                        requirements,
                        dag.get(request.task_id),
                        request.reason,
                    )
                    revised_tasks.append(revised)
                    if self.tool_runtime is not None:
                        self.tool_runtime.guardrail.reset_task_usage(
                            state["run_id"], request.task_id
                        )
                    handled_ids.add(request.request_id)
            except Exception as exc:
                task_id = unhandled_replans[0].task_id
                return {
                    "tasks": [
                        dag.get(task_id).model_copy(
                            update={
                                "status": TaskStatus.WAITING_HUMAN,
                                "error": str(exc),
                            }
                        )
                    ],
                    "workflow_status": WorkflowStatus.WAITING_HUMAN,
                    "human_requests": [
                        HumanRequest(task_id=task_id, reason=f"re-planning failed: {exc}")
                    ],
                    "next_agent": "human_intervention",
                    "dispatched_task_ids": [],
                }
            TaskDAG(
                [
                    next(
                        (item for item in revised_tasks if item.task_id == task.task_id),
                        task,
                    )
                    for task in state["tasks"]
                ]
            )
            return {
                "tasks": revised_tasks,
                "handled_replan_ids": handled_ids,
                "next_agent": "supervisor",
                "dispatched_task_ids": [],
                "checkpoints": [
                    CheckpointReference(
                        stage="tasks_locally_replanned",
                        task_ids=tuple(task.task_id for task in revised_tasks),
                    )
                ],
            }

        dag = TaskDAG(state["tasks"])
        decision = self.supervisor.decide(dag)
        with observation_scope(
            run_id=state["run_id"],
            user_id=state["user_id"],
            agent="supervisor",
        ):
            with self.observability.span(
                "supervisor.decision",
                kind="supervisor",
                metadata={
                    "next_agent": decision.next_agent,
                    "runnable_task_ids": list(decision.runnable_task_ids),
                    "blocked_task_ids": list(decision.blocked_task_ids),
                },
            ):
                pass
        self._record_episode(
            state,
            EpisodeType.SUPERVISOR_DECISION,
            stage="supervisor",
            decision=decision.next_agent,
            payload={
                "reason": decision.reason,
                "runnable_task_ids": list(decision.runnable_task_ids),
                "blocked_task_ids": list(decision.blocked_task_ids),
            },
        )
        if decision.blocked_task_ids:
            return {
                "tasks": dag.blocked_updates(),
                "workflow_status": WorkflowStatus.FAILED,
                "next_agent": decision.next_agent,
                "dispatched_task_ids": [],
                "errors": [
                    WorkflowError(
                        code="DEPENDENCY_FAILED",
                        message=decision.reason,
                        stage="supervisor",
                    )
                ],
            }
        if decision.runnable_task_ids:
            running = [
                dag.get(task_id).model_copy(
                    update={
                        "status": TaskStatus.RUNNING,
                        "attempts": dag.get(task_id).attempts + 1,
                        "error": None,
                    }
                )
                for task_id in decision.runnable_task_ids
            ]
            return {
                "tasks": running,
                "workflow_status": decision.workflow_status,
                "next_agent": decision.next_agent,
                "dispatched_task_ids": list(decision.runnable_task_ids),
                "checkpoints": [
                    CheckpointReference(
                        stage="tasks_dispatched",
                        task_ids=decision.runnable_task_ids,
                    )
                ],
            }
        update: dict[str, Any] = {
            "workflow_status": decision.workflow_status,
            "next_agent": decision.next_agent,
            "dispatched_task_ids": [],
        }
        if decision.workflow_status == WorkflowStatus.FAILED:
            update["errors"] = [
                WorkflowError(
                    code="DAG_STALLED",
                    message=decision.reason,
                    stage="supervisor",
                )
            ]
        return update

    def executors(self) -> Mapping[AgentName, TaskExecutor]:
        """Return Agent executors keyed by the graph's routable node names."""

        executors = {
            AgentName.RESEARCHER: self._research_executor(AgentName.RESEARCHER),
            AgentName.RETRIEVER: self._research_executor(AgentName.RETRIEVER),
            AgentName.DATABASE: self._research_executor(AgentName.DATABASE),
            AgentName.VERIFIER: self._verifier_executor,
            AgentName.WRITER: self._writer_executor,
        }
        return {
            agent: self._observed_executor(agent, executor)
            for agent, executor in executors.items()
        }

    def _observed_executor(
        self,
        agent: AgentName,
        executor: TaskExecutor,
    ) -> TaskExecutor:
        def observed(request: TaskExecutionRequest) -> TaskExecutionResult:
            with observation_scope(
                run_id=request.run_id,
                user_id=request.user_id,
                task_id=request.task.task_id,
                agent=agent.value,
            ):
                with self.observability.span(
                    f"agent.{agent.value}",
                    kind="agent",
                    metadata={"attempt": request.task.attempts},
                ) as span:
                    result = executor(request)
                    span.metadata["result_id"] = result.result_id
                    span.metadata["evidence_count"] = len(result.evidence_ids)
                    return result

        return observed

    def finalize_node(self, state: AgentState) -> dict[str, Any]:
        """Resolve the Writer's Blackboard payload into the final graph output."""

        for task in reversed(state["tasks"]):
            if task.assigned_agent != AgentName.WRITER:
                continue
            for result_id in reversed(task.result_ids):
                reference = state["intermediate_results"].get(result_id)
                if reference is None:
                    continue
                payload = self.blackboard.get_by_uri(reference.payload_ref).payload
                if isinstance(payload, dict):
                    payload = ReportDraft.model_validate(payload)
                if isinstance(payload, ReportDraft):
                    memory_id = f"history:{state['run_id']}"
                    memory = MemoryWrite(
                        memory_id=memory_id,
                        user_id=state["user_id"],
                        kind=MemoryKind.RESEARCH_HISTORY,
                        content=payload.markdown_report,
                        summary=payload.executive_summary,
                        metadata={
                            "run_id": state["run_id"],
                            "title": payload.title,
                        },
                        quality=0.7,
                        confidence=0.6,
                        reuse_signal=0.5,
                    )
                    self.saga.execute(
                        saga_id=f"finalize:{state['run_id']}",
                        run_id=state["run_id"],
                        steps=(
                            SagaStep(
                                name="persist_research_history",
                                action=lambda memory=memory: self.long_term_memory.upsert(memory),
                                compensate=lambda memory_id=memory_id: (
                                    self.long_term_memory.delete(memory_id)
                                ),
                            ),
                        ),
                    )
                    self._record_episode(
                        state,
                        EpisodeType.WORKFLOW_COMPLETED,
                        stage="finalize",
                        decision="report_persisted",
                        payload={"result_id": result_id},
                    )
                    return {
                        "final_report": payload.markdown_report,
                        "current_task": None,
                        "next_agent": None,
                        "workflow_status": WorkflowStatus.COMPLETED,
                        "checkpoints": [CheckpointReference(stage="workflow_completed")],
                    }
        raise ValueError("completed workflow has no ReportDraft in the Blackboard")

    def _research_executor(self, role: AgentName) -> TaskExecutor:
        def execute(request: TaskExecutionRequest) -> TaskExecutionResult:
            try:
                dependency_payloads = self.blackboard.payloads_for_tasks(
                    request.task.dependencies
                )
                try:
                    tool_results = self._execute_tool_calls(role, request)
                except Exception as exc:
                    if role != AgentName.DATABASE:
                        raise
                    tool_results = []
                    dependency_payloads.append(
                        {
                            "database_skipped": True,
                            "reason": str(exc),
                            "instruction": (
                                "The optional internal database was unavailable. Continue without "
                                "claiming that internal data was retrieved."
                            ),
                        }
                    )
                dependency_payloads.extend(tool_results)
                rag_context = None
                if (
                    role in {AgentName.RESEARCHER, AgentName.RETRIEVER}
                    and self.rag_pipeline is not None
                    and self.tool_runtime is None
                ):
                    retrieval_query = (
                        f"{request.task.title}. {request.task.description}. "
                        f"Research goal: {request.user_goal}"
                    )
                    rag_context = self.rag_pipeline.retrieve(retrieval_query)
                    dependency_payloads.append(rag_context)
                payload = self.researcher.research(
                    request.user_goal,
                    request.task,
                    dependency_payloads,
                )
                if role == AgentName.RESEARCHER and self._research_quality_gate_enabled:
                    payload, tool_results = self._run_research_quality_gate(
                        request,
                        payload,
                        dependency_payloads,
                        tool_results,
                    )
                elif role == AgentName.RESEARCHER:
                    searched_queries: set[str] = set()
                    for _ in range(self._max_research_tool_rounds - 1):
                        query = next(
                            (
                                candidate.strip()
                                for candidate in payload.recommended_queries
                                if candidate.strip()
                                and candidate.strip().casefold() not in searched_queries
                            ),
                            None,
                        )
                        if query is None:
                            break
                        searched_queries.add(query.casefold())
                        try:
                            followup_results = self._execute_followup_search(
                                request,
                                query,
                            )
                        except ToolQuotaExceeded:
                            break
                        if not followup_results or not any(
                            result.evidences for result in followup_results
                        ):
                            break
                        tool_results.extend(followup_results)
                        dependency_payloads.extend(followup_results)
                        payload = self.researcher.research(
                            request.user_goal,
                            request.task,
                            [payload, *dependency_payloads],
                        )
                if rag_context is not None:
                    payload = payload.model_copy(
                        update={
                            "retrieved_context": rag_context.context,
                            "sources": tuple(
                                ResearchSource(
                                    evidence_id=citation.evidence_id,
                                    chunk_id=citation.chunk_id,
                                    source_type=citation.source_type,
                                    source_uri=citation.source_uri,
                                    title=citation.title,
                                )
                                for citation in rag_context.citations
                            ),
                        }
                    )
                if any(result.is_error for result in tool_results):
                    raise ToolExecutionError("one or more selected tools returned an error")
                retrieved_evidence_count = sum(
                    len(result.evidences) for result in tool_results
                ) + (len(rag_context.citations) if rag_context is not None else 0)
                web_evidence_count = sum(
                    evidence.source_type == SourceType.WEB
                    for result in tool_results
                    for evidence in result.evidences
                )
                if (
                    role == AgentName.RETRIEVER
                    and (tool_results or rag_context is not None)
                    and retrieved_evidence_count == 0
                ):
                    raise EmptyRetrievalError(
                        f"retrieval task {request.task.task_id} returned no usable evidence"
                    )
                if (
                    self._require_external_evidence
                    and role in {AgentName.RESEARCHER, AgentName.RETRIEVER}
                    and web_evidence_count == 0
                ):
                    raise EmptyRetrievalError(
                        f"research task {request.task.task_id} returned no public web evidence"
                    )
                if tool_results:
                    tool_evidences = tuple(
                        evidence
                        for result in tool_results
                        for evidence in result.evidences
                    )
                    payload = payload.model_copy(
                        update={
                            "retrieved_context": _render_tool_results(
                                tool_results,
                                self._max_context_chars,
                            ),
                            "sources": _merge_research_sources(
                                payload.sources,
                                tool_evidences,
                            ),
                        }
                    )
            except Exception as exc:
                self._record_request_failure(request, role.value, exc)
                raise
            evidence_ids = tuple(
                dict.fromkeys(
                    [
                        evidence_id
                        for claim in payload.claims
                        for evidence_id in claim.evidence_ids
                    ]
                    + [source.evidence_id for source in payload.sources]
                )
            )
            result = self._store_result(request.task, payload, evidence_ids)
            self._record_request_success(request, role.value, result.result_id)
            return result

        return execute

    def _run_research_quality_gate(
        self,
        request: TaskExecutionRequest,
        payload: ResearchOutput,
        dependency_payloads: list[Any],
        tool_results: list[ToolResult],
    ) -> tuple[ResearchOutput, list[ToolResult]]:
        """Verify one task after its initial searches and fill only identified gaps."""

        searched_query_keys = {
            query.casefold() for query in _searched_web_queries(tool_results)
        }
        assessment: TaskQualityAssessment | None = None
        passed = False
        for round_index in range(self._max_research_tool_rounds):
            assessment = self.verifier.verify_task(
                request.user_goal,
                request.task,
                payload,
                tool_results,
            )
            passed = self._task_quality_passed(request.task, assessment, tool_results)
            if passed:
                break
            remaining = self._research_max_searches_per_task - len(searched_query_keys)
            if remaining <= 0 or round_index + 1 >= self._max_research_tool_rounds:
                break
            candidates = tuple(
                dict.fromkeys(
                    query.strip()
                    for query in (
                        *assessment.followup_queries,
                        *payload.recommended_queries,
                        *(
                            f"{request.task.title} {gap}"
                            for gap in assessment.missing_aspects
                        ),
                    )
                    if query.strip()
                    and query.strip().casefold() not in searched_query_keys
                )
            )
            batch = candidates[
                : min(self._research_followup_batch_size, remaining)
            ]
            if not batch:
                break
            followup_results: list[ToolResult] = []
            for query in batch:
                followup_results.extend(self._execute_followup_search(request, query))
                searched_query_keys.add(query.casefold())
            if not followup_results:
                break
            tool_results.extend(followup_results)
            dependency_payloads.extend(followup_results)
            payload = self.researcher.research(
                request.user_goal,
                request.task,
                [payload, *dependency_payloads],
            )

        if assessment is None:
            raise EmptyRetrievalError(
                f"research task {request.task.task_id} was not quality assessed"
            )
        payload = payload.model_copy(
            update={
                "quality_score": assessment.score,
                "quality_complete": passed,
                "searched_queries": tuple(_searched_web_queries(tool_results)),
                "quality_gaps": tuple(
                    dict.fromkeys(
                        (*assessment.missing_aspects, *assessment.unresolved_conflicts)
                    )
                ),
            }
        )
        if not passed:
            sources, domains = _web_source_coverage(tool_results)
            raise EmptyRetrievalError(
                f"research task {request.task.task_id} did not pass its quality gate: "
                f"score={assessment.score}, sources={len(sources)}, "
                f"domains={len(domains)}, searches={len(searched_query_keys)}"
            )
        return payload, tool_results

    def _task_quality_passed(
        self,
        task: ResearchTask,
        assessment: TaskQualityAssessment,
        tool_results: list[ToolResult],
    ) -> bool:
        sources, domains = _web_source_coverage(tool_results)
        criterion_results = {
            item.criterion.strip().casefold(): item.satisfied
            for item in assessment.criteria
        }
        criteria_passed = all(
            criterion_results.get(criterion.strip().casefold(), False)
            for criterion in task.completion_criteria
        )
        supported_claims = [
            claim
            for claim in assessment.claims
            if claim.verified and claim.supporting_evidence_ids
        ]
        claims_passed = not assessment.claims or (
            len(supported_claims) / len(assessment.claims) >= 0.8
        )
        return (
            assessment.task_id == task.task_id
            and assessment.complete
            and assessment.score >= self._research_completion_score
            and len(sources) >= self._research_min_unique_sources
            and len(domains) >= self._research_min_independent_domains
            and criteria_passed
            and claims_passed
            and not assessment.unresolved_conflicts
        )

    def _execute_tool_calls(
        self,
        role: AgentName,
        request: TaskExecutionRequest,
    ) -> list[ToolResult]:
        if self.tool_runtime is None or self.tool_selector is None:
            if self._require_external_evidence and role in {
                AgentName.RESEARCHER,
                AgentName.RETRIEVER,
            }:
                raise ToolExecutionError(
                    "live web research requires an enabled search tool and Tavily API key"
                )
            return []
        specs = self._tool_specs_for_role(role)
        if not specs and role not in {AgentName.RESEARCHER, AgentName.RETRIEVER}:
            return []
        web_search = next(
            (
                spec
                for spec in specs
                if "web" in spec.capabilities and "search" in spec.capabilities
            ),
            None,
        )
        evidence_required = self._require_external_evidence and role in {
            AgentName.RESEARCHER,
            AgentName.RETRIEVER,
        }
        if evidence_required and web_search is None:
            raise ToolExecutionError(
                "live web research requires a configured public web search tool"
            )
        plan = self.tool_selector.select(request.user_goal, request.task, specs)
        if (
            web_search is not None
            and role == AgentName.RESEARCHER
            and self._research_quality_gate_enabled
        ):
            plan = self._expand_planner_search_plan(request, plan, web_search)
        elif web_search is not None:
            bounded_calls = []
            web_search_added = False
            for call in plan.calls:
                if call.tool_name == web_search.name:
                    if web_search_added:
                        continue
                    web_search_added = True
                bounded_calls.append(call)
            plan = ToolCallPlan(calls=tuple(bounded_calls))
        if evidence_required and web_search is not None and not any(
            call.tool_name == web_search.name for call in plan.calls
        ):
            query = (
                f"{request.user_goal} {request.task.title} {request.task.description}"
            )[:1000]
            plan = ToolCallPlan(
                calls=(
                    *plan.calls[:4],
                    PlannedToolCall(
                        tool_name=web_search.name,
                        arguments={
                            "query": query,
                            "max_results": 8,
                            "search_depth": "advanced",
                            "include_domains": list(
                                _normalized_source_domains(request.task)
                            ),
                            "include_url_prefixes": list(
                                request.task.source_url_prefixes
                            ),
                        },
                        purpose="Fetch real public sources required by the research task",
                    ),
                )
            )
        return self._execute_tool_plan(role, request, plan)

    def _expand_planner_search_plan(
        self,
        request: TaskExecutionRequest,
        plan: ToolCallPlan,
        web_search: ToolSpec,
    ) -> ToolCallPlan:
        """Execute the Planner-owned initial query set for one research task."""

        selected_queries = [
            str(call.arguments.get("query", "")).strip()
            for call in plan.calls
            if call.tool_name == web_search.name
            and str(call.arguments.get("query", "")).strip()
        ]
        queries = list(
            dict.fromkeys(
                query.strip()
                for query in (*request.task.initial_search_queries, *selected_queries)
                if query.strip()
            )
        )
        base = (
            f"{request.user_goal} {request.task.title} {request.task.description}"
        ).strip()
        fallbacks = (
            base,
            f"{base} 官方报告 权威数据 最新进展",
            f"{base} 技术挑战 局限 风险",
            f"{base} 实施案例 失败案例 经验教训",
            f"{base} 市场采用 成本 监管 反方观点",
        )
        for query in fallbacks:
            normalized = query[:1000]
            if normalized not in queries:
                queries.append(normalized)
            if len(queries) >= self._research_initial_searches:
                break
        non_web_calls = tuple(
            call for call in plan.calls if call.tool_name != web_search.name
        )
        web_calls = tuple(
            PlannedToolCall(
                tool_name=web_search.name,
                arguments={
                    "query": query[:1000],
                    "max_results": 5,
                    "search_depth": "advanced",
                    "include_domains": list(_normalized_source_domains(request.task)),
                    "include_url_prefixes": list(request.task.source_url_prefixes),
                },
                purpose="Execute a Planner-defined evidence angle for the research task",
            )
            for query in queries[: self._research_initial_searches]
        )
        return ToolCallPlan(calls=(*non_web_calls[:5], *web_calls))

    def _tool_specs_for_role(self, role: AgentName) -> tuple[ToolSpec, ...]:
        if self.tool_runtime is None:
            return ()
        specs = self.tool_runtime.discover()
        if role == AgentName.RESEARCHER:
            return tuple(
                spec
                for spec in specs
                if {"search", "database"}.intersection(spec.capabilities)
            )
        capability = {
            AgentName.RETRIEVER: "search",
            AgentName.DATABASE: "database",
        }.get(role)
        if capability is None:
            return ()
        return tuple(spec for spec in specs if capability in spec.capabilities)

    def _execute_followup_search(
        self,
        request: TaskExecutionRequest,
        query: str,
    ) -> list[ToolResult]:
        web_search = next(
            (
                spec
                for spec in self._tool_specs_for_role(AgentName.RESEARCHER)
                if "web" in spec.capabilities and "search" in spec.capabilities
            ),
            None,
        )
        if web_search is None:
            return []
        return self._execute_tool_plan(
            AgentName.RESEARCHER,
            request,
            ToolCallPlan(
                calls=(
                    PlannedToolCall(
                        tool_name=web_search.name,
                        arguments={
                            "query": query[:1000],
                            "max_results": 5,
                            "search_depth": "advanced",
                            "include_domains": list(
                                _normalized_source_domains(request.task)
                            ),
                            "include_url_prefixes": list(
                                request.task.source_url_prefixes
                            ),
                        },
                        purpose="Fill an evidence gap identified by the Researcher",
                    ),
                )
            ),
        )

    def _execute_tool_plan(
        self,
        role: AgentName,
        request: TaskExecutionRequest,
        plan: ToolCallPlan,
    ) -> list[ToolResult]:
        if self.tool_runtime is None:
            return []
        context = ToolContext(
            run_id=request.run_id,
            user_id=request.user_id,
            task_id=request.task.task_id,
            permissions=self.tool_permissions,
            approvals=request.approvals,
        )
        results = []
        for call in plan.calls:
            arguments = dict(call.arguments)
            if call.tool_name == "web.search":
                arguments["max_results"] = max(5, int(arguments.get("max_results", 5)))
            call_id = self.evidence_store.start_tool_call(
                run_id=request.run_id,
                user_id=request.user_id,
                task_id=request.task.task_id,
                tool_name=call.tool_name,
                purpose=call.purpose,
                arguments=arguments,
            )
            with self.observability.span(
                f"tool.{call.tool_name}",
                kind="tool",
                metadata={"purpose": call.purpose},
            ) as span:
                try:
                    result = self.tool_runtime.invoke(call.tool_name, arguments, context)
                except Exception as exc:
                    self.evidence_store.finish_tool_call(call_id, error=str(exc))
                    if call.tool_name == "database.query" and isinstance(
                        exc,
                        ConnectionError | TimeoutError | psycopg.Error,
                    ):
                        results.append(
                            ToolResult(
                                tool_name=call.tool_name,
                                data={
                                    "database_skipped": True,
                                    "reason": str(exc),
                                },
                                metadata={"optional": True, "skipped": True},
                            )
                        )
                        continue
                    raise
                self.evidence_store.finish_tool_call(
                    call_id,
                    evidences=result.evidences,
                    error="tool returned an error result" if result.is_error else None,
                )
                span.metadata["is_error"] = result.is_error
                span.metadata["evidence_count"] = len(result.evidences)
                results.append(result)
        return results

    def apply_human_feedback(
        self,
        state: AgentState,
        feedback: HumanFeedback,
    ) -> dict[str, Any]:
        """Apply approvals and locally invalidate only affected tasks plus downstream nodes."""

        requirements = state.get("requirements")
        if requirements is None:
            raise ValueError("human feedback requires parsed requirements")
        updated_goal = feedback.revised_goal or state["user_goal"]
        constraints = tuple(
            dict.fromkeys((*requirements.constraints, *feedback.additional_constraints))
        )
        updated_requirements = requirements.model_copy(
            update={"objective": updated_goal, "constraints": constraints}
        )
        change_parts = []
        if feedback.revised_goal:
            change_parts.append(f"revised goal: {feedback.revised_goal}")
        if feedback.additional_constraints:
            change_parts.append(
                "additional constraints: " + ", ".join(feedback.additional_constraints)
            )
        direct_ids = set(feedback.affected_task_ids)
        known_ids = {task.task_id for task in state["tasks"]}
        unknown = direct_ids - known_ids
        if unknown:
            raise ValueError(f"feedback references unknown task ids: {sorted(unknown)}")
        if change_parts and not direct_ids:
            direct_ids = set(
                self.planner.analyze_impact(
                    updated_requirements,
                    state["tasks"],
                    "\n".join(change_parts),
                )
            )
        affected_ids = _downstream_closure(state["tasks"], direct_ids)
        task_updates = []
        reason = "\n".join(change_parts) or "human approval received"
        for task in state["tasks"]:
            if task.task_id in affected_ids:
                revised = self.planner.replan_task(updated_requirements, task, reason)
                task_updates.append(
                    revised.model_copy(
                        update={
                            "status": TaskStatus.PENDING,
                            "attempts": 0,
                            "result_ids": (),
                            "evidence_ids": (),
                            "quality_score": None,
                            "quality_complete": None,
                            "search_count": 0,
                            "quality_gaps": (),
                            "error": None,
                        }
                    )
                )
            elif task.status == TaskStatus.WAITING_HUMAN:
                task_updates.append(
                    task.model_copy(
                        update={"status": TaskStatus.PENDING, "error": None}
                    )
                )
            elif task.status == TaskStatus.RUNNING:
                # A feedback update replaces the queued Send operations at a
                # Supervisor checkpoint. Return unexecuted tasks to the DAG so
                # they are dispatched again instead of remaining RUNNING forever.
                task_updates.append(
                    task.model_copy(
                        update={
                            "status": TaskStatus.PENDING,
                            "attempts": max(0, task.attempts - 1),
                            "error": None,
                        }
                    )
                )
        return {
            "user_goal": updated_goal,
            "requirements": updated_requirements,
            "tasks": task_updates,
            "workflow_status": WorkflowStatus.RUNNING,
            "approvals": set(feedback.approvals),
            "next_agent": "supervisor",
            "dispatched_task_ids": [],
            "checkpoints": [
                CheckpointReference(
                    stage="human_feedback_applied",
                    task_ids=tuple(sorted(affected_ids)),
                )
            ],
        }

    def _verifier_executor(self, request: TaskExecutionRequest) -> TaskExecutionResult:
        try:
            payloads = self.blackboard.payloads_for_tasks(request.task.dependencies)
            payload = self.verifier.verify(request.user_goal, request.task, payloads)
        except Exception as exc:
            self._record_request_failure(request, "verifier", exc)
            raise
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for claim in payload.claims
                for evidence_id in claim.supporting_evidence_ids
            )
        )
        result = self._store_result(request.task, payload, evidence_ids)
        self._record_request_success(request, "verifier", result.result_id)
        return result

    def _writer_executor(self, request: TaskExecutionRequest) -> TaskExecutionResult:
        try:
            payloads = self.blackboard.payloads_for_tasks(request.task.dependencies)
            payload = self.writer.write(
                request.user_goal,
                request.requirements,
                request.task,
                payloads,
            )
            if self._require_external_evidence:
                evidence_records = self.evidence_store.list_evidences(request.run_id)
                if not evidence_records:
                    raise EmptyRetrievalError(
                        "report generation requires evidence from a real external tool call"
                    )
                payload = payload.model_copy(
                    update={
                        "markdown_report": _append_source_appendix(
                            payload.markdown_report,
                            evidence_records,
                        )
                    }
                )
        except Exception as exc:
            self._record_request_failure(request, "writer", exc)
            raise
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for section in payload.sections
                for evidence_id in section.supporting_evidence_ids
            )
        )
        result = self._store_result(request.task, payload, evidence_ids)
        self._record_request_success(request, "writer", result.result_id)
        return result

    def _record_episode(
        self,
        state: AgentState,
        event_type: EpisodeType,
        *,
        stage: str,
        decision: str | None = None,
        failure_reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.episodic_memory.record(
            EpisodeWrite(
                run_id=state["run_id"],
                user_id=state["user_id"],
                event_type=event_type,
                stage=stage,
                decision=decision,
                failure_reason=failure_reason,
                payload=payload or {},
            )
        )

    def _record_request_success(
        self,
        request: TaskExecutionRequest,
        stage: str,
        result_id: str,
    ) -> None:
        self.episodic_memory.record(
            EpisodeWrite(
                run_id=request.run_id,
                user_id=request.user_id,
                event_type=EpisodeType.TASK_COMPLETED,
                stage=stage,
                task_id=request.task.task_id,
                decision="task_completed",
                payload={"result_id": result_id},
            )
        )

    def _record_request_failure(
        self,
        request: TaskExecutionRequest,
        stage: str,
        error: Exception,
    ) -> None:
        self.episodic_memory.record(
            EpisodeWrite(
                run_id=request.run_id,
                user_id=request.user_id,
                event_type=EpisodeType.TASK_FAILED,
                stage=stage,
                task_id=request.task.task_id,
                failure_reason=str(error),
            )
        )

    def _store_result(
        self,
        task: ResearchTask,
        payload: BaseModel,
        evidence_ids: tuple[str, ...],
    ) -> TaskExecutionResult:
        result_id = f"result:{task.task_id}:{uuid4().hex}"
        payload_ref = self.blackboard.put(result_id, task.task_id, payload)
        quality_score = payload.quality_score if isinstance(payload, ResearchOutput) else None
        quality_complete = (
            payload.quality_complete if isinstance(payload, ResearchOutput) else None
        )
        search_count = len(payload.searched_queries) if isinstance(payload, ResearchOutput) else 0
        quality_gaps = payload.quality_gaps if isinstance(payload, ResearchOutput) else ()
        return TaskExecutionResult(
            result_id=result_id,
            payload_ref=payload_ref,
            evidence_ids=evidence_ids,
            quality_score=quality_score,
            quality_complete=quality_complete,
            search_count=search_count,
            quality_gaps=quality_gaps,
        )


def _to_research_source(evidence: ToolEvidence) -> ResearchSource:
    return ResearchSource(
        evidence_id=evidence.evidence_id,
        chunk_id=evidence.evidence_id,
        source_type=evidence.source_type,
        source_uri=evidence.source_uri,
        title=evidence.title,
    )


def _merge_research_sources(
    existing: tuple[ResearchSource, ...],
    evidences: tuple[ToolEvidence, ...],
) -> tuple[ResearchSource, ...]:
    sources = {source.evidence_id: source for source in existing}
    for evidence in evidences:
        sources[evidence.evidence_id] = _to_research_source(evidence)
    return tuple(sources.values())


def _render_tool_results(results: list[ToolResult], max_chars: int) -> str:
    """Compact every evidence result fairly instead of truncating later searches."""

    evidence_count = sum(len(result.evidences) for result in results)
    snippet_chars = max(180, min(800, max_chars // max(1, evidence_count)))
    sections: list[str] = []
    for result in results:
        query = (
            str(result.data.get("query", "")).strip()
            if isinstance(result.data, dict)
            else ""
        )
        lines = [f"Tool: {result.tool_name}", f"Query: {query or 'n/a'}"]
        if result.evidences:
            for evidence in result.evidences:
                lines.extend(
                    (
                        f"Evidence: {evidence.evidence_id}",
                        f"Title: {evidence.title}",
                        f"URL: {evidence.source_uri}",
                        f"Snippet: {evidence.content[:snippet_chars]}",
                    )
                )
        else:
            lines.append(f"Result: {str(result.data)[:snippet_chars]}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)[:max_chars]


def _searched_web_queries(results: list[ToolResult]) -> list[str]:
    queries: list[str] = []
    for result in results:
        if result.tool_name != "web.search" or not isinstance(result.data, dict):
            continue
        query = str(result.data.get("query", "")).strip()
        if query and query not in queries:
            queries.append(query)
    return queries


def _web_source_coverage(
    results: list[ToolResult],
) -> tuple[set[str], set[str]]:
    sources: set[str] = set()
    domains: set[str] = set()
    for result in results:
        for evidence in result.evidences:
            if evidence.source_type != SourceType.WEB:
                continue
            source_uri = evidence.source_uri.strip()
            if not source_uri:
                continue
            sources.add(source_uri)
            domain = urlparse(source_uri).netloc.casefold()
            if domain:
                domains.add(domain)
    return sources, domains


def _normalized_source_domains(task: ResearchTask) -> tuple[str, ...]:
    """Normalize Planner-provided hostnames before passing them to search tools."""

    domains: list[str] = []
    for value in task.source_domains:
        candidate = value.strip().casefold()
        if not candidate:
            continue
        if "://" in candidate:
            candidate = urlparse(candidate).netloc.casefold()
        else:
            candidate = candidate.split("/", 1)[0]
        candidate = candidate.strip(".")
        if candidate and candidate not in domains:
            domains.append(candidate)
    return tuple(domains)


def _append_source_appendix(
    report: str,
    evidences: tuple[EvidenceRecord, ...],
) -> str:
    if "## 可核验来源" in report:
        return report
    lines = [report.rstrip(), "", "## 可核验来源", ""]
    for evidence in evidences[:30]:
        title = evidence.title.replace("[", "").replace("]", "") or evidence.source_url
        lines.append(
            f"- [{title}]({evidence.source_url}) — `{evidence.evidence_id}` "
            f"(相关度 {evidence.confidence:.2f}, {evidence.created_time.isoformat()})"
        )
    return "\n".join(lines)


def _downstream_closure(
    tasks: list[ResearchTask],
    direct_ids: set[str],
) -> set[str]:
    affected = set(direct_ids)
    changed = True
    while changed:
        changed = False
        for task in tasks:
            if task.task_id not in affected and affected.intersection(task.dependencies):
                affected.add(task.task_id)
                changed = True
    return affected


def build_agent_workflow(
    runtime: AgentRuntime,
    *,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> Any:
    """Compile LangGraph with real Agent nodes and a shared Blackboard."""

    return build_workflow(
        runtime.executors(),
        checkpointer=checkpointer,
        requirement_parser=runtime.requirement_parser_node,
        planner=runtime.planner_node,
        supervisor=runtime.supervisor_node,
        finalizer=runtime.finalize_node,
        recovery_policy=runtime.recovery_policy,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )


def run_agent_workflow(
    user_goal: str | None,
    *,
    settings: Settings | None = None,
    runtime: AgentRuntime | None = None,
    tasks: list[ResearchTask] | None = None,
    config: dict[str, Any] | None = None,
    checkpointer: Any | None = None,
    thread_id: str | None = None,
    user_id: str = "anonymous",
    conversation_id: str | None = None,
    user_message: str | None = None,
    parent_run_id: str | None = None,
    selected_model: str | None = None,
    resume: bool = False,
    human_feedback: HumanFeedback | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> AgentState:
    """Start or resume an API-backed Agent workflow and return its final state."""

    active_runtime = runtime or AgentRuntime.from_settings(settings)
    if resume and (checkpointer is None or thread_id is None):
        raise ValueError("resume requires both checkpointer and thread_id")
    if human_feedback is not None and not resume:
        raise ValueError("human_feedback can only be applied when resume=true")
    if not resume and (user_goal is None or not user_goal.strip()):
        raise ValueError("user_goal is required when starting a workflow")

    run_id = thread_id or str(uuid4())
    active_config = dict(config or {})
    if checkpointer is not None:
        checkpoint_settings = checkpoint_config(run_id)
        active_config["configurable"] = {
            **active_config.get("configurable", {}),
            **checkpoint_settings["configurable"],
        }
    graph = build_agent_workflow(
        active_runtime,
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )
    graph_input = (
        None
        if resume
        else create_initial_state(
            user_goal or "",
            tasks,
            run_id=run_id,
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
            parent_run_id=parent_run_id,
            selected_model=selected_model,
        )
    )
    if resume and human_feedback is not None:
        snapshot = graph.get_state(active_config)
        if not snapshot.values:
            raise ValueError(f"checkpoint {run_id!r} has no workflow state")
        graph.update_state(
            active_config,
            active_runtime.apply_human_feedback(snapshot.values, human_feedback),
            as_node="planner",
        )
    return graph.invoke(graph_input, config=active_config)
