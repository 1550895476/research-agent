"""Offline tests for MCP discovery, tool governance, and Agent integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agents.researcher import ResearchOutput
from agents.runtime import AgentRuntime
from agents.tool_selector import PlannedToolCall, ToolCallPlan
from agents.verifier import CriterionAssessment, TaskQualityAssessment
from core.config import Settings
from graph.state import AgentName, ResearchTask
from graph.workflow import TaskExecutionRequest
from guardrail.middleware import (
    GuardedToolRuntime,
    GuardrailMiddleware,
    InvalidToolArguments,
    ToolPermissionDenied,
    ToolQuotaExceeded,
)
from tools.database import PostgresQueryTool
from tools.factory import create_tool_registry
from tools.registry import (
    MCPToolProvider,
    StructuredTool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolRisk,
    ToolSpec,
)
from tools.search import TavilySearchTool


class EchoInput(BaseModel):
    value: str


class EchoTool(StructuredTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="echo",
                description="Echo a validated value.",
                input_schema=EchoInput.model_json_schema(),
                capabilities=frozenset({"test"}),
                required_permission="echo:use",
                risk=ToolRisk.LOW,
            ),
            EchoInput,
        )
        self.calls = 0

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        self.calls += 1
        return ToolResult(tool_name=self.spec.name, data=arguments)


class KnowledgeCatalogTool(StructuredTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="knowledge_base.search",
                description="Search a test knowledge base.",
                input_schema=EchoInput.model_json_schema(),
                capabilities=frozenset({"search", "knowledge_base"}),
                risk=ToolRisk.LOW,
            ),
            EchoInput,
        )

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        return ToolResult(tool_name=self.spec.name, data=arguments)


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "results": [
                {
                    "title": "Agent report",
                    "url": "https://example.test/agent-report",
                    "content": "alpha evidence from a public source",
                    "score": 0.87,
                }
            ],
            "response_time": 0.1,
        }


class DiverseSearchClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        index = len(self.calls)
        return {
            "results": [
                {
                    "title": f"Independent report {index}",
                    "url": f"https://source-{index}.test/report",
                    "content": f"independent evidence {index}",
                    "score": 0.9,
                }
            ],
            "response_time": 0.1,
        }


class MixedRepositorySearchClient:
    def search(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "results": [
                {
                    "title": "Official repository",
                    "url": "https://github.com/langchain-ai/langgraph/tree/main/libs",
                    "content": "official evidence",
                    "score": 0.9,
                },
                {
                    "title": "Unrelated repository",
                    "url": "https://github.com/unrelated/langgraph-tutorial",
                    "content": "unofficial evidence",
                    "score": 0.95,
                },
            ],
            "response_time": 0.1,
        }


class FakeMCPProvider(MCPToolProvider):
    def __init__(self) -> None:
        self.discoveries = 0
        self.calls = 0

    def discover_tools(self) -> tuple[ToolSpec, ...]:
        self.discoveries += 1
        return (
            ToolSpec(
                name="remote.lookup",
                description="Look up a remote value.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                capabilities=frozenset({"remote"}),
                provider="mcp:test",
            ),
        )

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        del context
        self.calls += 1
        return ToolResult(tool_name=name, data=arguments)


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.description: object | None = None

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, sql: str, parameters: Any = None) -> None:
        self.executed.append((sql, parameters))
        if sql.strip().upper().startswith("SELECT ID"):
            self.description = object()

    def fetchmany(self, size: int) -> list[dict[str, Any]]:
        return [{"id": 1}, {"id": 2}][:size]


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


class ResearchOnlyModel:
    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        assert messages
        if schema is ResearchOutput:
            return ResearchOutput(summary="tool-backed research")
        raise AssertionError(f"unexpected schema: {schema.__name__}")


class IterativeResearchModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        assert schema is ResearchOutput
        assert messages
        self.calls += 1
        if self.calls == 1:
            return ResearchOutput(
                summary="initial research",
                information_gaps=("Need a second source",),
                recommended_queries=("beta independent evidence",),
            )
        return ResearchOutput(summary="research completed after follow-up")


class QualityGateModel:
    def __init__(self) -> None:
        self.research_calls = 0
        self.verification_calls = 0

    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        assert messages
        if schema is ResearchOutput:
            self.research_calls += 1
            return ResearchOutput(summary=f"research round {self.research_calls}")
        if schema is TaskQualityAssessment:
            self.verification_calls += 1
            complete = self.verification_calls >= 2
            return TaskQualityAssessment(
                task_id="quality_research",
                score=90 if complete else 60,
                complete=complete,
                criteria=(
                    CriterionAssessment(
                        criterion="Cover the task scope",
                        satisfied=complete,
                    ),
                ),
                missing_aspects=() if complete else ("independent counter-evidence",),
                followup_queries=(
                    ()
                    if complete
                    else ("gap evidence one", "gap evidence two")
                ),
            )
        raise AssertionError(f"unexpected schema: {schema.__name__}")


class SearchOnlySelector:
    def select(
        self,
        user_goal: str,
        task: ResearchTask,
        available_tools: Sequence[ToolSpec],
    ) -> ToolCallPlan:
        del user_goal, task
        assert {spec.name for spec in available_tools} == {"web.search"}
        return ToolCallPlan(
            calls=(
                PlannedToolCall(
                    tool_name="web.search",
                    arguments={"query": "alpha agent platform", "max_results": 3},
                ),
            )
        )


class EmptySelector:
    def select(
        self,
        user_goal: str,
        task: ResearchTask,
        available_tools: Sequence[ToolSpec],
    ) -> ToolCallPlan:
        del user_goal, task
        assert {spec.name for spec in available_tools} == {"web.search"}
        return ToolCallPlan()


class DatabaseOnlySelector:
    def select(
        self,
        user_goal: str,
        task: ResearchTask,
        available_tools: Sequence[ToolSpec],
    ) -> ToolCallPlan:
        del user_goal, task
        assert {spec.name for spec in available_tools} == {"database.query"}
        return ToolCallPlan(
            calls=(
                PlannedToolCall(
                    tool_name="database.query",
                    arguments={"sql": "SELECT * FROM internal_factors LIMIT 100"},
                ),
            )
        )


class CatalogRecordingSelector:
    def __init__(self) -> None:
        self.tool_names: set[str] = set()

    def select(
        self,
        user_goal: str,
        task: ResearchTask,
        available_tools: Sequence[ToolSpec],
    ) -> ToolCallPlan:
        del user_goal, task
        self.tool_names = {spec.name for spec in available_tools}
        return ToolCallPlan()


def _context(
    *,
    permissions: frozenset[str] = frozenset(),
    run_id: str = "run-1",
) -> ToolContext:
    return ToolContext(
        run_id=run_id,
        user_id="user-1",
        task_id="task-1",
        permissions=permissions,
    )


def test_registry_discovers_lazy_tools_without_loading_them() -> None:
    registry = ToolRegistry()
    loads = 0

    def load() -> EchoTool:
        nonlocal loads
        loads += 1
        return EchoTool()

    spec = EchoTool().spec
    registry.register_lazy(spec, load)

    assert registry.discover(capabilities=("test",)) == (spec,)
    assert loads == 0
    assert registry.get("echo").spec == spec
    assert registry.get("echo").spec == spec
    assert loads == 1


def test_web_search_filters_shared_hosts_by_exact_url_prefix() -> None:
    tool = TavilySearchTool(client=MixedRepositorySearchClient())

    result = tool.invoke(
        {
            "query": "LangGraph persistence",
            "include_domains": ["github.com"],
            "include_url_prefixes": ["https://github.com/langchain-ai/langgraph"],
        },
        _context(),
    )

    assert len(result.evidences) == 1
    assert result.evidences[0].source_uri.startswith(
        "https://github.com/langchain-ai/langgraph"
    )


def test_registry_discovers_and_invokes_an_mcp_provider() -> None:
    registry = ToolRegistry()
    provider = FakeMCPProvider()

    assert registry.register_mcp_provider(provider) == ("remote.lookup",)
    assert provider.discoveries == 1
    assert provider.calls == 0
    result = GuardrailMiddleware().execute(
        registry,
        "remote.lookup",
        {"query": "alpha"},
        _context(),
    )

    assert result.data == {"query": "alpha"}
    assert provider.calls == 1


def test_guardrail_rejects_invalid_arguments_and_missing_permission() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    guardrail = GuardrailMiddleware()

    with pytest.raises(InvalidToolArguments):
        guardrail.execute(registry, "echo", {}, _context(permissions=frozenset({"echo:use"})))
    with pytest.raises(ToolPermissionDenied):
        guardrail.execute(registry, "echo", {"value": "ok"}, _context())

    assert tool.calls == 0


def test_web_search_quota_is_enforced_before_external_call() -> None:
    client = FakeSearchClient()
    registry = ToolRegistry()
    registry.register(TavilySearchTool(client=client))
    guardrail = GuardrailMiddleware(max_web_search_calls_per_run=1)
    context = _context(permissions=frozenset({"search:execute"}))

    first = guardrail.execute(
        registry,
        "web.search",
        {"query": "agent platform"},
        context,
    )
    with pytest.raises(ToolQuotaExceeded):
        guardrail.execute(
            registry,
            "web.search",
            {"query": "second search"},
            context,
        )

    assert first.evidences[0].source_uri == "https://example.test/agent-report"
    assert len(client.calls) == 1


def test_web_search_quota_is_scoped_to_task_and_run() -> None:
    client = FakeSearchClient()
    registry = ToolRegistry()
    registry.register(TavilySearchTool(client=client))
    guardrail = GuardrailMiddleware(
        max_web_search_calls_per_run=3,
        max_web_search_calls_per_task=2,
    )
    first_task = _context(
        permissions=frozenset({"search:execute"}),
        run_id="quality-run",
    )
    second_task = first_task.model_copy(update={"task_id": "task-2"})

    guardrail.execute(registry, "web.search", {"query": "first"}, first_task)
    guardrail.execute(registry, "web.search", {"query": "second"}, first_task)
    with pytest.raises(ToolQuotaExceeded, match="task task-1"):
        guardrail.execute(registry, "web.search", {"query": "third"}, first_task)
    guardrail.execute(registry, "web.search", {"query": "other task"}, second_task)
    with pytest.raises(ToolQuotaExceeded, match="run quality-run"):
        guardrail.execute(registry, "web.search", {"query": "run limit"}, second_task)

    assert guardrail.usage("quality-run", "web.search") == 3
    assert guardrail.usage("quality-run", "web.search", task_id="task-1") == 2
    assert guardrail.usage("quality-run", "web.search", task_id="task-2") == 1


def test_database_tool_blocks_mutation_before_opening_connection() -> None:
    opened = False

    def connection_factory() -> FakeConnection:
        nonlocal opened
        opened = True
        return FakeConnection()

    registry = ToolRegistry()
    registry.register(
        PostgresQueryTool("postgresql://unused", connection_factory=connection_factory)
    )

    with pytest.raises(InvalidToolArguments):
        GuardrailMiddleware().execute(
            registry,
            "database.query",
            {"sql": "DELETE FROM research_memories"},
            _context(permissions=frozenset({"database:read"})),
        )

    assert opened is False


def test_database_tool_uses_read_only_transaction_and_row_limit() -> None:
    connection = FakeConnection()
    tool = PostgresQueryTool(
        "postgresql://unused",
        max_rows=1,
        connection_factory=lambda: connection,
    )
    registry = ToolRegistry()
    registry.register(tool)

    result = GuardrailMiddleware().execute(
        registry,
        "database.query",
        {"sql": "SELECT id FROM vendors", "max_rows": 10},
        _context(permissions=frozenset({"database:read"})),
    )

    assert connection.cursor_instance.executed[0][0] == "SET TRANSACTION READ ONLY"
    assert result.data == {"rows": [{"id": 1}], "truncated": True}
    assert result.evidences[0].source_uri.startswith("database://query/")


def test_database_tool_is_not_registered_by_default() -> None:
    registry = create_tool_registry(Settings())

    assert registry.discover(capabilities=("database",)) == ()


def test_autonomous_researcher_discovers_search_knowledge_and_database_tools() -> None:
    registry = ToolRegistry()
    registry.register(TavilySearchTool(client=FakeSearchClient()))
    registry.register(KnowledgeCatalogTool())
    registry.register(
        PostgresQueryTool(
            "postgresql://unused",
            connection_factory=FakeConnection,
        )
    )
    selector = CatalogRecordingSelector()
    model = ResearchOnlyModel()
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        tool_runtime=GuardedToolRuntime(registry, GuardrailMiddleware()),
        tool_selector=selector,
    )
    task = ResearchTask(
        task_id="autonomous_research",
        title="Autonomous research",
        description="Choose relevant evidence tools",
        assigned_agent=AgentName.RESEARCHER,
    )

    runtime.executors()[AgentName.RESEARCHER](
        TaskExecutionRequest(
            user_goal="Research agent platforms",
            run_id="run-tool-catalog",
            user_id="user-1",
            task=task,
        )
    )

    assert selector.tool_names == {
        "web.search",
        "knowledge_base.search",
        "database.query",
    }


def test_database_failure_is_recorded_and_skipped_by_autonomous_researcher() -> None:
    def unavailable_database() -> FakeConnection:
        raise ConnectionError("database is not configured")

    registry = ToolRegistry()
    registry.register(
        PostgresQueryTool(
            "postgresql://unused",
            connection_factory=unavailable_database,
        )
    )
    model = ResearchOnlyModel()
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        tool_runtime=GuardedToolRuntime(registry, GuardrailMiddleware()),
        tool_selector=DatabaseOnlySelector(),
        tool_permissions=frozenset({"database:read"}),
    )
    task = ResearchTask(
        task_id="optional_database",
        title="Optional internal data",
        description="Use internal data if available",
        assigned_agent=AgentName.RESEARCHER,
    )

    result = runtime.executors()[AgentName.RESEARCHER](
        TaskExecutionRequest(
            user_goal="Research agent platforms",
            run_id="run-optional-database",
            user_id="user-1",
            task=task,
        )
    )

    calls = runtime.evidence_store.list_tool_calls("run-optional-database")
    assert result.evidence_ids == ()
    assert calls[0].status == "failed"
    assert calls[0].error == "database is not configured"


def test_researcher_uses_selected_guarded_search_and_keeps_evidence_ids() -> None:
    client = FakeSearchClient()
    registry = ToolRegistry()
    registry.register(TavilySearchTool(client=client))
    tool_runtime = GuardedToolRuntime(registry, GuardrailMiddleware())
    model = ResearchOnlyModel()
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        tool_runtime=tool_runtime,
        tool_selector=SearchOnlySelector(),
        tool_permissions=frozenset({"search:execute"}),
    )
    task = ResearchTask(
        task_id="retrieve",
        title="Retrieve sources",
        description="Find public evidence",
        assigned_agent=AgentName.RESEARCHER,
    )

    result = runtime.executors()[AgentName.RESEARCHER](
        TaskExecutionRequest(
            user_goal="Research agent platforms",
            run_id="run-tools",
            user_id="user-1",
            task=task,
        )
    )
    payload = runtime.blackboard.get_by_uri(result.payload_ref).payload

    assert result.evidence_ids == (payload.sources[0].evidence_id,)
    assert payload.sources[0].source_uri == "https://example.test/agent-report"
    assert "alpha evidence" in payload.retrieved_context
    assert len(client.calls) == 1


def test_required_external_research_forces_real_search_and_records_provenance() -> None:
    client = FakeSearchClient()
    registry = ToolRegistry()
    registry.register(TavilySearchTool(client=client))
    model = ResearchOnlyModel()
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        tool_runtime=GuardedToolRuntime(registry, GuardrailMiddleware()),
        tool_selector=EmptySelector(),
        tool_permissions=frozenset({"search:execute"}),
        require_external_evidence=True,
    )
    task = ResearchTask(
        task_id="live_retrieve",
        title="Live retrieval",
        description="Find current public sources",
        assigned_agent=AgentName.RESEARCHER,
    )

    result = runtime.executors()[AgentName.RESEARCHER](
        TaskExecutionRequest(
            user_goal="Research current agent platforms",
            run_id="run-live-tools",
            user_id="user-1",
            task=task,
        )
    )

    calls = runtime.evidence_store.list_tool_calls("run-live-tools")
    evidences = runtime.evidence_store.list_evidences("run-live-tools")
    assert len(client.calls) == 1
    assert client.calls[0]["search_depth"] == "advanced"
    assert result.evidence_ids == (evidences[0].evidence_id,)
    assert calls[0].status == "succeeded"
    assert calls[0].tool_name == "web.search"
    assert calls[0].evidence_ids == (evidences[0].evidence_id,)
    assert evidences[0].source_url == "https://example.test/agent-report"
    assert "alpha evidence" in evidences[0].content


def test_researcher_runs_bounded_followup_search_for_identified_gap() -> None:
    client = FakeSearchClient()
    registry = ToolRegistry()
    registry.register(TavilySearchTool(client=client))
    model = IterativeResearchModel()
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        tool_runtime=GuardedToolRuntime(registry, GuardrailMiddleware()),
        tool_selector=SearchOnlySelector(),
        tool_permissions=frozenset({"search:execute"}),
        max_research_tool_rounds=2,
    )
    task = ResearchTask(
        task_id="iterative_research",
        title="Investigate evidence",
        description="Search, analyze, and fill material evidence gaps",
        assigned_agent=AgentName.RESEARCHER,
    )

    result = runtime.executors()[AgentName.RESEARCHER](
        TaskExecutionRequest(
            user_goal="Research agent platforms",
            run_id="run-iterative-tools",
            user_id="user-1",
            task=task,
        )
    )
    payload = runtime.blackboard.get_by_uri(result.payload_ref).payload

    assert model.calls == 2
    assert len(client.calls) == 2
    assert client.calls[1]["query"] == "beta independent evidence"
    assert payload.summary == "research completed after follow-up"


def test_research_task_runs_planner_queries_then_verifier_driven_followups() -> None:
    client = DiverseSearchClient()
    registry = ToolRegistry()
    registry.register(TavilySearchTool(client=client))
    model = QualityGateModel()
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        tool_runtime=GuardedToolRuntime(registry, GuardrailMiddleware()),
        tool_selector=SearchOnlySelector(),
        tool_permissions=frozenset({"search:execute"}),
        research_quality_gate_enabled=True,
        research_initial_searches=5,
        research_followup_batch_size=2,
        research_max_searches_per_task=10,
        research_completion_score=80,
        research_min_unique_sources=5,
        research_min_independent_domains=3,
        max_research_tool_rounds=3,
    )
    initial_queries = tuple(f"planner query {index}" for index in range(1, 6))
    task = ResearchTask(
        task_id="quality_research",
        title="Research challenges",
        description="Execute the Planner-owned research scope",
        initial_search_queries=initial_queries,
        completion_criteria=("Cover the task scope",),
        source_domains=("docs.example.test", "https://github.com/example/project"),
        assigned_agent=AgentName.RESEARCHER,
    )

    result = runtime.executors()[AgentName.RESEARCHER](
        TaskExecutionRequest(
            user_goal="Research platform challenges",
            run_id="run-quality-gate",
            user_id="user-1",
            task=task,
        )
    )
    payload = runtime.blackboard.get_by_uri(result.payload_ref).payload

    assert [call["query"] for call in client.calls[:5]] == list(initial_queries)
    assert [call["query"] for call in client.calls[5:]] == [
        "gap evidence one",
        "gap evidence two",
    ]
    assert len(client.calls) == 7
    assert all(
        call["include_domains"] == ["docs.example.test", "github.com"]
        for call in client.calls
    )
    assert model.research_calls == 2
    assert model.verification_calls == 2
    assert result.quality_complete is True
    assert result.quality_score == 90
    assert result.search_count == 7
    assert payload.quality_complete is True
    assert payload.quality_score == 90
    assert payload.searched_queries == (*initial_queries, "gap evidence one", "gap evidence two")
