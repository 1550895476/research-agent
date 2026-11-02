"""Budget governance, Lead routing, approval, and Worker isolation tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from agents.lead import LeadRouter
from agents.researcher import ResearchClaim, ResearchOutput
from agents.runtime import AgentRuntime, run_agent_workflow
from agents.verifier import VerificationOutput, VerifiedClaim
from core.budget import BudgetExceeded, BudgetLedger, RunBudget, TaskBudget
from evals.metrics import evaluate_research_quality
from evidence.store import EvidenceRecord
from graph.state import (
    AgentName,
    ResearchRequirement,
    ResearchTask,
    ResearchType,
    TaskStatus,
    WorkflowStatus,
)
from graph.workflow import TaskExecutionRequest
from guardrail.middleware import GuardedToolRuntime, GuardrailMiddleware
from tests.test_agents import SchemaFakeModel
from tests.test_tools import FakeSearchClient
from tools.registry import ToolRegistry, ToolSpec
from tools.search import TavilySearchTool


def test_budget_snapshot_remains_available_after_token_circuit_break() -> None:
    policy = RunBudget(
        max_input_tokens=1000,
        max_output_tokens=1000,
        max_tasks=3,
        large_input_cost_per_million=2,
    )
    ledger = BudgetLedger(policy)
    ledger.start_run("run-budget")
    ledger.register_plan("run-budget", 3)
    ledger.reserve_model_input(
        "run-budget",
        "research",
        tier="large",
        tokens=900,
    )

    with pytest.raises(BudgetExceeded, match="input token budget exceeded"):
        ledger.reserve_model_input(
            "run-budget",
            "research",
            tier="large",
            tokens=101,
        )

    snapshot = ledger.snapshot("run-budget")
    assert snapshot.status == "exhausted"
    assert snapshot.input_tokens == 900
    assert snapshot.estimated_cost_usd == pytest.approx(0.0018)
    assert "input token budget exceeded" in (snapshot.exhausted_reason or "")


def test_budget_ledger_is_atomic_under_parallel_model_accounting() -> None:
    ledger = BudgetLedger(
        RunBudget(
            max_tasks=3,
            max_input_tokens=100000,
            max_output_tokens=100000,
        )
    )
    ledger.start_run("parallel")

    def account(_: int) -> None:
        ledger.reserve_model_input(
            "parallel",
            "worker",
            tier="small",
            tokens=10,
            task_budget=TaskBudget(max_input_tokens=100000),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(account, range(100)))

    snapshot = ledger.snapshot("parallel")
    assert snapshot.input_tokens == 1000
    assert snapshot.model_calls == 100


def test_lead_routes_simple_request_to_single_and_comparison_to_multi() -> None:
    lead = LeadRouter(run_budget=RunBudget(max_tasks=8, max_concurrent_workers=4))
    simple = ResearchRequirement(
        objective="解释一个术语",
        key_questions=("它是什么？",),
    )
    comparison = ResearchRequirement(
        objective="比较三种 Agent 沙箱方案",
        research_type=ResearchType.TECHNOLOGY,
        key_questions=("容器如何？", "gVisor 如何？", "microVM 如何？"),
    )

    assert lead.decide(simple).mode == "single"
    decision = lead.decide(comparison)
    assert decision.mode == "multi"
    assert 2 <= decision.recommended_workers <= 4


def test_high_cost_plan_waits_for_explicit_plan_approval() -> None:
    model = SchemaFakeModel()
    policy = RunBudget(max_tasks=12)
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        run_budget=policy,
        budget_ledger=BudgetLedger(policy),
        lead_router=LeadRouter(run_budget=policy, approval_mode="always"),
    )

    state = run_agent_workflow("研究 Agent 平台", runtime=runtime)

    assert state["workflow_status"] == WorkflowStatus.WAITING_HUMAN
    assert state["lead_decision"].requires_approval is True
    assert state["human_requests"][-1].approval_token == f"plan:{state['run_id']}"
    assert not runtime.blackboard.snapshot()


def test_research_worker_discovers_only_read_only_tools() -> None:
    registry = ToolRegistry()
    registry.register_lazy(
        ToolSpec(
            name="docs.read",
            description="Read documentation",
            input_schema={"type": "object"},
            capabilities=frozenset({"read", "knowledge_base"}),
        ),
        lambda: pytest.fail("discovery must not load tools"),
    )
    registry.register_lazy(
        ToolSpec(
            name="messages.send",
            description="Send a message",
            input_schema={"type": "object"},
            capabilities=frozenset({"search", "send"}),
        ),
        lambda: pytest.fail("discovery must not load tools"),
    )
    runtime = AgentRuntime(
        small_model=SchemaFakeModel(),
        large_model=SchemaFakeModel(),
        tool_runtime=GuardedToolRuntime(registry, GuardrailMiddleware()),
    )

    specs = runtime._tool_specs_for_role(AgentName.RESEARCHER)

    assert {spec.name for spec in specs} == {"docs.read"}


class ConflictThenResolvedModel:
    def __init__(self) -> None:
        self.verification_calls = 0

    def invoke(self, schema, messages):  # noqa: ANN001, ANN201
        del messages
        if schema is VerificationOutput:
            self.verification_calls += 1
            if self.verification_calls == 1:
                return VerificationOutput(
                    verified=False,
                    confidence=0.4,
                    conflicts=("冷启动时间分别为 100ms 和 2s",),
                )
            return VerificationOutput(
                verified=True,
                confidence=0.8,
                claims=(
                    VerifiedClaim(
                        statement="测试条件不同，两个数字不可直接比较",
                        verified=True,
                        confidence=0.8,
                    ),
                ),
            )
        raise AssertionError(f"unexpected schema: {schema.__name__}")


def test_verifier_runs_bounded_conflict_arbitration_search() -> None:
    model = ConflictThenResolvedModel()
    registry = ToolRegistry()
    registry.register(TavilySearchTool(client=FakeSearchClient()))
    runtime = AgentRuntime(
        small_model=model,
        large_model=model,
        tool_runtime=GuardedToolRuntime(registry, GuardrailMiddleware()),
        tool_permissions=frozenset({"search:execute"}),
        run_budget=RunBudget(max_tasks=3, max_conflict_tasks=1),
    )
    research = ResearchTask(
        task_id="research",
        title="研究",
        description="收集数据",
        assigned_agent=AgentName.RESEARCHER,
    )
    verifier = ResearchTask(
        task_id="verify",
        title="裁决冲突",
        description="核查不一致数据",
        dependencies=("research",),
        assigned_agent=AgentName.VERIFIER,
    )
    runtime.blackboard.put(
        "research-result",
        research.task_id,
        ResearchOutput(summary="两个来源给出不同冷启动时间"),
    )
    result = runtime._verifier_executor(
        TaskExecutionRequest(
            user_goal="比较两种沙箱",
            run_id="conflict-run",
            user_id="user",
            task=verifier,
        )
    )

    payload = runtime.blackboard.get_by_uri(result.payload_ref).payload
    assert model.verification_calls == 2
    assert payload.conflicts == ()
    assert len(runtime.evidence_store.list_tool_calls("conflict-run")) == 1


def test_quality_metrics_detect_ungrounded_citations_and_invalid_urls() -> None:
    task = ResearchTask(
        task_id="research",
        title="研究",
        description="研究官方资料",
        source_domains=("official.example",),
        assigned_agent=AgentName.RESEARCHER,
        status=TaskStatus.COMPLETED,
        quality_complete=True,
    )
    payload = ResearchOutput(
        summary="结论",
        claims=(
            ResearchClaim(
                claim_id="c1",
                statement="有证据的结论",
                confidence=0.8,
                evidence_ids=("e1", "invented"),
            ),
        ),
    )
    evidences = (
        EvidenceRecord(
            evidence_id="e1",
            run_id="run",
            user_id="user",
            task_id="research",
            tool_name="web.search",
            source_type="web",
            source_url="https://official.example/report",
            content="source",
            confidence=0.9,
        ),
        EvidenceRecord(
            evidence_id="e2",
            run_id="run",
            user_id="user",
            task_id="research",
            tool_name="web.search",
            source_type="web",
            source_url="unknown://bad",
            content="bad source",
            confidence=0.2,
        ),
    )

    metrics = evaluate_research_quality({"tasks": [task]}, [payload], evidences)

    assert metrics.claim_support_rate == 1
    assert metrics.citation_grounding_rate == 0.5
    assert metrics.structurally_valid_url_rate == 0.5
    assert metrics.first_party_sources == 1
