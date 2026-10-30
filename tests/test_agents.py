"""Agent-level tests using a schema-aware fake model with no external API calls."""

from collections import Counter
from collections.abc import Sequence
from threading import Lock

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agents.planner import PlannedTask, PlannerAgent, ResearchPlan
from agents.researcher import ResearchClaim, ResearcherAgent, ResearchOutput
from agents.runtime import AgentRuntime, run_agent_workflow
from agents.verifier import VerificationOutput, VerifiedClaim
from agents.writer import ReportDraft, ReportSection
from graph.state import (
    AgentName,
    ResearchRequirement,
    ResearchType,
    TaskStatus,
    WorkflowStatus,
)
from memory.episodic import EpisodeType
from memory.long_term import MemoryKind


class SchemaFakeModel:
    """Return deterministic values based on the requested Pydantic schema."""

    def __init__(self) -> None:
        self._calls: list[str] = []
        self._lock = Lock()

    @property
    def calls(self) -> Counter[str]:
        with self._lock:
            return Counter(self._calls)

    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        assert messages
        with self._lock:
            self._calls.append(schema.__name__)
        if schema is ResearchRequirement:
            return ResearchRequirement(
                objective="研究大模型 Agent 平台发展趋势",
                research_type=ResearchType.INDUSTRY,
                key_questions=("主要厂商有哪些？", "未来技术方向是什么？"),
                deliverables=("结构化研究报告",),
            )
        if schema is ResearchPlan:
            return ResearchPlan(
                rationale="并行收集三类信息，再验证并写作",
                tasks=(
                    PlannedTask(
                        task_id="analyze_market",
                        title="市场分析",
                        description="分析市场和厂商格局",
                        source_domains=("official.example",),
                        source_url_prefixes=("https://official.example/research",),
                        assigned_agent=AgentName.RESEARCHER,
                    ),
                    PlannedTask(
                        task_id="find_sources",
                        title="公开资料",
                        description="设计公开资料检索方案",
                        assigned_agent=AgentName.RETRIEVER,
                    ),
                    PlannedTask(
                        task_id="check_internal",
                        title="内部知识",
                        description="识别内部知识缺口",
                        assigned_agent=AgentName.DATABASE,
                    ),
                    PlannedTask(
                        task_id="verify_claims",
                        title="验证结论",
                        description="交叉检查结论",
                        dependencies=("analyze_market", "find_sources", "check_internal"),
                        assigned_agent=AgentName.VERIFIER,
                    ),
                    PlannedTask(
                        task_id="write_report",
                        title="生成报告",
                        description="编写最终报告",
                        dependencies=("verify_claims",),
                        assigned_agent=AgentName.WRITER,
                    ),
                ),
            )
        if schema is ResearchOutput:
            return ResearchOutput(
                summary="阶段性研究结果",
                claims=(
                    ResearchClaim(
                        claim_id="claim_1",
                        statement="Agent 平台正在强化编排和工具治理",
                        confidence=0.6,
                    ),
                ),
                information_gaps=("尚未连接外部检索工具",),
            )
        if schema is VerificationOutput:
            return VerificationOutput(
                verified=False,
                confidence=0.45,
                claims=(
                    VerifiedClaim(
                        statement="Agent 平台正在强化编排和工具治理",
                        verified=False,
                        confidence=0.45,
                        conflict="缺少外部来源证据",
                    ),
                ),
                limitations=("Step 3 尚未连接检索工具",),
            )
        if schema is ReportDraft:
            return ReportDraft(
                title="大模型 Agent 平台趋势调研",
                executive_summary="当前结果为未接入外部检索的 MVP 输出。",
                sections=(
                    ReportSection(
                        heading="研究结论",
                        content="现有结论仍需外部证据验证。",
                    ),
                ),
                limitations=("尚未接入外部资料",),
                markdown_report="# 大模型 Agent 平台趋势调研\n\n现有结论仍需外部证据验证。",
            )
        raise AssertionError(f"unexpected schema: {schema.__name__}")


class NoRetrieverPlanModel:
    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        assert schema is ResearchPlan
        assert messages
        return ResearchPlan(
            rationale="A deliberately incomplete plan",
            tasks=(
                PlannedTask(
                    task_id="analyze_topic",
                    title="Analyze",
                    description="Analyze the topic",
                    assigned_agent=AgentName.RESEARCHER,
                ),
                PlannedTask(
                    task_id="verify_topic",
                    title="Verify",
                    description="Verify findings",
                    dependencies=("analyze_topic",),
                    assigned_agent=AgentName.VERIFIER,
                ),
                PlannedTask(
                    task_id="write_topic",
                    title="Write",
                    description="Write report",
                    dependencies=("verify_topic",),
                    assigned_agent=AgentName.WRITER,
                ),
            ),
        )


class ManyModulePlanModel:
    """Return more than the former fixed 2-4 Researcher tasks."""

    def invoke(
        self,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> BaseModel:
        assert schema is ResearchPlan
        assert "dynamic number" in str(messages[0].content)
        research_tasks = tuple(
            PlannedTask(
                task_id=f"research_module_{index}",
                title=f"Module {index}",
                description=f"Research module {index}",
                assigned_agent=AgentName.RESEARCHER,
            )
            for index in range(1, 7)
        )
        return ResearchPlan(
            rationale="Research every explicitly requested module in parallel.",
            tasks=(
                *research_tasks,
                PlannedTask(
                    task_id="verify_modules",
                    title="Verify modules",
                    description="Cross-check all module findings.",
                    dependencies=tuple(task.task_id for task in research_tasks),
                    assigned_agent=AgentName.VERIFIER,
                ),
                PlannedTask(
                    task_id="write_modules",
                    title="Write report",
                    description="Write the integrated report.",
                    dependencies=("verify_modules",),
                    assigned_agent=AgentName.WRITER,
                ),
            ),
        )


def test_real_agent_runtime_completes_with_structured_fake_model() -> None:
    model = SchemaFakeModel()
    runtime = AgentRuntime(small_model=model, large_model=model)

    assert isinstance(runtime.researcher, ResearcherAgent)
    assert not hasattr(runtime, "researchers")

    state = run_agent_workflow("调研大模型 Agent 平台", runtime=runtime)

    assert state["workflow_status"] == WorkflowStatus.COMPLETED
    assert state["requirements"] is not None
    assert all(task.status == TaskStatus.COMPLETED for task in state["tasks"])
    assert {task.assigned_agent for task in state["tasks"]}.issubset(
        {AgentName.RESEARCHER, AgentName.VERIFIER, AgentName.WRITER}
    )
    market_task = next(task for task in state["tasks"] if task.task_id == "analyze_market")
    assert market_task.source_domains == ("official.example",)
    assert market_task.source_url_prefixes == ("https://official.example/research",)
    assert state["final_report"].startswith("# 大模型 Agent 平台趋势调研")
    assert len(runtime.blackboard.snapshot()) == 5
    history = runtime.long_term_memory.search(
        "anonymous",
        kinds=(MemoryKind.RESEARCH_HISTORY,),
    )
    episodes = runtime.episodic_memory.list_events(user_id="anonymous")
    assert len(history) == 1
    assert any(event.event_type == EpisodeType.PLAN_CREATED for event in episodes)
    assert any(event.event_type == EpisodeType.WORKFLOW_COMPLETED for event in episodes)
    assert model.calls == Counter(
        {
            "ResearchRequirement": 1,
            "ResearchPlan": 1,
            "ResearchOutput": 3,
            "VerificationOutput": 1,
            "ReportDraft": 1,
        }
    )


def test_planner_uses_autonomous_researchers_before_verification() -> None:
    planner = PlannerAgent(NoRetrieverPlanModel(), require_retrieval=True)
    requirement = ResearchRequirement(
        objective="Research current agent platforms",
        key_questions=("Which sources support the findings?",),
    )

    tasks = planner.plan(requirement)

    researchers = [
        task for task in tasks if task.assigned_agent == AgentName.RESEARCHER
    ]
    verifier = next(task for task in tasks if task.assigned_agent == AgentName.VERIFIER)
    writer = next(task for task in tasks if task.assigned_agent == AgentName.WRITER)

    assert researchers
    assert not any(
        task.assigned_agent in {AgentName.RETRIEVER, AgentName.DATABASE}
        for task in tasks
    )
    assert {task.task_id for task in researchers}.issubset(verifier.dependencies)
    assert verifier.task_id in writer.dependencies


def test_planner_supports_n_parallel_research_modules() -> None:
    planner = PlannerAgent(ManyModulePlanModel())
    requirement = ResearchRequirement(
        objective="Research six product modules",
        key_questions=tuple(f"What is module {index}?" for index in range(1, 7)),
    )

    tasks = planner.plan(requirement)

    researchers = [
        task for task in tasks if task.assigned_agent == AgentName.RESEARCHER
    ]
    verifier = next(task for task in tasks if task.assigned_agent == AgentName.VERIFIER)
    assert len(researchers) == 6
    assert all(not task.dependencies for task in researchers)
    assert all(len(task.initial_search_queries) == 5 for task in researchers)
    assert all(task.completion_criteria for task in researchers)
    assert {task.task_id for task in researchers}.issubset(verifier.dependencies)
