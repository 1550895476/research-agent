"""Planner Agent: structured research requirements to a validated task DAG."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.model import StructuredModel
from graph.state import AgentName, ResearchRequirement, ResearchTask, TaskDAG


class PlannedTask(BaseModel):
    """LLM-facing task schema without mutable execution fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    initial_search_queries: tuple[str, ...] = Field(default=(), max_length=10)
    completion_criteria: tuple[str, ...] = Field(default=(), max_length=10)
    source_domains: tuple[str, ...] = Field(
        default=(),
        max_length=20,
        description=(
            "Allowed hostnames when the user requires official or domain-restricted sources; "
            "for example docs.example.com or github.com. Never include URL paths."
        ),
    )
    source_url_prefixes: tuple[str, ...] = Field(
        default=(),
        max_length=20,
        description=(
            "Allowed HTTPS URL prefixes for an official repository or site section; for example "
            "https://github.com/example/project. Leave empty when domain filtering is enough."
        ),
    )
    dependencies: tuple[str, ...] = ()
    assigned_agent: AgentName


class ResearchPlan(BaseModel):
    """Schema-constrained Planner response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rationale: str = Field(min_length=1, max_length=2000)
    tasks: tuple[PlannedTask, ...] = Field(min_length=3, max_length=20)

    @model_validator(mode="after")
    def validate_task_roles(self) -> ResearchPlan:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("planned task_id values must be unique")
        agents = {task.assigned_agent for task in self.tasks}
        if AgentName.VERIFIER not in agents:
            raise ValueError("research plan must contain a verifier task")
        if AgentName.WRITER not in agents:
            raise ValueError("research plan must contain a writer task")
        return self


class PlanImpact(BaseModel):
    """Task identifiers affected by an incremental requirement change."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rationale: str = Field(min_length=1, max_length=2000)
    affected_task_ids: tuple[str, ...] = ()


class TaskRevision(BaseModel):
    """Local replacement for one failed or user-invalidated DAG node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rationale: str = Field(min_length=1, max_length=2000)
    task: PlannedTask


class PlannerAgent:
    """Generate and validate a dependency-aware research plan."""

    def __init__(self, model: StructuredModel, *, require_retrieval: bool = False) -> None:
        self._model = model
        self._require_retrieval = require_retrieval

    def plan(
        self,
        requirement: ResearchRequirement,
        *,
        memory_context: tuple[str, ...] = (),
    ) -> list[ResearchTask]:
        """Return executable tasks or raise when the generated graph is invalid."""

        evidence_requirement = (
            "Every researcher task must collect current public web evidence before making factual "
            "claims. "
            if self._require_retrieval
            else "Researchers should use tools whenever the task needs external facts. "
        )
        messages = [
            SystemMessage(
                content=(
                    "You are the Planner for a multi-agent research system. Produce a compact DAG "
                    "with a dynamic number of domain research tasks assigned to researcher, "
                    "followed by verification and one final writer. Create one researcher task "
                    "for each distinct research module explicitly requested by the user. When "
                    "modules are not explicit, derive the smallest complete set from the objective "
                    "and key questions. Keep independent researcher tasks dependency-free so all N "
                    "modules can run in parallel; do not use a fixed researcher-task count. "
                    "For every researcher task, define exactly five complementary "
                    "initial_search_queries covering different evidence angles, plus concrete "
                    "completion_criteria that a Verifier can score. Planner owns module scope and "
                    "the initial search plan; researchers execute it and may only perform "
                    "gap-filling searches within that scope. "
                    "When the user restricts sources to official publishers or named websites, "
                    "populate source_domains with the relevant official hostnames for each "
                    "research task. Use hostnames only, without schemes or paths; otherwise leave "
                    "source_domains empty. When an official repository or a specific site section "
                    "is required, also populate source_url_prefixes with exact HTTPS URL prefixes "
                    "so unrelated content on shared hosts such as github.com is excluded. "
                    "A researcher is an "
                    "autonomous task owner that dynamically selects web search, knowledge-base, "
                    "and permitted read-only database tools; do not create retriever or database "
                    "Agent tasks because those capabilities are tools. "
                    + evidence_requirement
                    + "Include at least one verifier after the research tasks and exactly one "
                    "final writer after verification. Dependencies must reference existing "
                    "task_id values and the graph must be acyclic."
                )
            ),
            HumanMessage(
                content=(
                    f"Requirement:\n{requirement.model_dump_json(indent=2)}\n\n"
                    "Relevant historical memory and failure lessons:\n"
                    + ("\n".join(memory_context) if memory_context else "None")
                )
            ),
        ]
        draft = self._model.invoke(ResearchPlan, messages)
        tasks = [
            ResearchTask(
                task_id=task.task_id,
                title=task.title,
                description=task.description,
                initial_search_queries=task.initial_search_queries,
                completion_criteria=task.completion_criteria,
                source_domains=task.source_domains,
                source_url_prefixes=task.source_url_prefixes,
                dependencies=task.dependencies,
                assigned_agent=_normalize_research_role(task.assigned_agent),
            )
            for task in draft.tasks
        ]
        tasks = _ensure_research_execution_contract(tasks, requirement)
        tasks = _ensure_research_verification_path(tasks)
        TaskDAG(tasks)
        writer_count = sum(task.assigned_agent == AgentName.WRITER for task in tasks)
        if writer_count != 1:
            raise ValueError("research plan must contain exactly one writer task")
        return tasks

    def analyze_impact(
        self,
        requirement: ResearchRequirement,
        tasks: list[ResearchTask],
        change_summary: str,
    ) -> tuple[str, ...]:
        """Identify directly affected nodes; downstream closure is computed deterministically."""

        messages = [
            SystemMessage(
                content=(
                    "Identify only DAG tasks whose inputs or expected outputs are invalidated by "
                    "the user's change. Do not include downstream tasks merely because they depend "
                    "on an affected task; the runtime computes dependency closure. Return only "
                    "task_id values present in the supplied plan."
                )
            ),
            HumanMessage(
                content=(
                    f"Updated requirement:\n{requirement.model_dump_json(indent=2)}\n\n"
                    f"Change:\n{change_summary}\n\nCurrent tasks:\n"
                    + "\n".join(task.model_dump_json() for task in tasks)
                )
            ),
        ]
        impact = self._model.invoke(PlanImpact, messages)
        known = {task.task_id for task in tasks}
        unknown = set(impact.affected_task_ids) - known
        if unknown:
            raise ValueError(f"impact analysis returned unknown task ids: {sorted(unknown)}")
        return impact.affected_task_ids

    def replan_task(
        self,
        requirement: ResearchRequirement,
        task: ResearchTask,
        reason: str,
    ) -> ResearchTask:
        """Revise one task while preserving its identifier and dependency boundary."""

        messages = [
            SystemMessage(
                content=(
                    "Locally revise exactly one research DAG node. Preserve task_id and dependency "
                    "IDs so unaffected completed work remains reusable. Change the description, "
                    "queries, or assigned research role to address the failure or user update."
                )
            ),
            HumanMessage(
                content=(
                    f"Requirement:\n{requirement.model_dump_json(indent=2)}\n\n"
                    f"Current task:\n{task.model_dump_json(indent=2)}\n\nReason:\n{reason}"
                )
            ),
        ]
        revision = self._model.invoke(TaskRevision, messages)
        planned = revision.task
        if planned.task_id != task.task_id:
            raise ValueError("local re-planning must preserve task_id")
        if planned.dependencies != task.dependencies:
            raise ValueError("local re-planning must preserve dependencies")
        assigned_agent = _normalize_research_role(planned.assigned_agent)
        if task.assigned_agent in {AgentName.VERIFIER, AgentName.WRITER}:
            assigned_agent = task.assigned_agent
        elif assigned_agent in {AgentName.VERIFIER, AgentName.WRITER}:
            assigned_agent = AgentName.RESEARCHER
        revised = task.model_copy(
            update={
                "title": planned.title,
                "description": planned.description,
                "initial_search_queries": planned.initial_search_queries,
                "completion_criteria": planned.completion_criteria,
                "source_domains": planned.source_domains,
                "source_url_prefixes": planned.source_url_prefixes,
                "assigned_agent": assigned_agent,
                "status": task.status,
                "quality_score": None,
                "quality_complete": None,
                "search_count": 0,
                "quality_gaps": (),
                "error": None,
            }
        )
        return _ensure_research_execution_contract([revised], requirement)[0]


def _normalize_research_role(agent: AgentName) -> AgentName:
    """Map legacy retrieval Agent roles to the autonomous Researcher."""

    if agent in {AgentName.RETRIEVER, AgentName.DATABASE}:
        return AgentName.RESEARCHER
    return agent


def _ensure_research_execution_contract(
    tasks: list[ResearchTask],
    requirement: ResearchRequirement,
) -> list[ResearchTask]:
    """Give every research node a Planner-owned five-query plan and quality criteria."""

    normalized: list[ResearchTask] = []
    for task in tasks:
        if task.assigned_agent != AgentName.RESEARCHER:
            normalized.append(task)
            continue
        queries = list(
            dict.fromkeys(
                query.strip()
                for query in task.initial_search_queries
                if query.strip()
            )
        )
        base = f"{requirement.objective} {task.title} {task.description}".strip()
        fallbacks = (
            base,
            f"{base} 官方报告 权威数据 最新进展",
            f"{base} 技术挑战 局限 风险",
            f"{base} 实施案例 失败案例 经验教训",
            f"{base} 市场采用 成本 监管 反方观点",
        )
        for query in fallbacks:
            if query not in queries:
                queries.append(query[:1000])
            if len(queries) == 5:
                break
        criteria = tuple(
            dict.fromkeys(
                criterion.strip()
                for criterion in task.completion_criteria
                if criterion.strip()
            )
        ) or (
            f"完整回答子任务：{task.title}",
            "关键事实结论均绑定可追溯证据",
            "使用多个相互独立来源并识别来源间冲突",
        )
        normalized.append(
            task.model_copy(
                update={
                    "initial_search_queries": tuple(queries[:5]),
                    "completion_criteria": criteria,
                }
            )
        )
    return normalized


def _ensure_research_verification_path(
    tasks: list[ResearchTask],
) -> list[ResearchTask]:
    """Ensure every research result reaches verification and the final writer."""

    research_ids = tuple(
        task.task_id for task in tasks if task.assigned_agent == AgentName.RESEARCHER
    )
    verifier_ids = tuple(
        task.task_id for task in tasks if task.assigned_agent == AgentName.VERIFIER
    )
    if not research_ids:
        raise ValueError("research plan must contain at least one researcher task")
    return [
        task.model_copy(
            update={
                "dependencies": tuple(dict.fromkeys((*task.dependencies, *research_ids)))
            }
        )
        if task.assigned_agent == AgentName.VERIFIER
        else task.model_copy(
            update={
                "dependencies": tuple(dict.fromkeys((*task.dependencies, *verifier_ids)))
            }
        )
        if task.assigned_agent == AgentName.WRITER
        else task
        for task in tasks
    ]
