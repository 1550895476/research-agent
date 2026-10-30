"""Pre-execution validation, authorization, quota, and risk middleware."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Any

from pydantic import BaseModel, ConfigDict

from tools.database import validate_read_only_sql
from tools.registry import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolRisk,
    ToolSpec,
)


class GuardrailError(RuntimeError):
    """Base error raised before an unsafe or invalid tool call can execute."""


class InvalidToolArguments(GuardrailError):
    """Tool arguments failed schema or semantic validation."""


class ToolPermissionDenied(GuardrailError):
    """Execution identity lacks the tool's required permission."""


class ToolQuotaExceeded(GuardrailError):
    """A per-task or per-run tool invocation quota has been exhausted."""


class HumanApprovalRequired(GuardrailError):
    """A high-risk tool requires an explicit approval token."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.approval_token = f"tool:{tool_name}"
        super().__init__(f"tool {tool_name} is high risk and requires human approval")


class GuardrailDecision(BaseModel):
    """Auditable result of all checks performed before tool execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    tool_name: str
    checks: tuple[str, ...]
    risk: ToolRisk


class GuardrailMiddleware:
    """Validate every tool call before control reaches the external runtime."""

    def __init__(
        self,
        *,
        max_web_search_calls_per_run: int = 100,
        max_web_search_calls_per_task: int = 10,
    ) -> None:
        if max_web_search_calls_per_run < 1:
            raise ValueError("max_web_search_calls_per_run must be positive")
        if max_web_search_calls_per_task < 1:
            raise ValueError("max_web_search_calls_per_task must be positive")
        self._max_web_search_calls_per_run = max_web_search_calls_per_run
        self._max_web_search_calls_per_task = max_web_search_calls_per_task
        self._run_usage: dict[tuple[str, str], int] = {}
        self._task_usage: dict[tuple[str, str, str], int] = {}
        self._lock = RLock()

    def execute(
        self,
        registry: ToolRegistry,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Run checks in order and invoke the tool only after every check passes."""

        tool = registry.get(tool_name)
        validated = self._validate_arguments(tool.spec, tool.validate_input, arguments)
        self._check_permission(tool.spec, context)
        self._check_semantic_risk(tool.spec, validated)
        self._check_static_risk(tool.spec, context)
        self._consume_quota(tool.spec, context)
        return tool.invoke(validated, context)

    def evaluate(
        self,
        registry: ToolRegistry,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> GuardrailDecision:
        """Validate without executing or consuming quota, useful for approval UIs."""

        tool = registry.get(tool_name)
        validated = self._validate_arguments(tool.spec, tool.validate_input, arguments)
        self._check_permission(tool.spec, context)
        self._check_semantic_risk(tool.spec, validated)
        self._check_static_risk(tool.spec, context)
        return GuardrailDecision(
            allowed=True,
            tool_name=tool_name,
            checks=("arguments", "permission", "semantic_risk", "static_risk"),
            risk=tool.spec.risk,
        )

    def usage(self, run_id: str, tool_name: str, *, task_id: str | None = None) -> int:
        with self._lock:
            if task_id is None:
                return self._run_usage.get((run_id, tool_name), 0)
            return self._task_usage.get((run_id, task_id, tool_name), 0)

    def reset_task_usage(self, run_id: str, task_id: str) -> None:
        """Start a fresh per-task budget after an explicit Planner revision."""

        with self._lock:
            keys = [
                key
                for key in self._task_usage
                if key[0] == run_id and key[1] == task_id
            ]
            for key in keys:
                del self._task_usage[key]

    @staticmethod
    def _validate_arguments(
        spec: ToolSpec,
        validator: Any,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return validator(arguments)
        except Exception as exc:
            raise InvalidToolArguments(
                f"invalid arguments for tool {spec.name}: {exc}"
            ) from exc

    @staticmethod
    def _check_permission(spec: ToolSpec, context: ToolContext) -> None:
        required = spec.required_permission
        if required and required not in context.permissions and "*" not in context.permissions:
            raise ToolPermissionDenied(
                f"permission {required!r} is required for tool {spec.name}"
            )

    @staticmethod
    def _check_semantic_risk(spec: ToolSpec, arguments: dict[str, Any]) -> None:
        if "database" not in spec.capabilities:
            return
        sql = arguments.get("sql")
        if isinstance(sql, str):
            try:
                validate_read_only_sql(sql)
            except ValueError as exc:
                raise InvalidToolArguments(str(exc)) from exc

    @staticmethod
    def _check_static_risk(spec: ToolSpec, context: ToolContext) -> None:
        if spec.risk != ToolRisk.HIGH:
            return
        accepted_tokens = {"high_risk", f"tool:{spec.name}"}
        if not accepted_tokens.intersection(context.approvals):
            raise HumanApprovalRequired(spec.name)

    def _consume_quota(self, spec: ToolSpec, context: ToolContext) -> None:
        if "web" not in spec.capabilities or "search" not in spec.capabilities:
            return
        run_key = (context.run_id, spec.name)
        task_key = (context.run_id, context.task_id, spec.name)
        with self._lock:
            run_used = self._run_usage.get(run_key, 0)
            if run_used >= self._max_web_search_calls_per_run:
                raise ToolQuotaExceeded(
                    f"web search quota exceeded for run {context.run_id}"
                )
            task_used = self._task_usage.get(task_key, 0)
            if task_used >= self._max_web_search_calls_per_task:
                raise ToolQuotaExceeded(
                    f"web search quota exceeded for task {context.task_id}"
                )
            self._run_usage[run_key] = run_used + 1
            self._task_usage[task_key] = task_used + 1


class GuardedToolRuntime:
    """Facade used by Agents for discovery and guarded execution."""

    def __init__(self, registry: ToolRegistry, guardrail: GuardrailMiddleware) -> None:
        self.registry = registry
        self.guardrail = guardrail

    def discover(self, capabilities: Sequence[str] = ()) -> tuple[ToolSpec, ...]:
        return self.registry.discover(capabilities=capabilities)

    def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        return self.guardrail.execute(self.registry, tool_name, arguments, context)
