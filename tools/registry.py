"""MCP-compatible tool contracts, lazy registry, and stdio client provider."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from threading import RLock
from typing import Any, Protocol, TypeVar

from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field

from rag.retriever import SourceType


class ToolRisk(StrEnum):
    """Static risk level advertised during tool discovery."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolSpec(BaseModel):
    """Compact MCP-style tool metadata used for discovery and model routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    capabilities: frozenset[str] = frozenset()
    required_permission: str | None = None
    risk: ToolRisk = ToolRisk.LOW
    provider: str = "local"


class ToolContext(BaseModel):
    """Execution identity and authorization data passed outside model-controlled arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    permissions: frozenset[str] = frozenset()
    approvals: frozenset[str] = frozenset()


class ToolEvidence(BaseModel):
    """Source-backed evidence returned by a tool without placing it in graph state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(min_length=1)
    source_type: SourceType
    source_uri: str = Field(min_length=1)
    title: str = ""
    content: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)


class ToolResult(BaseModel):
    """Normalized result shape for local and remote MCP tools."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    tool_name: str
    data: Any = None
    evidences: tuple[ToolEvidence, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_error: bool = False


class Tool(Protocol):
    """Executable tool contract stored in ToolRegistry."""

    @property
    def spec(self) -> ToolSpec: ...

    def validate_input(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult: ...


InputModel = TypeVar("InputModel", bound=BaseModel)


class StructuredTool:
    """Base class that validates local tool arguments through a Pydantic schema."""

    def __init__(self, spec: ToolSpec, input_model: type[InputModel]) -> None:
        self._spec = spec
        self._input_model = input_model

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate_input(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = self._input_model.model_validate(arguments)
        return validated.model_dump(mode="python")

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        raise NotImplementedError


ToolLoader = Callable[[], Tool]


class ToolRegistry:
    """Thread-safe registry supporting metadata discovery and on-demand tool loading."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._tools: dict[str, Tool] = {}
        self._loaders: dict[str, ToolLoader] = {}
        self._lock = RLock()

    def register(self, tool: Tool) -> None:
        """Register an already constructed tool."""

        with self._lock:
            self._ensure_available(tool.spec.name)
            self._specs[tool.spec.name] = tool.spec
            self._tools[tool.spec.name] = tool

    def register_lazy(self, spec: ToolSpec, loader: ToolLoader) -> None:
        """Expose metadata now and defer heavyweight construction until first use."""

        with self._lock:
            self._ensure_available(spec.name)
            self._specs[spec.name] = spec
            self._loaders[spec.name] = loader

    def register_mcp_provider(self, provider: MCPToolProvider) -> tuple[str, ...]:
        """Discover remote MCP metadata and register lazy proxy tools."""

        names = []
        for spec in provider.discover_tools():
            self.register_lazy(
                spec,
                lambda spec=spec: MCPProxyTool(provider, spec),
            )
            names.append(spec.name)
        return tuple(names)

    def discover(
        self,
        *,
        capabilities: Sequence[str] = (),
    ) -> tuple[ToolSpec, ...]:
        """Return matching metadata without loading tool implementations."""

        required = set(capabilities)
        with self._lock:
            specs = tuple(self._specs.values())
        return tuple(
            sorted(
                (spec for spec in specs if required.issubset(spec.capabilities)),
                key=lambda spec: spec.name,
            )
        )

    def get(self, name: str) -> Tool:
        """Resolve a tool, invoking its loader exactly when first requested."""

        with self._lock:
            tool = self._tools.get(name)
            loader = self._loaders.get(name)
        if tool is not None:
            return tool
        if loader is None:
            raise KeyError(f"unknown tool: {name}")
        loaded = loader()
        if loaded.spec.name != name:
            raise ValueError(
                f"tool loader for {name!r} returned {loaded.spec.name!r}"
            )
        with self._lock:
            existing = self._tools.setdefault(name, loaded)
            self._loaders.pop(name, None)
            return existing

    def contains(self, name: str) -> bool:
        with self._lock:
            return name in self._specs

    def _ensure_available(self, name: str) -> None:
        if name in self._specs:
            raise ValueError(f"tool already registered: {name}")


class MCPToolProvider(Protocol):
    """Discovery and invocation boundary implemented by MCP transports."""

    def discover_tools(self) -> tuple[ToolSpec, ...]: ...

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult: ...


class MCPProxyTool:
    """Lazy local proxy for one remotely discovered MCP tool."""

    def __init__(self, provider: MCPToolProvider, spec: ToolSpec) -> None:
        self._provider = provider
        self._spec = spec
        self._validator = Draft202012Validator(spec.input_schema)

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate_input(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._validator.validate(arguments)
        return dict(arguments)

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return self._provider.call_tool(self.spec.name, arguments, context)


class MCPStdioServerConfig(BaseModel):
    """Command configuration for an MCP server launched over stdio."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str = Field(min_length=1)
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    name_prefix: str = ""
    required_permission: str | None = None
    default_capabilities: frozenset[str] = frozenset()
    capability_overrides: dict[str, frozenset[str]] = Field(default_factory=dict)
    risk: ToolRisk = ToolRisk.LOW


class MCPStdioToolProvider:
    """Official MCP ClientSession adapter using a fresh stdio session per operation."""

    def __init__(self, config: MCPStdioServerConfig) -> None:
        self._config = config

    def discover_tools(self) -> tuple[ToolSpec, ...]:
        return _run_async(self._discover)

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        del context
        remote_name = name.removeprefix(self._config.name_prefix)
        return _run_async(lambda: self._call(remote_name, name, arguments))

    async def _discover(self) -> tuple[ToolSpec, ...]:
        async with self._session() as session:
            response = await session.list_tools()
        return tuple(
            ToolSpec(
                name=f"{self._config.name_prefix}{tool.name}",
                description=tool.description or f"MCP tool {tool.name}",
                input_schema=dict(tool.inputSchema),
                capabilities=self._config.capability_overrides.get(
                    tool.name,
                    self._config.default_capabilities,
                ),
                required_permission=self._config.required_permission,
                risk=self._config.risk,
                provider="mcp:stdio",
            )
            for tool in response.tools
        )

    async def _call(
        self,
        remote_name: str,
        registered_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        async with self._session() as session:
            response = await session.call_tool(remote_name, arguments)
        content = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else str(item)
            for item in response.content
        ]
        structured = getattr(response, "structuredContent", None)
        if isinstance(structured, dict):
            try:
                normalized = ToolResult.model_validate(structured)
            except ValueError:
                pass
            else:
                return normalized.model_copy(
                    update={
                        "tool_name": registered_name,
                        "metadata": {
                            **normalized.metadata,
                            "provider": "mcp:stdio",
                        },
                    }
                )
        return ToolResult(
            tool_name=registered_name,
            data={"content": content, "structured_content": structured},
            is_error=bool(response.isError),
            metadata={"provider": "mcp:stdio"},
        )

    def _session(self) -> _MCPSessionContext:
        parameters = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args),
            env=self._config.env,
        )
        return _MCPSessionContext(parameters)


class _MCPSessionContext:
    def __init__(self, parameters: StdioServerParameters) -> None:
        self._parameters = parameters
        self._transport: Any = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> ClientSession:
        self._transport = stdio_client(self._parameters)
        read, write = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        session = await self._session.__aenter__()
        await session.initialize()
        return session

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._session is not None:
            await self._session.__aexit__(exc_type, exc, traceback)
        if self._transport is not None:
            await self._transport.__aexit__(exc_type, exc, traceback)


AsyncResult = TypeVar("AsyncResult")


def _run_async(
    factory: Callable[[], Coroutine[Any, Any, AsyncResult]],
) -> AsyncResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-sync") as executor:
        return executor.submit(lambda: asyncio.run(factory())).result()
