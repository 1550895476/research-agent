"""Environment-backed application configuration."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    small_model: str | None = None
    large_model: str | None = None
    structured_output_method: Literal["json_schema", "function_calling", "json_mode"] = (
        "json_schema"
    )
    model_timeout_seconds: float = Field(default=60.0, gt=0)
    model_max_retries: int = Field(default=2, ge=0, le=10)
    max_agent_context_chars: int = Field(default=16000, ge=1000)
    context_soft_watermark: float = Field(default=0.70, gt=0, lt=1)
    context_hard_watermark: float = Field(default=0.85, gt=0, lt=1)
    context_tool_result_max_chars: int = Field(default=2400, ge=200)
    context_summary_max_chars: int = Field(default=1200, ge=200)
    context_compression_max_events: int = Field(default=1000, ge=10, le=100000)
    dynamic_model_routing: bool = True
    model_router_complexity_chars: int = Field(default=4000, ge=500, le=100000)

    max_concurrent_workers: int = Field(default=5, ge=1, le=100)
    max_total_tasks: int = Field(default=12, ge=3, le=100)
    max_total_searches: int = Field(default=100, ge=1, le=10000)
    max_total_input_tokens: int = Field(default=500000, ge=1000, le=100000000)
    max_total_output_tokens: int = Field(default=200000, ge=1000, le=100000000)
    max_total_cost_usd: float = Field(default=0, ge=0)
    max_run_seconds: float = Field(default=1200, gt=0, le=604800)
    max_conflict_tasks: int = Field(default=2, ge=0, le=20)
    small_input_cost_per_million: float = Field(default=0, ge=0)
    small_output_cost_per_million: float = Field(default=0, ge=0)
    large_input_cost_per_million: float = Field(default=0, ge=0)
    large_output_cost_per_million: float = Field(default=0, ge=0)
    task_max_searches: int = Field(default=10, ge=0, le=100)
    task_max_sources: int = Field(default=12, ge=1, le=200)
    task_max_input_tokens: int = Field(default=60000, ge=100, le=10000000)
    task_max_output_tokens: int = Field(default=20000, ge=100, le=10000000)
    task_max_seconds: float = Field(default=300, gt=0, le=86400)
    task_max_retries: int = Field(default=2, ge=0, le=20)
    plan_approval_mode: Literal["off", "auto", "always"] = "auto"
    plan_approval_task_threshold: int = Field(default=8, ge=3, le=100)
    plan_approval_token_threshold: int = Field(default=200000, ge=1000)
    plan_approval_cost_threshold_usd: float = Field(default=5, ge=0)

    tavily_api_key: SecretStr | None = None
    tools_enabled: bool = False
    require_external_evidence: bool = True
    tool_permissions: str = "search:execute,knowledge_base:read,database:read"
    search_max_calls_per_run: int = Field(default=100, ge=1, le=1000)
    search_max_calls_per_task: int = Field(default=10, ge=1, le=100)
    research_quality_gate_enabled: bool = True
    research_initial_searches: int = Field(default=5, ge=1, le=10)
    research_followup_batch_size: int = Field(default=2, ge=1, le=5)
    research_max_searches_per_task: int = Field(default=10, ge=1, le=100)
    research_completion_score: int = Field(default=80, ge=0, le=100)
    research_min_unique_sources: int = Field(default=5, ge=1, le=100)
    research_min_independent_domains: int = Field(default=3, ge=1, le=50)
    research_max_tool_rounds: int = Field(default=3, ge=1, le=10)
    database_tool_enabled: bool = False
    database_connect_timeout_seconds: int = Field(default=3, ge=1, le=30)
    database_statement_timeout_ms: int = Field(default=10000, ge=100, le=120000)
    database_max_rows: int = Field(default=100, ge=1, le=1000)
    recovery_enabled: bool = True
    recovery_max_tool_retries: int = Field(default=2, ge=0, le=10)
    recovery_max_task_replans: int = Field(default=1, ge=0, le=5)
    recovery_backoff_seconds: float = Field(default=0.25, ge=0, le=10)
    saga_backend: Literal["memory", "postgres"] = "memory"
    saga_auto_setup: bool = True
    database_url: str = Field(
        default="postgresql://research_agent:research_agent@localhost:5432/research_agent"
    )
    checkpoint_backend: Literal["memory", "postgres"] = "memory"
    checkpoint_auto_setup: bool = True
    blackboard_backend: Literal["memory", "postgres"] = "memory"
    long_term_memory_backend: Literal["memory", "postgres"] = "memory"
    episodic_memory_backend: Literal["memory", "postgres"] = "memory"
    memory_compression_chars: int = Field(default=8000, ge=500)
    memory_min_value_score: float = Field(default=0.2, ge=0, le=1)
    memory_max_records_per_user: int = Field(default=1000, ge=10)
    embedding_dimension: int = Field(default=1024, ge=1)
    rag_enabled: bool = False
    rag_backend: Literal["memory", "postgres"] = "postgres"
    bge_m3_model: str = "BAAI/bge-m3"
    bge_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    bge_device: str = "cpu"
    bge_use_fp16: bool = False
    rag_candidate_k: int = Field(default=30, ge=1, le=200)
    rag_top_k: int = Field(default=8, ge=1, le=50)
    rag_rrf_k: int = Field(default=60, ge=1)
    rag_context_chars: int = Field(default=12000, ge=1000)

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "research-agent-local"

    @model_validator(mode="after")
    def validate_research_budgets(self) -> Self:
        if self.context_soft_watermark >= self.context_hard_watermark:
            raise ValueError(
                "CONTEXT_SOFT_WATERMARK must be lower than CONTEXT_HARD_WATERMARK"
            )
        if self.research_max_searches_per_task < self.research_initial_searches:
            raise ValueError(
                "RESEARCH_MAX_SEARCHES_PER_TASK must be >= RESEARCH_INITIAL_SEARCHES"
            )
        if self.search_max_calls_per_task < self.research_max_searches_per_task:
            raise ValueError(
                "SEARCH_MAX_CALLS_PER_TASK must be >= RESEARCH_MAX_SEARCHES_PER_TASK"
            )
        if self.research_min_independent_domains > self.research_min_unique_sources:
            raise ValueError(
                "RESEARCH_MIN_INDEPENDENT_DOMAINS must be <= "
                "RESEARCH_MIN_UNIQUE_SOURCES"
            )
        if self.max_concurrent_workers > self.max_total_tasks:
            raise ValueError("MAX_CONCURRENT_WORKERS must be <= MAX_TOTAL_TASKS")
        if self.task_max_searches > self.max_total_searches:
            raise ValueError("TASK_MAX_SEARCHES must be <= MAX_TOTAL_SEARCHES")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
