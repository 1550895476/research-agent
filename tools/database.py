"""Read-only PostgreSQL query tool with defense-in-depth SQL restrictions."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from hashlib import sha256
from typing import Any

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field

from rag.retriever import SourceType
from tools.registry import (
    StructuredTool,
    ToolContext,
    ToolEvidence,
    ToolResult,
    ToolRisk,
    ToolSpec,
)


class DatabaseQueryInput(BaseModel):
    """Validated arguments for one bounded, read-only SQL statement."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1, max_length=20000)
    parameters: tuple[Any, ...] = ()
    max_rows: int = Field(default=100, ge=1, le=1000)


ConnectionFactory = Callable[[], psycopg.Connection[Any]]


class PostgresQueryTool(StructuredTool):
    """Execute SELECT-like SQL in a read-only transaction with bounded output."""

    def __init__(
        self,
        database_url: str,
        *,
        connect_timeout_seconds: int = 3,
        statement_timeout_ms: int = 10000,
        max_rows: int = 100,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        super().__init__(
            database_tool_spec(),
            DatabaseQueryInput,
        )
        self._database_url = database_url
        self._connect_timeout_seconds = max(1, min(30, connect_timeout_seconds))
        self._statement_timeout_ms = max(100, statement_timeout_ms)
        self._max_rows = min(1000, max(1, max_rows))
        self._connection_factory = connection_factory or self._connect

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        request = DatabaseQueryInput.model_validate(arguments)
        validate_read_only_sql(request.sql)
        row_limit = min(request.max_rows, self._max_rows)
        with self._connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(self._statement_timeout_ms),),
                )
                cursor.execute(request.sql, request.parameters)
                rows = cursor.fetchmany(row_limit + 1) if cursor.description else []
        truncated = len(rows) > row_limit
        normalized_rows = [_json_safe(row) for row in rows[:row_limit]]
        query_hash = sha256(request.sql.encode()).hexdigest()[:24]
        source_uri = f"database://query/{query_hash}"
        evidences = ()
        if normalized_rows:
            evidences = (
                ToolEvidence(
                    evidence_id=f"evidence:database:{query_hash}",
                    source_type=SourceType.DATABASE,
                    source_uri=source_uri,
                    title="Read-only database query result",
                    content=json.dumps(normalized_rows, ensure_ascii=False, default=str),
                    confidence=0.9,
                ),
            )
        return ToolResult(
            tool_name=self.spec.name,
            data={"rows": normalized_rows, "truncated": truncated},
            evidences=evidences,
            metadata={"row_count": len(normalized_rows), "max_rows": row_limit},
        )

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(
            self._database_url,
            connect_timeout=self._connect_timeout_seconds,
            row_factory=dict_row,
        )


def validate_read_only_sql(sql: str) -> None:
    normalized = _strip_sql_comments(sql).strip()
    statements = [statement.strip() for statement in normalized.split(";") if statement.strip()]
    if len(statements) != 1:
        raise ValueError("database tool accepts exactly one SQL statement")
    first_keyword = re.match(r"^[A-Za-z]+", statements[0])
    if first_keyword is None or first_keyword.group(0).upper() not in {
        "SELECT",
        "WITH",
        "EXPLAIN",
        "SHOW",
    }:
        raise ValueError("database tool only permits read-only query statements")
    forbidden = re.compile(
        r"\b(ALTER|CALL|COPY|CREATE|DELETE|DO|DROP|GRANT|INSERT|MERGE|"
        r"REINDEX|REVOKE|TRUNCATE|UPDATE|VACUUM)\b",
        flags=re.IGNORECASE,
    )
    if forbidden.search(statements[0]):
        raise ValueError("database tool rejected a mutating or administrative SQL keyword")
    dangerous_function = re.compile(
        r"\b(dblink|lo_export|lo_import|pg_cancel_backend|pg_ls_dir|pg_read_binary_file|"
        r"pg_read_file|pg_reload_conf|pg_sleep|pg_stat_file|pg_terminate_backend|"
        r"set_config)\s*\(",
        flags=re.IGNORECASE,
    )
    if dangerous_function.search(statements[0]):
        raise ValueError("database tool rejected a privileged or disruptive SQL function")


def _strip_sql_comments(sql: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\r\n]*", " ", without_blocks)


def _json_safe(row: Any) -> Any:
    if isinstance(row, dict):
        return {str(key): value for key, value in row.items()}
    if isinstance(row, tuple | list):
        return list(row)
    return row


def database_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="database.query",
        description=(
            "Execute one parameterized, read-only PostgreSQL SELECT/WITH/EXPLAIN/SHOW "
            "statement and return a bounded result set."
        ),
        input_schema=DatabaseQueryInput.model_json_schema(),
        capabilities=frozenset({"database", "sql", "read"}),
        required_permission="database:read",
        risk=ToolRisk.MEDIUM,
    )
