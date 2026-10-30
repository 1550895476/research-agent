"""Configurable LangGraph checkpoint backends and resume configuration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row

from core.config import Settings

_ALLOWED_STATE_TYPES = (
    ("graph.state", "AgentName"),
    ("graph.state", "CheckpointReference"),
    ("graph.state", "EvidenceReference"),
    ("graph.state", "HumanRequest"),
    ("graph.state", "ResearchRequirement"),
    ("graph.state", "ResearchTask"),
    ("graph.state", "ResearchType"),
    ("graph.state", "ReplanRequest"),
    ("graph.state", "ResultReference"),
    ("graph.state", "TaskStatus"),
    ("graph.state", "WorkflowError"),
    ("graph.state", "WorkflowStatus"),
    ("graph.state", "WorkerStatusEvent"),
)


def checkpoint_config(thread_id: str, *, namespace: str = "") -> dict[str, Any]:
    """Build the RunnableConfig required by every checkpointed LangGraph invocation."""

    normalized_thread_id = thread_id.strip()
    if not normalized_thread_id:
        raise ValueError("thread_id must not be empty")
    return {
        "configurable": {
            "thread_id": normalized_thread_id,
            "checkpoint_ns": namespace,
        }
    }


class CheckpointManager:
    """Own a shared in-memory saver or scoped PostgreSQL saver connections."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_STATE_TYPES)
        self._memory_saver = InMemorySaver(serde=self._serde)
        self._setup_lock = Lock()
        self._postgres_setup_complete = False

    @property
    def backend(self) -> str:
        return self._settings.checkpoint_backend

    @contextmanager
    def open(self) -> Iterator[BaseCheckpointSaver[Any]]:
        """Yield a live saver for graph compilation and invocation."""

        if self._settings.checkpoint_backend == "memory":
            yield self._memory_saver
            return
        with psycopg.connect(
            self._settings.database_url,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as connection:
            saver = PostgresSaver(connection, serde=self._serde)
            self._setup_postgres(saver)
            yield saver

    def has_checkpoint(self, thread_id: str, *, namespace: str = "") -> bool:
        """Return whether a thread has at least one persisted checkpoint."""

        config = checkpoint_config(thread_id, namespace=namespace)
        with self.open() as saver:
            return saver.get_tuple(config) is not None

    def load_state(self, thread_id: str, *, namespace: str = "") -> dict[str, Any] | None:
        """Load the latest checkpoint channel values for progress/status APIs."""

        config = checkpoint_config(thread_id, namespace=namespace)
        with self.open() as saver:
            checkpoint_tuple = saver.get_tuple(config)
        if checkpoint_tuple is None:
            return None
        return dict(checkpoint_tuple.checkpoint.get("channel_values", {}))

    def _setup_postgres(self, saver: PostgresSaver) -> None:
        if not self._settings.checkpoint_auto_setup or self._postgres_setup_complete:
            return
        with self._setup_lock:
            if self._postgres_setup_complete:
                return
            saver.setup()
            self._postgres_setup_complete = True
