"""Watermark-driven context compression for Agent prompts and follow-up turns."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ContextItem(BaseModel):
    """One independently compressible piece of prompt context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(min_length=1)
    kind: Literal[
        "active",
        "constraint",
        "regenerable",
        "tool_output",
        "old_turn",
        "completed_step",
        "validation",
        "reference",
    ] = "completed_step"
    content: str
    summary: str = ""
    reference: str | None = None
    version: str | None = None
    replay_hint: str | None = None
    segment: str = "default"
    required: bool = False
    priority: int = Field(default=50, ge=0, le=100)


class ContextSnapshot(BaseModel):
    """Minimum state that must survive a level-five context rebuild."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: str
    constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    pending_items: tuple[str, ...] = ()
    exact_references: tuple[str, ...] = ()
    validation_results: tuple[str, ...] = ()
    next_action: str = "Continue the current task from this snapshot."
    blocked_by: tuple[str, ...] = ()


class ContextCompressionRecord(BaseModel):
    """Checkpoint-safe audit record for one compression decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    context_id: str
    trigger: Literal["soft", "hard"]
    applied_level: int = Field(ge=1, le=5)
    boundary: str
    original_chars: int = Field(ge=0)
    compressed_chars: int = Field(ge=0)
    soft_limit_chars: int = Field(ge=1)
    hard_limit_chars: int = Field(ge=1)
    saved_chars: int = Field(ge=0)
    compression_ratio: float = Field(ge=0, le=1)
    references_preserved: int = Field(ge=0)
    prefix_changed: bool = True
    stable_prefix_preserved: bool = True
    estimated_break_even_requests: int = Field(ge=1)
    checkpoint_required: bool = False
    original_hash: str
    compressed_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class CompressionResult:
    """Rendered context and the optional audit event produced for it."""

    text: str
    record: ContextCompressionRecord | None


class ContextCompressor:
    """Apply a five-level compression ladder at soft and hard watermarks.

    Prompt assembly is treated as a natural task boundary. Crossing the soft watermark
    starts the ladder there. Crossing the hard watermark marks the event as requiring a
    durable checkpoint; the graph already checkpoints the dispatch state before workers run.
    """

    def __init__(
        self,
        *,
        max_chars: int = 16000,
        soft_watermark: float = 0.70,
        hard_watermark: float = 0.85,
        tool_result_max_chars: int = 2400,
        summary_max_chars: int = 1200,
        max_events: int = 1000,
    ) -> None:
        if max_chars < 1000:
            raise ValueError("max_chars must be at least 1000")
        if not 0 < soft_watermark < hard_watermark < 1:
            raise ValueError("watermarks must satisfy 0 < soft < hard < 1")
        if tool_result_max_chars < 200:
            raise ValueError("tool_result_max_chars must be at least 200")
        if summary_max_chars < 200:
            raise ValueError("summary_max_chars must be at least 200")
        self.max_chars = max_chars
        self.soft_watermark = soft_watermark
        self.hard_watermark = hard_watermark
        self.tool_result_max_chars = tool_result_max_chars
        self.summary_max_chars = summary_max_chars
        self.max_events = max_events
        self._events: list[ContextCompressionRecord] = []
        self._lock = RLock()

    def compress_payloads(
        self,
        payloads: Sequence[Any],
        *,
        context_id: str,
        snapshot: ContextSnapshot | None = None,
        max_chars: int | None = None,
    ) -> CompressionResult:
        """Render model/tool payloads fairly and compress only when a watermark is crossed."""

        items = context_items_from_payloads(payloads)
        return self.compress(
            items,
            context_id=context_id,
            snapshot=snapshot,
            max_chars=max_chars,
        )

    def compress(
        self,
        items: Sequence[ContextItem],
        *,
        context_id: str,
        snapshot: ContextSnapshot | None = None,
        max_chars: int | None = None,
        boundary: str = "prompt_assembly",
    ) -> CompressionResult:
        """Apply layers in order and stop as soon as the soft target is reached."""

        limit = max_chars or self.max_chars
        if limit < 1000:
            raise ValueError("max_chars must be at least 1000")
        normalized = tuple(_normalize_item(item) for item in items)
        merged_snapshot = _merge_snapshot(snapshot, _snapshot_from_items(normalized))
        preserved_references = _collect_references(normalized, merged_snapshot)
        original = _render_items(normalized)
        soft_limit = max(1, int(limit * self.soft_watermark))
        hard_limit = max(soft_limit + 1, int(limit * self.hard_watermark))
        if len(original) <= soft_limit:
            return CompressionResult(text=original, record=None)

        trigger: Literal["soft", "hard"] = (
            "hard" if len(original) >= hard_limit else "soft"
        )
        working = list(normalized)
        applied_level = 0
        rendered = original

        for level, transform in (
            (1, self._discard_regenerable),
            (2, self._truncate_tool_outputs),
            (3, self._coarsen_completed_history),
        ):
            working = transform(working)
            candidate = _render_items(working)
            if candidate != rendered:
                applied_level = level
                rendered = candidate
            if len(rendered) <= soft_limit and applied_level:
                break

        if len(rendered) > soft_limit:
            working = self._summarize_segments(working)
            rendered = _render_items(working)
            applied_level = 4

        if len(rendered) > soft_limit:
            rendered = _render_snapshot(merged_snapshot, soft_limit)
            applied_level = 5

        # A context made only of required active content may not shrink in levels 1-4.
        if not applied_level:
            rendered = _render_snapshot(merged_snapshot, soft_limit)
            applied_level = 5

        saved_chars = max(0, len(original) - len(rendered))

        record = ContextCompressionRecord(
            context_id=context_id,
            trigger=trigger,
            applied_level=applied_level,
            boundary=boundary,
            original_chars=len(original),
            compressed_chars=len(rendered),
            soft_limit_chars=soft_limit,
            hard_limit_chars=hard_limit,
            saved_chars=saved_chars,
            compression_ratio=(
                round(len(rendered) / len(original), 6) if original else 0
            ),
            references_preserved=len(preserved_references),
            estimated_break_even_requests=max(
                1,
                ceil(len(rendered) / max(1, saved_chars)),
            ),
            checkpoint_required=trigger == "hard",
            original_hash=_content_hash(original),
            compressed_hash=_content_hash(rendered),
        )
        self._record(record)
        return CompressionResult(text=rendered, record=record)

    def snapshot(
        self,
        *,
        context_prefix: str | None = None,
    ) -> tuple[ContextCompressionRecord, ...]:
        """Return immutable compression audit data, optionally scoped to one run."""

        with self._lock:
            events = tuple(self._events)
        if context_prefix is None:
            return events
        return tuple(event for event in events if event.context_id.startswith(context_prefix))

    def _record(self, record: ContextCompressionRecord) -> None:
        with self._lock:
            self._events.append(record)
            if len(self._events) > self.max_events:
                del self._events[: len(self._events) - self.max_events]

    def _discard_regenerable(self, items: list[ContextItem]) -> list[ContextItem]:
        compressed: list[ContextItem] = []
        for item in items:
            if item.kind != "regenerable" or not item.reference:
                compressed.append(item)
                continue
            summary = item.summary or _extract_summary(item.content, self.summary_max_chars)
            marker = [summary, f"重新读取：{item.reference}"]
            if item.version:
                marker.append(f"版本/内容哈希：{item.version}")
            else:
                marker.append(f"内容哈希：{_content_hash(item.content)}")
            if item.replay_hint:
                marker.append(f"重放参数：{item.replay_hint}")
            compressed.append(item.model_copy(update={"content": "\n".join(marker)}))
        return compressed

    def _truncate_tool_outputs(self, items: list[ContextItem]) -> list[ContextItem]:
        compressed: list[ContextItem] = []
        for item in items:
            if item.kind != "tool_output" or len(item.content) <= self.tool_result_max_chars:
                compressed.append(item)
                continue
            compressed.append(
                item.model_copy(
                    update={
                        "content": _truncate_tool_output(
                            item.content,
                            self.tool_result_max_chars,
                            item.reference,
                            item.replay_hint,
                        )
                    }
                )
            )
        return compressed

    def _coarsen_completed_history(self, items: list[ContextItem]) -> list[ContextItem]:
        compressed: list[ContextItem] = []
        for item in items:
            if item.kind not in {"old_turn", "completed_step"}:
                compressed.append(item)
                continue
            summary = item.summary or _extract_summary(item.content, self.summary_max_chars)
            references = f"\n精确引用：{item.reference}" if item.reference else ""
            compressed.append(item.model_copy(update={"content": summary + references}))
        return compressed

    def _summarize_segments(self, items: list[ContextItem]) -> list[ContextItem]:
        grouped: dict[str, list[ContextItem]] = defaultdict(list)
        for item in items:
            grouped[item.segment].append(item)
        segments: list[ContextItem] = []
        for segment, members in grouped.items():
            lines = []
            references: list[str] = []
            for member in sorted(members, key=lambda value: value.priority, reverse=True):
                summary = (
                    member.content
                    if member.required and member.kind == "active"
                    else member.summary or _extract_summary(member.content, 360)
                )
                lines.append(f"- {member.item_id}: {summary}")
                if member.reference:
                    references.append(member.reference)
            if references:
                lines.append("精确引用：" + ", ".join(dict.fromkeys(references)))
            segments.append(
                ContextItem(
                    item_id=f"segment:{segment}",
                    kind="completed_step",
                    content=(
                        "\n".join(lines)
                        if any(
                            member.required and member.kind == "active"
                            for member in members
                        )
                        else _fit_text("\n".join(lines), self.summary_max_chars)
                    ),
                    summary=lines[0] if lines else segment,
                    segment=segment,
                    required=any(member.required for member in members),
                    priority=max(member.priority for member in members),
                )
            )
        return segments


def context_items_from_payloads(payloads: Sequence[Any]) -> tuple[ContextItem, ...]:
    """Convert heterogeneous Agent dependencies into independently compressible items."""

    items: list[ContextItem] = []
    for index, payload in enumerate(payloads):
        segment = f"payload-{index + 1}"
        tool_name = str(getattr(payload, "tool_name", "")).strip()
        evidences = tuple(getattr(payload, "evidences", ()) or ())
        if tool_name and evidences:
            data = getattr(payload, "data", {})
            query = str(data.get("query", "")) if isinstance(data, dict) else ""
            items.append(
                ContextItem(
                    item_id=f"{segment}:tool",
                    kind="tool_output",
                    content=_serialize_payload(data),
                    summary=f"{tool_name} 返回 {len(evidences)} 条证据。",
                    replay_hint=(
                        f"tool={tool_name}; query={query}"
                        if query
                        else f"tool={tool_name}"
                    ),
                    segment=segment,
                    priority=65,
                )
            )
            for evidence_index, evidence in enumerate(evidences, start=1):
                evidence_id = str(getattr(evidence, "evidence_id", evidence_index))
                source_uri = str(getattr(evidence, "source_uri", "")).strip() or None
                title = str(getattr(evidence, "title", "")).strip()
                content = str(getattr(evidence, "content", ""))
                items.append(
                    ContextItem(
                        item_id=f"{segment}:evidence:{evidence_id}",
                        kind="regenerable" if source_uri else "tool_output",
                        content=content,
                        summary=(
                            f"Evidence {evidence_id}; title={title or 'n/a'}; "
                            f"source={source_uri or 'inline'}"
                        ),
                        reference=source_uri,
                        version=evidence_id,
                        replay_hint=(
                            f"tool={tool_name}; query={query}" if query else f"tool={tool_name}"
                        ),
                        segment=segment,
                        priority=80,
                    )
                )
            continue

        serialized = _serialize_payload(payload)
        summary = _payload_summary(payload, serialized)
        references = _payload_references(payload)
        payload_type = type(payload).__name__
        items.append(
            ContextItem(
                item_id=f"{segment}:{payload_type}",
                kind=(
                    "old_turn"
                    if payload_type == "ReportDraft"
                    else "completed_step"
                ),
                content=serialized,
                summary=summary,
                reference=references[0] if references else None,
                version=_content_hash(serialized),
                segment=segment,
                priority=70,
            )
        )
        for reference_index, reference in enumerate(references[1:], start=2):
            items.append(
                ContextItem(
                    item_id=f"{segment}:reference:{reference_index}",
                    kind="reference",
                    content=reference,
                    summary=reference,
                    reference=reference,
                    segment=segment,
                    priority=85,
                )
            )
    return tuple(items)


def _normalize_item(item: ContextItem) -> ContextItem:
    return item.model_copy(
        update={
            "content": item.content.strip(),
            "summary": " ".join(item.summary.split()),
            "segment": item.segment.strip() or "default",
        }
    )


def _render_items(items: Iterable[ContextItem]) -> str:
    grouped: dict[str, list[ContextItem]] = defaultdict(list)
    for item in items:
        grouped[item.segment].append(item)
    sections: list[str] = []
    for segment, members in grouped.items():
        lines = [f"## Context segment: {segment}"]
        for item in members:
            lines.extend((f"### {item.item_id} [{item.kind}]", item.content))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _snapshot_from_items(items: Sequence[ContextItem]) -> ContextSnapshot:
    active = [item for item in items if item.kind == "active"]
    constraints = [item.content for item in items if item.kind == "constraint"]
    completed = [
        item.summary or _extract_summary(item.content, 500)
        for item in items
        if item.kind in {"old_turn", "completed_step"}
    ]
    pending = [
        item.summary or _extract_summary(item.content, 500)
        for item in items
        if item.required and item.kind not in {"constraint", "reference"}
    ]
    validation = [item.content for item in items if item.kind == "validation"]
    references = _collect_references(items, None)
    goal = active[-1].content if active else "Continue the current Agent task."
    return ContextSnapshot(
        goal=goal,
        constraints=tuple(constraints),
        decisions=tuple(completed),
        pending_items=tuple(pending),
        exact_references=tuple(references),
        validation_results=tuple(validation),
        next_action="Use this state snapshot to complete the current Agent task.",
    )


def _merge_snapshot(
    explicit: ContextSnapshot | None,
    derived: ContextSnapshot,
) -> ContextSnapshot:
    if explicit is None:
        return derived

    def unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value for group in groups for value in group if value))

    return ContextSnapshot(
        goal=explicit.goal or derived.goal,
        constraints=unique(explicit.constraints, derived.constraints),
        decisions=unique(explicit.decisions, derived.decisions),
        pending_items=unique(explicit.pending_items, derived.pending_items),
        exact_references=unique(
            explicit.exact_references,
            derived.exact_references,
        ),
        validation_results=unique(
            explicit.validation_results,
            derived.validation_results,
        ),
        next_action=explicit.next_action or derived.next_action,
        blocked_by=unique(explicit.blocked_by, derived.blocked_by),
    )


def _render_snapshot(snapshot: ContextSnapshot, max_chars: int) -> str:
    sections = (
        ("当前目标", (snapshot.goal,)),
        ("约束条件", snapshot.constraints),
        ("关键决策及理由", snapshot.decisions),
        ("未完成事项", snapshot.pending_items),
        ("精确引用", snapshot.exact_references),
        ("验证结果", snapshot.validation_results),
        ("阻塞项", snapshot.blocked_by),
        ("下一步行动", (snapshot.next_action,)),
    )
    header_budget = 32 + sum(len(title) + 8 for title, _ in sections)
    content_budget = max(200, max_chars - header_budget)
    populated = sum(bool(values) for _, values in sections) or 1
    per_section = max(80, content_budget // populated)
    rendered: list[str] = ["# 上下文状态快照"]
    for title, values in sections:
        if not values:
            continue
        body = "\n".join(f"- {value}" for value in values if value)
        rendered.extend((f"\n## {title}", _fit_text(body, per_section)))
    return _fit_text("\n".join(rendered), max_chars)


def _truncate_tool_output(
    content: str,
    max_chars: int,
    reference: str | None,
    replay_hint: str | None,
) -> str:
    lines = content.splitlines()
    error_pattern = re.compile(r"error|failed|failure|exception|traceback|错误|失败", re.I)
    errors = [line.strip() for line in lines if error_pattern.search(line)][:8]
    metadata = [f"原始行数：{len(lines)}"]
    if errors:
        metadata.append("关键错误：\n" + "\n".join(errors))
    if reference:
        metadata.append(f"完整输出：{reference}")
    if replay_hint:
        metadata.append(f"重放参数：{replay_hint}")
    reserved = len("\n".join(metadata)) + 40
    sample = _fit_text(content, max(80, max_chars - reserved))
    return _fit_text("\n".join((*metadata, "关键片段：", sample)), max_chars)


def _payload_summary(payload: Any, serialized: str) -> str:
    for attribute in ("summary", "executive_summary"):
        value = getattr(payload, attribute, None)
        if value:
            return _fit_text(str(value), 600)
    if isinstance(payload, dict):
        for key in ("summary", "result", "error", "message"):
            if payload.get(key):
                return _fit_text(str(payload[key]), 600)
    return _extract_summary(serialized, 600)


def _payload_references(payload: Any) -> list[str]:
    references: list[str] = []
    for collection_name in ("sources", "citations", "evidences"):
        for item in tuple(getattr(payload, collection_name, ()) or ()):
            for attribute in ("source_uri", "payload_ref", "evidence_id", "result_id"):
                value = str(getattr(item, attribute, "")).strip()
                if value:
                    references.append(value)
    return list(dict.fromkeys(references))


def _serialize_payload(payload: Any) -> str:
    if isinstance(payload, BaseModel):
        return payload.model_dump_json(indent=2)
    if isinstance(payload, (dict, list, tuple)):
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return str(payload)


def _extract_summary(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    sentences = re.split(r"(?<=[。！？.!?])\s+", normalized)
    selected: list[str] = []
    for sentence in sentences:
        if len(" ".join((*selected, sentence))) > max_chars:
            break
        selected.append(sentence)
    return " ".join(selected) or _fit_text(normalized, max_chars)


def _collect_references(
    items: Sequence[ContextItem],
    snapshot: ContextSnapshot | None,
) -> list[str]:
    references = [item.reference for item in items if item.reference]
    if snapshot is not None:
        references.extend(snapshot.exact_references)
    return list(dict.fromkeys(reference for reference in references if reference))


def _fit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n…[context compressed]…\n"
    available = max(0, max_chars - len(marker))
    head = int(available * 0.65)
    tail = available - head
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
