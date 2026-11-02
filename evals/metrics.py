"""Deterministic, source-grounded quality metrics for completed or partial runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from agents.researcher import ResearchOutput
from agents.verifier import VerificationOutput
from evidence.store import EvidenceRecord
from graph.state import AgentName, AgentState, ResearchTask, TaskStatus


class ResearchQualityMetrics(BaseModel):
    """Run-level rubric metrics; URL validity is structural, not a live HEAD check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_completion_rate: float = Field(ge=0, le=1)
    research_tasks: int = Field(ge=0)
    quality_gate_evaluated_tasks: int = Field(ge=0)
    quality_gate_pass_rate: float | None = Field(default=None, ge=0, le=1)
    claims_total: int = Field(ge=0)
    evidence_supported_claims: int = Field(ge=0)
    claim_support_rate: float | None = Field(default=None, ge=0, le=1)
    citation_references: int = Field(ge=0)
    grounded_citation_references: int = Field(ge=0)
    citation_grounding_rate: float | None = Field(default=None, ge=0, le=1)
    verified_claims: int = Field(ge=0)
    unresolved_conflicts: int = Field(ge=0)
    sources_total: int = Field(ge=0)
    structurally_valid_source_urls: int = Field(ge=0)
    structurally_valid_url_rate: float | None = Field(default=None, ge=0, le=1)
    first_party_sources: int = Field(ge=0)
    independent_domains: int = Field(ge=0)


def evaluate_research_quality(
    state: AgentState | dict[str, Any],
    payloads: Sequence[Any],
    evidences: Sequence[EvidenceRecord],
) -> ResearchQualityMetrics:
    tasks: list[ResearchTask] = list(state.get("tasks", []))
    research_tasks = [
        task for task in tasks if task.assigned_agent == AgentName.RESEARCHER
    ]
    evaluated = [
        task for task in research_tasks if task.quality_complete is not None
    ]
    research_outputs = [
        payload for payload in payloads if isinstance(payload, ResearchOutput)
    ]
    verification_outputs = [
        payload for payload in payloads if isinstance(payload, VerificationOutput)
    ]
    claims = [claim for payload in research_outputs for claim in payload.claims]
    evidence_ids = {record.evidence_id for record in evidences}
    citation_ids = [evidence_id for claim in claims for evidence_id in claim.evidence_ids]
    grounded_ids = [evidence_id for evidence_id in citation_ids if evidence_id in evidence_ids]
    verified_claims = sum(
        claim.verified
        for payload in verification_outputs
        for claim in payload.claims
    )
    conflicts = sum(len(payload.conflicts) for payload in verification_outputs)
    valid_urls = [record for record in evidences if _structurally_valid_url(record.source_url)]
    domains = {
        urlparse(record.source_url).netloc.casefold()
        for record in valid_urls
        if urlparse(record.source_url).netloc
    }
    first_party = sum(_is_first_party(record.source_url, tasks) for record in valid_urls)
    return ResearchQualityMetrics(
        task_completion_rate=_rate(
            sum(task.status == TaskStatus.COMPLETED for task in tasks),
            len(tasks),
        )
        or 0,
        research_tasks=len(research_tasks),
        quality_gate_evaluated_tasks=len(evaluated),
        quality_gate_pass_rate=(
            _rate(sum(task.quality_complete is True for task in evaluated), len(evaluated))
            if evaluated
            else None
        ),
        claims_total=len(claims),
        evidence_supported_claims=sum(bool(claim.evidence_ids) for claim in claims),
        claim_support_rate=(
            _rate(sum(bool(claim.evidence_ids) for claim in claims), len(claims))
            if claims
            else None
        ),
        citation_references=len(citation_ids),
        grounded_citation_references=len(grounded_ids),
        citation_grounding_rate=(
            _rate(len(grounded_ids), len(citation_ids)) if citation_ids else None
        ),
        verified_claims=verified_claims,
        unresolved_conflicts=conflicts,
        sources_total=len(evidences),
        structurally_valid_source_urls=len(valid_urls),
        structurally_valid_url_rate=(
            _rate(len(valid_urls), len(evidences)) if evidences else None
        ),
        first_party_sources=first_party,
        independent_domains=len(domains),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _structurally_valid_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_first_party(source_url: str, tasks: Sequence[ResearchTask]) -> bool:
    parsed = urlparse(source_url)
    domain = parsed.netloc.casefold().removeprefix("www.")
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return True
    for task in tasks:
        for allowed in task.source_domains:
            normalized = allowed.casefold().removeprefix("www.").strip(".")
            if domain == normalized or domain.endswith(f".{normalized}"):
                return True
        normalized_url = source_url.casefold().rstrip("/")
        if any(
            normalized_url == prefix.casefold().rstrip("/")
            or normalized_url.startswith(f"{prefix.casefold().rstrip('/')}/")
            for prefix in task.source_url_prefixes
        ):
            return True
    return False
