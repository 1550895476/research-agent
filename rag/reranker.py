"""BGE cross-encoder reranking with a deterministic test fallback."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class RerankCandidate(BaseModel):
    """Minimal candidate data required by a cross-encoder reranker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    text: str
    fusion_score: float = Field(ge=0)


class RerankScore(BaseModel):
    """Cross-encoder relevance score keyed by chunk identifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    score: float


class Reranker(Protocol):
    """Reranking contract consumed by RAGPipeline."""

    def score(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankScore]: ...


class FusionScoreReranker:
    """Keep RRF order when the optional cross-encoder is disabled."""

    def score(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankScore]:
        del query
        return [
            RerankScore(chunk_id=candidate.chunk_id, score=candidate.fusion_score)
            for candidate in candidates
        ]


class BGEReranker:
    """Lazy wrapper around FlagEmbedding.FlagReranker."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str = "cpu",
        use_fp16: bool = False,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16
        self._model: Any | None = None
        self._lock = RLock()

    def score(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[RerankScore]:
        if not candidates:
            return []
        model = self._load_model()
        pairs = [[query, candidate.text] for candidate in candidates]
        with self._lock:
            raw_scores = model.compute_score(pairs, normalize=True)
        if isinstance(raw_scores, int | float):
            raw_scores = [raw_scores]
        return [
            RerankScore(chunk_id=candidate.chunk_id, score=float(score))
            for candidate, score in zip(candidates, raw_scores, strict=True)
        ]

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RuntimeError(
                    "BGE-Reranker requires optional dependencies; "
                    "install with: pip install -e '.[rag]'"
                ) from exc
            self._model = FlagReranker(
                self._model_name,
                devices=self._device,
                use_fp16=self._use_fp16,
            )
            return self._model
