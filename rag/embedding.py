"""BGE-M3 dense and lexical sparse embedding adapter with lazy model loading."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class HybridEmbedding(BaseModel):
    """Dense vector and BGE-M3 lexical weights for one text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dense: tuple[float, ...] = Field(min_length=1)
    sparse: dict[str, float] = Field(default_factory=dict)


class HybridEmbeddingModel(Protocol):
    """Encoder contract used by indexing and retrieval pipelines."""

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]: ...


class BGEM3EmbeddingModel:
    """Lazy wrapper around FlagEmbedding.BGEM3FlagModel."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str = "cpu",
        use_fp16: bool = False,
        expected_dimension: int = 1024,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16
        self._expected_dimension = expected_dimension
        self._model: Any | None = None
        self._lock = RLock()

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        normalized = [" ".join(text.split()) for text in texts]
        if not normalized or any(not text for text in normalized):
            raise ValueError("texts must contain at least one non-empty string")
        model = self._load_model()
        with self._lock:
            encoded = model.encode(
                normalized,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        dense_vectors = encoded["dense_vecs"]
        sparse_vectors = encoded.get("lexical_weights") or [{} for _ in normalized]
        results: list[HybridEmbedding] = []
        for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True):
            dense_values = dense.tolist() if hasattr(dense, "tolist") else list(dense)
            if len(dense_values) != self._expected_dimension:
                raise ValueError(
                    f"BGE-M3 returned dimension {len(dense_values)}; "
                    f"expected {self._expected_dimension}"
                )
            results.append(
                HybridEmbedding(
                    dense=tuple(float(value) for value in dense_values),
                    sparse={str(token): float(weight) for token, weight in sparse.items()},
                )
            )
        return results

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise RuntimeError(
                    "BGE-M3 requires optional dependencies; install with: pip install -e '.[rag]'"
                ) from exc
            self._model = BGEM3FlagModel(
                self._model_name,
                devices=self._device,
                use_fp16=self._use_fp16,
            )
            return self._model

