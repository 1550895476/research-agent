"""Query rewriting for retrieval-oriented search expressions."""

from __future__ import annotations

import re
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.model import StructuredModel


class RewrittenQuery(BaseModel):
    """Normalized primary query plus bounded expansion queries and keywords."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_query: str = Field(min_length=1)
    rewritten_query: str = Field(min_length=1)
    search_queries: tuple[str, ...] = Field(min_length=1, max_length=6)
    keywords: tuple[str, ...] = Field(default=(), max_length=20)


class QueryRewriter(Protocol):
    """Storage-independent query rewrite contract."""

    def rewrite(self, query: str) -> RewrittenQuery: ...


class HeuristicQueryRewriter:
    """Deterministic fallback that works without an LLM."""

    def rewrite(self, query: str) -> RewrittenQuery:
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("query must not be empty")
        keywords = tuple(
            dict.fromkeys(
                token.lower()
                for token in re.findall(r"[A-Za-z0-9_+.-]{2,}|[\u4e00-\u9fff]{2,}", normalized)
            )
        )
        return RewrittenQuery(
            original_query=normalized,
            rewritten_query=normalized,
            search_queries=(normalized,),
            keywords=keywords[:20],
        )


class LLMQueryRewriter:
    """Use a small structured model to remove ambiguity and create search expansions."""

    def __init__(self, model: StructuredModel) -> None:
        self._model = model

    def rewrite(self, query: str) -> RewrittenQuery:
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("query must not be empty")
        messages = [
            SystemMessage(
                content=(
                    "You rewrite research questions for hybrid retrieval. Preserve proper nouns, "
                    "time ranges, geography, and technical terms. Produce one precise primary "
                    "query, 1-5 diverse search queries, and discriminative keywords. Do not answer "
                    "the question and do not add unsupported constraints."
                )
            ),
            HumanMessage(content=normalized),
        ]
        result = self._model.invoke(RewrittenQuery, messages)
        return result.model_copy(update={"original_query": normalized})
