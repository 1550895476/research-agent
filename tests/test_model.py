"""Structured-output compatibility tests for OpenAI-compatible providers."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from agents.model import LangChainStructuredModel


class ExampleOutput(BaseModel):
    value: str


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeRunnable:
    def __init__(self, result: object) -> None:
        self._result = result

    def invoke(self, messages: list[object]) -> object:
        assert messages
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeChatModel:
    def __init__(self, results: dict[str, object], raw_content: str = "") -> None:
        self.results = results
        self.structured_calls: list[dict[str, Any]] = []
        self.raw_calls = 0
        self.raw_content = raw_content

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> FakeRunnable:
        assert schema is ExampleOutput
        self.structured_calls.append(kwargs)
        return FakeRunnable(self.results[str(kwargs["method"])])

    def invoke(self, messages: list[object]) -> AIMessage:
        assert messages
        self.raw_calls += 1
        return AIMessage(content=self.raw_content)


def test_json_schema_capability_error_falls_back_and_caches_method() -> None:
    provider = FakeChatModel(
        {
            "json_schema": ProviderError(
                "This response_format type is unavailable now",
                400,
            ),
            "function_calling": {"value": "fallback worked"},
        }
    )
    model = LangChainStructuredModel(provider, method="json_schema")  # type: ignore[arg-type]

    first = model.invoke(ExampleOutput, [HumanMessage(content="first")])
    second = model.invoke(ExampleOutput, [HumanMessage(content="second")])

    assert first.value == "fallback worked"
    assert second.value == "fallback worked"
    assert [call["method"] for call in provider.structured_calls] == [
        "json_schema",
        "function_calling",
        "function_calling",
    ]
    assert provider.structured_calls[0]["strict"] is True
    assert "strict" not in provider.structured_calls[1]


def test_json_mode_capability_error_falls_back_to_plain_json() -> None:
    provider = FakeChatModel(
        {
            "json_mode": ProviderError("response_format is unsupported", 400),
        },
        raw_content='```json\n{"value":"plain JSON worked"}\n```',
    )
    model = LangChainStructuredModel(provider, method="json_mode")  # type: ignore[arg-type]

    result = model.invoke(ExampleOutput, [HumanMessage(content="fallback")])

    assert result.value == "plain JSON worked"
    assert provider.raw_calls == 1


def test_non_capability_error_is_not_hidden_by_fallback() -> None:
    error = ProviderError("invalid API key", 401)
    provider = FakeChatModel({"json_schema": error})
    model = LangChainStructuredModel(provider, method="json_schema")  # type: ignore[arg-type]

    with pytest.raises(ProviderError) as captured:
        model.invoke(ExampleOutput, [HumanMessage(content="must fail")])

    assert captured.value is error
    assert len(provider.structured_calls) == 1
    assert provider.raw_calls == 0
