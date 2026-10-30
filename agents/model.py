"""Model construction and schema-first invocation for OpenAI-compatible chat APIs."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from enum import StrEnum
from threading import Lock
from typing import Protocol, TypeVar
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from core.config import Settings

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)
logger = logging.getLogger(__name__)


class ModelTier(StrEnum):
    """Static model tiers used before Dynamic Model Router is added in Step 7."""

    SMALL = "small"
    LARGE = "large"


class ModelConfigurationError(ValueError):
    """Raised when an API-backed agent cannot be configured safely."""


class StructuredModel(Protocol):
    """Minimal schema-first model contract consumed by all agents."""

    def invoke(
        self,
        schema: type[StructuredOutput],
        messages: Sequence[BaseMessage],
    ) -> StructuredOutput:
        """Return one instance conforming to schema."""


class LangChainStructuredModel:
    """Adapt a chat model with capability-aware structured-output fallback."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        method: str = "json_schema",
    ) -> None:
        self._model = model
        self._method = method
        self._active_method = method
        self._method_lock = Lock()

    def invoke(
        self,
        schema: type[StructuredOutput],
        messages: Sequence[BaseMessage],
    ) -> StructuredOutput:
        last_capability_error: Exception | None = None
        for method in self._fallback_methods():
            try:
                response = self._invoke_with_method(method, schema, messages)
            except Exception as exc:  # noqa: BLE001 - provider SDK errors vary by vendor
                if method == "plain_json" or not _is_capability_error(exc):
                    raise
                last_capability_error = exc
                logger.warning(
                    "Structured output method %s is unavailable; trying a compatible fallback",
                    method,
                )
                continue
            self._remember_working_method(method)
            return response
        if last_capability_error is not None:
            raise last_capability_error
        raise RuntimeError("no structured output method was attempted")

    def _fallback_methods(self) -> tuple[str, ...]:
        with self._method_lock:
            active_method = self._active_method
        fallback_order = {
            "json_schema": ("json_schema", "function_calling", "json_mode", "plain_json"),
            "function_calling": ("function_calling", "json_mode", "plain_json"),
            "json_mode": ("json_mode", "plain_json"),
            "plain_json": ("plain_json",),
        }
        methods = fallback_order.get(active_method, (active_method, "plain_json"))
        return tuple(dict.fromkeys(methods))

    def _invoke_with_method(
        self,
        method: str,
        schema: type[StructuredOutput],
        messages: Sequence[BaseMessage],
    ) -> StructuredOutput:
        if method == "plain_json":
            return self._invoke_plain_json(schema, messages)

        kwargs: dict[str, object] = {"method": method}
        if method == "json_schema":
            kwargs["strict"] = True
        runnable = self._model.with_structured_output(schema, **kwargs)
        invocation_messages = (
            _messages_with_json_schema(schema, messages)
            if method == "json_mode"
            else list(messages)
        )
        response = runnable.invoke(invocation_messages)
        return schema.model_validate(response)

    def _invoke_plain_json(
        self,
        schema: type[StructuredOutput],
        messages: Sequence[BaseMessage],
    ) -> StructuredOutput:
        response = self._model.invoke(_messages_with_json_schema(schema, messages))
        content = _message_text(response)
        try:
            return schema.model_validate_json(content)
        except ValueError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                raise
            return schema.model_validate_json(content[start : end + 1])

    def _remember_working_method(self, method: str) -> None:
        with self._method_lock:
            previous_method = self._active_method
            self._active_method = method
        if method != previous_method:
            logger.warning(
                "Structured output automatically switched from %s to %s for this runtime",
                previous_method,
                method,
            )


def _messages_with_json_schema(
    schema: type[StructuredOutput],
    messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    format_message = SystemMessage(
        content=(
            "Return only one valid JSON object matching the following JSON Schema. "
            "Do not use Markdown fences or add explanatory text.\n"
            f"{schema_json}"
        )
    )
    return [format_message, *messages]


def _message_text(message: object) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        return "".join(parts).strip()
    return str(content).strip()


def _is_capability_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code not in {400, 404, 422}:
        return False
    message = str(exc).lower()
    capability_terms = (
        "response_format",
        "json_schema",
        "structured output",
        "structured_output",
        "function calling",
        "function_calling",
        "tool_choice",
        "strict",
    )
    rejection_terms = (
        "unavailable",
        "unsupported",
        "not support",
        "not allowed",
        "unknown",
        "invalid",
    )
    return any(term in message for term in capability_terms) and any(
        term in message for term in rejection_terms
    )


class ModelFactory:
    """Create small/large structured models from environment-backed settings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create(self, tier: ModelTier) -> LangChainStructuredModel:
        model_name = (
            self._settings.small_model if tier == ModelTier.SMALL else self._settings.large_model
        )
        if not model_name:
            raise ModelConfigurationError(f"{tier.value}_model must be configured")

        api_key = self._resolve_api_key()
        model = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url=self._settings.openai_base_url,
            temperature=0,
            timeout=self._settings.model_timeout_seconds,
            max_retries=self._settings.model_max_retries,
        )
        return LangChainStructuredModel(
            model,
            method=self._settings.structured_output_method,
        )

    def _resolve_api_key(self) -> str:
        if self._settings.openai_api_key:
            return self._settings.openai_api_key.get_secret_value()
        hostname = (urlparse(self._settings.openai_base_url).hostname or "").lower()
        if hostname == "api.openai.com":
            raise ModelConfigurationError("OPENAI_API_KEY is required for the OpenAI API")
        # Local OpenAI-compatible servers commonly require a non-empty placeholder.
        return "not-required"
