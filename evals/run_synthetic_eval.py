"""Run deterministic API evaluations without external model or search calls."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from agents.runtime import AgentRuntime
from api.main import create_app
from core.config import Settings
from tests.test_agents import SchemaFakeModel

CASES_PATH = Path(__file__).with_name("synthetic_cases.json")
TERMINAL_STATUSES = {"completed", "failed", "waiting_human"}


def _load_cases() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def _finish_turn(
    client: TestClient,
    payload: dict[str, Any],
    *,
    max_segments: int = 12,
) -> tuple[dict[str, Any], int, float]:
    started = perf_counter()
    response = client.post("/research", json=payload)
    data = response.json()
    if response.status_code != 200:
        raise AssertionError(f"turn start failed ({response.status_code}): {data}")
    segments = 1
    while data.get("can_resume") and data.get("status") not in TERMINAL_STATUSES:
        if segments >= max_segments:
            raise AssertionError(f"turn exceeded {max_segments} segments: {payload['thread_id']}")
        response = client.post(
            "/research",
            json={
                "resume": True,
                "thread_id": data["thread_id"],
                "user_id": data.get("user_id", "eval-user"),
            },
        )
        data = response.json()
        if response.status_code != 200:
            raise AssertionError(f"turn resume failed ({response.status_code}): {data}")
        segments += 1
    return data, segments, (perf_counter() - started) * 1000


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 2)


def evaluate() -> dict[str, Any]:
    """Execute all cases and return orchestration metrics suitable for CI assertions."""

    cases = _load_cases()
    fake_model = SchemaFakeModel()
    runtime = AgentRuntime(small_model=fake_model, large_model=fake_model)

    def fake_runtime_from_settings(
        cls: type[AgentRuntime],
        settings: Settings | None = None,
        **kwargs: object,
    ) -> AgentRuntime:
        del cls, settings, kwargs
        return runtime

    settings = Settings(
        app_env="evaluation",
        openai_api_key="offline-eval-key",
        small_model="small-test-model",
        large_model="large-test-model",
        tools_enabled=False,
        require_external_evidence=False,
        research_quality_gate_enabled=False,
    )

    completed: list[bool] = []
    resumed: list[bool] = []
    linked: list[bool] = []
    model_persisted: list[bool] = []
    trace_present: list[bool] = []
    trace_safe: list[bool] = []
    report_present: list[bool] = []
    segment_counts: list[int] = []
    latencies_ms: list[float] = []
    turn_results: list[dict[str, Any]] = []

    with patch.object(
        AgentRuntime,
        "from_settings",
        classmethod(fake_runtime_from_settings),
    ):
        client = TestClient(create_app(settings))
        for case in cases["research_cases"]:
            conversation_id = f"eval-{case['id']}"
            first_thread = f"{conversation_id}-turn-1"
            first, first_segments, first_latency = _finish_turn(
                client,
                {
                    "user_goal": case["goal"],
                    "user_id": "eval-user",
                    "thread_id": first_thread,
                    "conversation_id": conversation_id,
                    "model": case["model"],
                },
            )
            second_thread = f"{conversation_id}-turn-2"
            second, second_segments, second_latency = _finish_turn(
                client,
                {
                    "user_goal": case["follow_up"],
                    "user_id": "eval-user",
                    "thread_id": second_thread,
                    "conversation_id": conversation_id,
                    "parent_thread_id": first_thread,
                    "model": case["model"],
                },
            )

            for turn, segments, latency, expected_message in (
                (first, first_segments, first_latency, case["goal"]),
                (second, second_segments, second_latency, case["follow_up"]),
            ):
                serialized_trace = json.dumps(
                    turn.get("reasoning_steps", []),
                    ensure_ascii=False,
                )
                completed.append(turn.get("status") == "completed")
                resumed.append(segments > 1 and turn.get("status") == "completed")
                model_persisted.append(turn.get("selected_model") == case["model"])
                trace_present.append(bool(turn.get("reasoning_steps")))
                trace_safe.append(expected_message not in serialized_trace)
                report_present.append(bool(turn.get("final_report")))
                segment_counts.append(segments)
                latencies_ms.append(latency)
                turn_results.append(
                    {
                        "thread_id": turn["thread_id"],
                        "status": turn["status"],
                        "segments": segments,
                        "latency_ms": round(latency, 2),
                        "reasoning_steps": len(turn.get("reasoning_steps", [])),
                    }
                )
            linked.append(
                second.get("conversation_id") == conversation_id
                and second.get("parent_thread_id") == first_thread
                and second.get("user_goal") == case["follow_up"]
            )

        validation_results = []
        for case in cases["validation_cases"]:
            response = client.post("/research", json=case["payload"])
            validation_results.append(
                {
                    "id": case["id"],
                    "expected_status": case["expected_status"],
                    "actual_status": response.status_code,
                    "passed": response.status_code == case["expected_status"],
                }
            )

    validation_passes = [result["passed"] for result in validation_results]
    return {
        "dataset_version": cases["version"],
        "research_cases": len(cases["research_cases"]),
        "research_turns": len(turn_results),
        "validation_cases": len(validation_results),
        "metrics": {
            "workflow_completion_rate": _rate(completed),
            "checkpoint_resume_success_rate": _rate(resumed),
            "follow_up_link_rate": _rate(linked),
            "model_selection_persistence_rate": _rate(model_persisted),
            "reasoning_trace_presence_rate": _rate(trace_present),
            "reasoning_trace_prompt_privacy_rate": _rate(trace_safe),
            "report_generation_rate": _rate(report_present),
            "api_validation_pass_rate": _rate(validation_passes),
            "segments_per_turn_mean": round(mean(segment_counts), 2),
            "latency_ms_p50": round(median(latencies_ms), 2),
            "latency_ms_p95": _percentile(latencies_ms, 0.95),
            "fake_model_calls": sum(fake_model.calls.values()),
        },
        "validation_results": validation_results,
        "turn_results": turn_results,
        "limitations": [
            "Uses a deterministic schema-aware fake model; factual correctness is not measured.",
            (
                "External search is disabled; citation precision, source quality and "
                "freshness are not measured."
            ),
            "Token and monetary cost are not representative of a live provider.",
        ],
    }


def main() -> None:
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
