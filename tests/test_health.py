"""Smoke tests for the initialized HTTP service."""

import pytest
from fastapi.testclient import TestClient

from agents.runtime import AgentRuntime
from api.main import _follow_up_context, _follow_up_goal, _research_response, create_app
from core.config import Settings
from graph.state import WorkflowStatus, create_initial_state
from tests.test_agents import SchemaFakeModel


def test_health_endpoint() -> None:
    client = TestClient(create_app(Settings(app_env="test")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "research-agent",
        "version": "0.1.0",
        "environment": "test",
    }


def test_root_serves_research_console() -> None:
    client = TestClient(create_app(Settings(app_env="test")))

    response = client.get("/")

    assert response.status_code == 200
    assert "DeepResearch" in response.text
    assert "多智能体调研工作台" in response.text
    assert 'id="modelSelect"' in response.text
    assert 'id="conversationMessages"' in response.text
    assert 'id="reasoningTimeline"' in response.text
    assert 'id="followupForm"' in response.text


def test_readiness_does_not_require_secrets_for_process_startup() -> None:
    client = TestClient(create_app(Settings(app_env="test")))

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "openai_configured": False,
        "search_configured": False,
        "langsmith_enabled": False,
    }


def test_runtime_config_never_exposes_secrets() -> None:
    client = TestClient(create_app(Settings(app_env="test")))

    response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert "openai_api_key" not in payload
    assert "tavily_api_key" not in payload
    assert "langsmith_api_key" not in payload
    assert payload["openai_configured"] is False


def test_runtime_config_updates_readiness_without_returning_key() -> None:
    client = TestClient(create_app(Settings(app_env="test")))
    secret = "test-secret-must-not-be-returned"

    response = client.post(
        "/config",
        json={
            "openai_api_key": secret,
            "openai_base_url": "https://example.invalid/v1",
            "small_model": "small-test-model",
            "large_model": "large-test-model",
            "tavily_api_key": "test-tavily-secret",
            "tools_enabled": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["openai_configured"] is True
    assert response.json()["tavily_configured"] is True
    assert response.json()["available_models"] == ["small-test-model", "large-test-model"]
    assert secret not in response.text
    assert "test-tavily-secret" not in response.text
    assert client.get("/ready").json() == {
        "status": "ready",
        "openai_configured": True,
        "search_configured": True,
        "langsmith_enabled": False,
    }


def test_runtime_config_validates_research_quality_budget() -> None:
    client = TestClient(create_app(Settings(app_env="test")))

    response = client.post(
        "/config",
        json={
            "research_initial_searches": 8,
            "research_max_searches_per_task": 5,
        },
    )

    assert response.status_code == 422
    assert "RESEARCH_MAX_SEARCHES_PER_TASK" in response.json()["detail"]


def test_research_endpoint_reports_missing_model_configuration() -> None:
    client = TestClient(create_app(Settings(app_env="test")))

    response = client.post("/research", json={"user_goal": "研究 Agent 平台"})

    assert response.status_code == 503
    assert "small_model" in response.json()["detail"]


def test_live_research_rejects_model_only_execution() -> None:
    client = TestClient(
        create_app(
            Settings(
                app_env="test",
                openai_api_key="test-key",
                small_model="small-test-model",
                large_model="large-test-model",
                tools_enabled=False,
                require_external_evidence=True,
            )
        )
    )

    response = client.post("/research", json={"user_goal": "研究 Agent 平台"})

    assert response.status_code == 409
    assert "Tavily" in response.json()["detail"]
    assert "不会用模型内置知识" in response.json()["detail"]


def test_resume_request_requires_thread_id() -> None:
    client = TestClient(create_app(Settings(app_env="test")))

    response = client.post("/research", json={"resume": True})

    assert response.status_code == 422


def test_research_response_marks_idle_checkpoint_as_resumable_pause() -> None:
    state = create_initial_state(
        "Research interruption",
        run_id="thread-paused-response",
        user_id="user-1",
    )

    paused = _research_response(state, None, execution_active=False)
    active = _research_response(state, None, execution_active=True)

    assert paused.status == WorkflowStatus.PAUSED
    assert paused.paused is True
    assert paused.can_resume is True
    assert paused.user_goal == "Research interruption"
    assert active.status == WorkflowStatus.PLANNING
    assert active.paused is False
    assert active.can_resume is True


def test_http_api_resumes_supervisor_segments_without_restarting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SchemaFakeModel()
    runtime = AgentRuntime(small_model=model, large_model=model)
    monkeypatch.setattr(
        AgentRuntime,
        "from_settings",
        classmethod(lambda cls, settings=None, **kwargs: runtime),
    )
    client = TestClient(
        create_app(
            Settings(
                app_env="test",
                openai_api_key="test-key",
                small_model="small-test-model",
                large_model="large-test-model",
                require_external_evidence=False,
            )
        )
    )

    response = client.post(
        "/research",
        json={
            "user_goal": "调研 Agent 平台",
            "user_id": "user-1",
            "thread_id": "thread-http-segments",
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "paused"
    assert payload["can_resume"] is True

    for _ in range(10):
        response = client.post(
            "/research",
            json={
                "resume": True,
                "thread_id": "thread-http-segments",
                "user_id": "user-1",
            },
        )
        payload = response.json()
        if not payload["can_resume"]:
            break

    assert payload["status"] == "completed"
    assert payload["final_report"].startswith("# 大模型 Agent 平台趋势调研")
    assert sum(model.calls.values()) == 7


def test_http_api_supports_follow_up_turn_with_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SchemaFakeModel()
    runtime = AgentRuntime(small_model=model, large_model=model)
    created_settings: list[Settings | None] = []

    def runtime_from_settings(
        cls: type[AgentRuntime],
        settings: Settings | None = None,
        **kwargs: object,
    ) -> AgentRuntime:
        del cls, kwargs
        created_settings.append(settings)
        return runtime

    monkeypatch.setattr(
        AgentRuntime,
        "from_settings",
        classmethod(runtime_from_settings),
    )
    client = TestClient(
        create_app(
            Settings(
                app_env="test",
                openai_api_key="test-key",
                small_model="small-test-model",
                large_model="large-test-model",
                require_external_evidence=False,
            )
        )
    )

    first = client.post(
        "/research",
        json={
            "user_goal": "调研 Agent 平台",
            "user_id": "user-1",
            "thread_id": "turn-1",
            "conversation_id": "conversation-1",
        },
    ).json()
    for _ in range(10):
        if not first["can_resume"]:
            break
        first = client.post(
            "/research",
            json={"resume": True, "thread_id": "turn-1", "user_id": "user-1"},
        ).json()
    assert first["status"] == "completed"

    follow_up = client.post(
        "/research",
        json={
            "user_goal": "只比较企业级产品，并补充部署方式",
            "user_id": "user-1",
            "thread_id": "turn-2",
            "conversation_id": "conversation-1",
            "parent_thread_id": "turn-1",
            "model": "large-test-model",
        },
    )
    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["conversation_id"] == "conversation-1"
    assert payload["parent_thread_id"] == "turn-1"
    assert payload["user_goal"] == "只比较企业级产品，并补充部署方式"
    assert payload["selected_model"] == "large-test-model"
    assert created_settings[-1] is not None
    assert created_settings[-1].small_model == "large-test-model"
    assert created_settings[-1].large_model == "large-test-model"

    for _ in range(10):
        if not payload["can_resume"]:
            break
        payload = client.post(
            "/research",
            json={"resume": True, "thread_id": "turn-2", "user_id": "user-1"},
        ).json()
    assert payload["status"] == "completed"
    assert any(step["name"] == "workflow_completed" for step in payload["reasoning_steps"])


def test_follow_up_context_is_bounded_and_keeps_current_message() -> None:
    parent = {
        "run_id": "turn-1",
        "user_message": "第一轮问题",
        "final_report": "上一轮结论" * 500,
    }
    goal = _follow_up_goal(
        parent,
        "当前追问",
        1200,
    )
    compressed = _follow_up_context(parent, "当前追问", 1200)

    assert goal is not None
    assert compressed is not None
    assert compressed.record is not None
    assert compressed.record.boundary == "conversation_turn"
    assert compressed.record.checkpoint_required is True
    assert "第一轮问题" in goal
    assert goal.endswith("当前追问")
    assert len(goal) < 1400
