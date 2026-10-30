"""Regression gate for the generated offline evaluation dataset."""

from evals.run_synthetic_eval import evaluate


def test_synthetic_orchestration_evaluation_passes_all_gates() -> None:
    result = evaluate()
    metrics = result["metrics"]

    assert result["research_cases"] == 8
    assert result["research_turns"] == 16
    assert result["validation_cases"] == 5
    assert metrics["workflow_completion_rate"] == 1.0
    assert metrics["checkpoint_resume_success_rate"] == 1.0
    assert metrics["follow_up_link_rate"] == 1.0
    assert metrics["model_selection_persistence_rate"] == 1.0
    assert metrics["reasoning_trace_presence_rate"] == 1.0
    assert metrics["reasoning_trace_prompt_privacy_rate"] == 1.0
    assert metrics["report_generation_rate"] == 1.0
    assert metrics["api_validation_pass_rate"] == 1.0
