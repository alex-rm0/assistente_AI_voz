from __future__ import annotations

import pytest

from evals.compare_model_reports import DEFAULT_SUBSET, render_side_by_side
from evals.report import render_markdown, summarize
from evals.schemas import CaseEvaluation, EvalCase, TurnCase, TurnEvaluation, TurnResult


def _report(
    provider: str,
    model: str,
    *,
    passed: bool,
    latency_ms: float,
    cost: float,
    assertions: list[dict],
    response: str = "Podes rever este texto, por favor?",
    case_ids: list[str] | None = None,
) -> dict:
    return {
        "summary": {"provider": provider, "model": model},
        "cases": [
            {
                "id": case_id,
                "turns": [
                    {
                        "passed": passed,
                        "assertions": assertions,
                        "result": {
                            "final_response": response,
                            "latency_ms": latency_ms,
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "estimated_cost_usd": cost,
                            "provider": provider,
                            "model": model,
                            "llm_calls": 1,
                        },
                    }
                ],
            }
            for case_id in (case_ids or ["model_comparison_ptpt_002"])
        ],
    }


def test_comparison_report_includes_human_review_fields() -> None:
    left = _report("ollama", "llama3.1:8b", passed=True, latency_ms=500, cost=0.0, assertions=[])
    right = _report("anthropic", "claude-haiku-4-5-20251001", passed=True, latency_ms=900, cost=0.001, assertions=[])

    markdown = render_side_by_side(left, right, ["model_comparison_ptpt_002"])

    assert "## Revisao humana" in markdown
    assert "naturalidade: [ ] ollama/llama3.1:8b / [ ] anthropic/claude-haiku-4-5-20251001 / [ ] empate" in markdown
    assert "preferencia final" in markdown
    assert "notas:" in markdown
    assert "## Resumo final de revisao humana" in markdown


def test_comparison_report_calculates_costs_latency_and_failures() -> None:
    left = _report(
        "ollama",
        "llama3.1:8b",
        passed=False,
        latency_ms=500,
        cost=0.0,
        assertions=[{"name": "no_ungrounded_computer_observation", "passed": False}],
    )
    right = _report(
        "anthropic",
        "claude-haiku-4-5-20251001",
        passed=True,
        latency_ms=1000,
        cost=0.002,
        assertions=[{"name": "max_questions", "passed": False}],
    )

    markdown = render_side_by_side(left, right, ["model_comparison_ptpt_002"])

    assert "| casos passados | 0 | 1 |" in markdown
    assert "| latencia media ms | 500 | 1000 |" in markdown
    assert "| custo total USD | 0 | 0.002 |" in markdown
    assert "| falhas do modelo | 1 | 1 |" in markdown
    assert "| perguntas desnecessarias | 0 | 1 |" in markdown
    assert "| alegacoes nao fundamentadas | 1 | 0 |" in markdown


def test_comparison_report_fails_when_expected_cases_are_missing() -> None:
    left = _report("ollama", "llama3.1:8b", passed=True, latency_ms=500, cost=0.0, assertions=[])
    right = _report("anthropic", "claude-haiku-4-5-20251001", passed=True, latency_ms=900, cost=0.001, assertions=[])

    with pytest.raises(ValueError, match="missing_left=.*model_quality_email_001"):
        render_side_by_side(left, right, DEFAULT_SUBSET)


def test_default_comparison_subset_has_ten_cases_and_renders_twenty_rows() -> None:
    left = _report(
        "ollama",
        "llama3.1:8b",
        passed=True,
        latency_ms=500,
        cost=0.0,
        assertions=[],
        case_ids=list(DEFAULT_SUBSET),
    )
    right = _report(
        "anthropic",
        "claude-haiku-4-5-20251001",
        passed=True,
        latency_ms=900,
        cost=0.001,
        assertions=[],
        case_ids=list(DEFAULT_SUBSET),
    )

    markdown = render_side_by_side(left, right, DEFAULT_SUBSET)

    assert len(DEFAULT_SUBSET) == 10
    assert "- Casos comparados: 10" in markdown
    assert markdown.count("| model_quality_") == 20


def test_ptpt_report_flags_revisar_and_okay() -> None:
    left = _report(
        "ollama",
        "llama3.1:8b",
        passed=True,
        latency_ms=500,
        cost=0.0,
        assertions=[],
        response="Okay, posso revisar isso.",
    )
    right = _report("anthropic", "claude-haiku-4-5-20251001", passed=True, latency_ms=900, cost=0.001, assertions=[])

    markdown = render_side_by_side(left, right, ["model_comparison_ptpt_002"])

    assert "| model_comparison_ptpt_002 | ollama/llama3.1:8b | sim | 500 ms | 100/20 | $0.000000 | rever |" in markdown


def test_pipeline_failure_is_shown_separately_from_model_failure() -> None:
    left = _report(
        "ollama",
        "llama3.1:8b",
        passed=False,
        latency_ms=500,
        cost=0.0,
        assertions=[{"name": "memory_write_action", "passed": False}],
    )
    right = _report(
        "anthropic",
        "claude-haiku-4-5-20251001",
        passed=False,
        latency_ms=900,
        cost=0.001,
        assertions=[{"name": "memory_write_action", "passed": False}],
    )

    markdown = render_side_by_side(left, right, ["model_comparison_ptpt_002"])

    assert "nenhuma | memory_write_action" in markdown
    assert "| falhas do modelo | 0 | 0 |" in markdown
    assert "| falhas partilhadas do pipeline | 1 |  |" in markdown


def test_model_quality_comparison_fails_when_llm_calls_are_zero() -> None:
    left = _report(
        "ollama",
        "llama3.1:8b",
        passed=True,
        latency_ms=500,
        cost=0.0,
        assertions=[],
        case_ids=["model_quality_email_001"],
    )
    right = _report(
        "anthropic",
        "claude-haiku-4-5-20251001",
        passed=True,
        latency_ms=900,
        cost=0.001,
        assertions=[],
        case_ids=["model_quality_email_001"],
    )
    left["cases"][0]["turns"][0]["result"]["llm_calls"] = 0

    with pytest.raises(ValueError, match="model_quality exige exactamente uma chamada LLM"):
        render_side_by_side(left, right, ["model_quality_email_001"])


def _case_eval(category: str, case_id: str) -> CaseEvaluation:
    result = TurnResult(
        user_message="x",
        final_response="Resposta.",
        selected_path="GENERAL_CONVERSATION",
        response_source="RESPONSE_COMPOSER",
        model="llama3.1:8b",
        model_source="settings.json",
        llm_calls=1,
        llm_call_sources=["RESPONSE_COMPOSER"],
        tools_used=[],
        selected_memory_ids=[],
        memory_write_action=None,
        grounding_sources=[],
        latency_ms=10.0,
        exception_type=None,
        exception_message=None,
        provider="ollama",
        requested_provider="ollama",
        requested_model="llama3.1:8b",
    )
    turn_eval = TurnEvaluation(
        turn_index=0,
        user_message="x",
        result=result,
        assertions=[],
        passed=True,
    )
    return CaseEvaluation(
        case=EvalCase(id=case_id, category=category, description="", turns=(TurnCase(user="x"),)),
        turn_evaluations=[turn_eval],
        passed=True,
        provider="ollama",
        model="llama3.1:8b",
        model_source="settings.json",
    )


def test_system_behavior_report_does_not_use_model_quality_diagnostics() -> None:
    evaluations = [_case_eval("system_behavior", "system_behavior_context_honesty_005")]
    markdown = render_markdown("run", summarize(evaluations, "ollama", "llama3.1:8b"), evaluations)

    assert "## Diagnóstico system_behavior" in markdown
    assert "## Diagnóstico model_quality" not in markdown
    assert "`llm_calls=0` pode ser o comportamento correto" in markdown


def test_model_quality_report_does_not_use_system_behavior_diagnostics() -> None:
    evaluations = [_case_eval("model_quality", "model_quality_email_001")]
    markdown = render_markdown("run", summarize(evaluations, "ollama", "llama3.1:8b"), evaluations)

    assert "## Diagnóstico model_quality" in markdown
    assert "## Diagnóstico system_behavior" not in markdown
    assert "Cada caso deve exigir `llm_calls=1`" in markdown
