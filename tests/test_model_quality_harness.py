from __future__ import annotations

from pathlib import Path

import pytest

from evals.assertions import run_turn_assertions
from evals.compare_model_reports import render_side_by_side
from evals.harness import EvalRun
from evals.report import render_markdown, summarize
from evals.run_evals import load_case_ids, load_cases
from evals.schemas import CaseEvaluation, EvalCase, TurnCase, TurnEvaluation, TurnExpectation, TurnResult


def _result(text: str, *, llm_calls: int = 1, response_source: str = "RESPONSE_COMPOSER") -> TurnResult:
    return TurnResult(
        user_message="x",
        final_response=text,
        selected_path="GENERAL_CONVERSATION",
        response_source=response_source,
        model="llama3.1:8b",
        model_source="provider:ollama",
        llm_calls=llm_calls,
        llm_call_sources=["RESPONSE_COMPOSER"] if llm_calls else [],
        tools_used=[],
        selected_memory_ids=[],
        memory_write_action=None,
        grounding_sources=[],
        latency_ms=1.0,
        exception_type=None,
        exception_message=None,
        provider="ollama",
        requested_provider="ollama",
        requested_model="llama3.1:8b",
    )


def _case_eval(case_id: str, user_messages: list[str], *, response: str = "Resposta.") -> CaseEvaluation:
    turns = []
    for index, user in enumerate(user_messages):
        result = _result(response)
        result.user_message = user
        turns.append(
            TurnEvaluation(
                turn_index=index,
                user_message=user,
                result=result,
                assertions=[],
                passed=True,
            )
        )
    return CaseEvaluation(
        case=EvalCase(
            id=case_id,
            category="model_quality",
            description="",
            turns=tuple(TurnCase(user=user) for user in user_messages),
        ),
        turn_evaluations=turns,
        passed=True,
        provider="ollama",
        model="llama3.1:8b",
        model_source="provider:ollama",
    )


def _report(case_eval: CaseEvaluation) -> dict:
    return {
        "summary": {"provider": case_eval.provider, "model": case_eval.model},
        "cases": [case_eval.to_dict()],
    }


def test_eval_run_creates_fresh_case_directories_and_cleans_existing_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ECHO_TEST_DATA_DIR", str(tmp_path / "eval-root"))
    run = EvalRun(keep_data=True)

    case_a = run.case_dir("case_a")
    (case_a / "data").mkdir()
    (case_a / "data" / "history.json").write_text("[{\"role\":\"user\",\"content\":\"exame\"}]", encoding="utf-8")
    case_b = run.case_dir("case_b")
    case_a_again = run.case_dir("case_a")

    assert case_a != case_b
    assert case_a_again == case_a
    assert not (case_a_again / "data" / "history.json").exists()


def test_case_order_does_not_change_case_directories(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ECHO_TEST_DATA_DIR", str(tmp_path / "eval-root-order"))
    run = EvalRun(keep_data=True)

    forward = [run.case_dir(case_id).name for case_id in ("case_a", "case_b")]
    backward = [run.case_dir(case_id).name for case_id in ("case_b", "case_a")]

    assert forward == ["case_a", "case_b"]
    assert backward == ["case_b", "case_a"]


def test_model_quality_subset_prompts_match_expected_cases() -> None:
    cases = load_cases("model_quality", None, False, load_case_ids("evals/model_quality_subset.txt"))
    prompts = {case.id: case.turns[-1].user for case in cases}

    assert "modelo local" in prompts["model_quality_comparison_006"]
    assert "modelo cloud" in prompts["model_quality_comparison_006"]
    assert "exame" not in prompts["model_quality_comparison_006"].lower()
    assert "quatro ideias principais" in prompts["model_quality_summary_007"]


def test_semantically_correct_technical_explanation_does_not_require_literal_python() -> None:
    expected = TurnExpectation(
        must_contain_each_any=(
            ("síncrona", "sincrona"),
            ("assíncrona", "assincrona"),
            ("bloqueia", "aguarda", "espera"),
            ("continua", "outras tarefas", "sem bloquear"),
        )
    )

    outcomes = run_turn_assertions(
        expected,
        _result("Uma função síncrona aguarda a operação terminar. Uma assíncrona pode continuar outras tarefas sem bloquear."),
    )

    assert all(outcome.passed for outcome in outcomes)


def test_summary_without_four_points_fails_when_four_points_are_required() -> None:
    outcomes = run_turn_assertions(
        TurnExpectation(min_bullet_points=4),
        _result("A biblioteca alargou o horário durante os exames para apoiar estudantes."),
    )

    failed = [outcome for outcome in outcomes if outcome.name == "min_bullet_points"]
    assert failed
    assert failed[0].passed is False


def test_model_cloud_comparison_about_exam_fails_irrelevance() -> None:
    outcomes = run_turn_assertions(
        TurnExpectation(must_not_contain=("exame", "estudo", "data")),
        _result("Para um exame, estudar por exercícios pode ajudar mais se a data estiver próxima."),
    )

    failed = [outcome for outcome in outcomes if outcome.name.startswith("must_not_contain")]
    assert failed


def test_report_shows_prompt_history_response_and_failed_assertions() -> None:
    result = _result("Um parágrafo só.")
    assertion = run_turn_assertions(TurnExpectation(min_bullet_points=4), result)
    turn = TurnEvaluation(
        turn_index=0,
        user_message="Transforma em quatro ideias principais.",
        result=result,
        assertions=assertion,
        passed=False,
        failure_classification="MODEL_BEHAVIOR",
    )
    evaluation = CaseEvaluation(
        case=EvalCase(
            id="model_quality_summary_007",
            category="model_quality",
            description="",
            turns=(TurnCase(user="Transforma em quatro ideias principais."),),
        ),
        turn_evaluations=[turn],
        passed=False,
        provider="ollama",
        model="llama3.1:8b",
    )

    markdown = render_markdown("run", summarize([evaluation], "ollama", "llama3.1:8b"), [evaluation])

    assert "## Prompts executados" in markdown
    assert "Transforma em quatro ideias principais." in markdown
    assert "turnos anteriores no caso" in markdown
    assert "min_bullet_points" in markdown


def test_comparison_requires_exactly_same_inputs_between_reports() -> None:
    left = _report(_case_eval("model_quality_email_001", ["Escreve um email curto."]))
    right = _report(_case_eval("model_quality_email_001", ["Escreve um email longo."]))

    with pytest.raises(ValueError, match="mesmos inputs"):
        render_side_by_side(left, right, ["model_quality_email_001"])


def test_rewrite_can_return_two_useful_versions_without_failing() -> None:
    expected = TurnExpectation(
        must_contain_any=("aceder", "ficheiros", "ecrã", "rever"),
        must_not_contain=("acessar", "arquivos", "tela", "revisar", "você", "Okay"),
        max_words=70,
    )
    response = (
        "Versão principal: \"Vou aceder aos ficheiros e proceder à revisão posteriormente.\" "
        "Alternativa: \"Irei consultar os ficheiros e rever o conteúdo mais tarde.\""
    )

    outcomes = run_turn_assertions(expected, _result(response))

    assert all(outcome.passed for outcome in outcomes)


def test_emotional_support_passes_with_broad_properties() -> None:
    expected = TurnExpectation(
        must_contain_each_any=(
            ("percebo", "faz sentido", "normal", "entendo", "compreendo", "não é fácil"),
            ("bloqueado", "cansado", "medo", "ritmo", "difícil", "pressão"),
            ("um passo", "pequeno passo", "respirar", "começar por", "o que está a acontecer"),
        ),
        max_questions=1,
    )
    response = "Compreendo. Quando estás cansado e com medo de não acompanhar o ritmo, faz sentido sentires-te bloqueado. Começa por um pequeno passo."

    outcomes = run_turn_assertions(expected, _result(response))

    assert all(outcome.passed for outcome in outcomes)


def test_concrete_planning_passes_without_literal_study_words() -> None:
    expected = TurnExpectation(
        must_contain_each_any=(
            ("semana", "7 dias", "segunda", "terça", "quarta"),
            ("Estratégias Algorítmicas", "algoritmos"),
            ("exercícios", "revisão", "rever", "prática", "slides", "blocos", "noite", "depois do trabalho"),
        )
    )
    response = (
        "Assumo uma semana genérica. Segunda e terça revê algoritmos base; quarta faz blocos curtos à noite; "
        "quinta resolve exercícios; sexta faz revisão final."
    )

    outcomes = run_turn_assertions(expected, _result(response))

    assert all(outcome.passed for outcome in outcomes)


def test_honesty_response_can_explicitly_refuse_to_invent() -> None:
    expected = TurnExpectation(
        must_contain_each_any=(
            ("não consigo confirmar", "não tenho dados suficientes", "não tenho informação suficiente"),
            ("não inventes", "não devo inventar", "sem fontes", "confirmar"),
        ),
        must_not_contain=("Chrome aberto", "Zoom aberto", "VS Code aberto", "você"),
    )
    response = "Não tenho dados suficientes para confirmar isso. É melhor não inventar sem fontes."

    outcomes = run_turn_assertions(expected, _result(response))

    assert all(outcome.passed for outcome in outcomes)


def test_email_question_inside_generated_email_is_allowed_but_outer_question_fails() -> None:
    expected = TurnExpectation(max_outer_questions=0)
    valid_email = 'Exmo. Professor, poderia rever o meu relatório amanhã? Obrigado.'
    invalid_followup = '"Exmo. Professor, poderia rever o meu relatório amanhã?" Queres que ajuste o tom?'

    assert all(outcome.passed for outcome in run_turn_assertions(expected, _result(valid_email)))

    outcomes = run_turn_assertions(expected, _result(invalid_followup))
    failed = [outcome for outcome in outcomes if outcome.name == "max_outer_questions"]
    assert failed
    assert failed[0].passed is False
