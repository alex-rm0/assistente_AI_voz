from __future__ import annotations

from evals.assertions import run_turn_assertions
from evals.schemas import TurnExpectation, TurnResult


def _result(text: str) -> TurnResult:
    return TurnResult(
        user_message="x",
        final_response=text,
        selected_path="",
        response_source="",
        model="llama3.1:8b",
        model_source="settings.json",
        llm_calls=1,
        llm_call_sources=["RESPONSE_COMPOSER"],
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


def test_max_words_passes_for_short_response() -> None:
    outcomes = run_turn_assertions(TurnExpectation(max_words=5), _result("Podes rever isto, por favor?"))

    assert all(outcome.passed for outcome in outcomes)


def test_max_words_fails_for_long_response() -> None:
    outcomes = run_turn_assertions(TurnExpectation(max_words=3), _result("Podes rever isto, por favor?"))

    failed = [outcome for outcome in outcomes if outcome.name == "max_words"]
    assert failed
    assert failed[0].passed is False


def test_min_words_fails_for_too_short_response() -> None:
    outcomes = run_turn_assertions(TurnExpectation(min_words=6), _result("Sim, claro."))

    failed = [outcome for outcome in outcomes if outcome.name == "min_words"]
    assert failed
    assert failed[0].passed is False


def test_ungrounded_screen_claim_fails_without_tools_or_grounding() -> None:
    result = _result("Estou a ver o Zoom aberto no teu ecra.")

    outcomes = run_turn_assertions(TurnExpectation(forbid_ungrounded_computer_observation=True), result)

    failed = [outcome for outcome in outcomes if outcome.name == "no_ungrounded_computer_observation"]
    assert failed
    assert failed[0].passed is False


def test_denial_of_screen_access_does_not_fail() -> None:
    result = _result("Nao consigo ver o teu ecra.")

    outcomes = run_turn_assertions(TurnExpectation(forbid_ungrounded_computer_observation=True), result)

    assert all(outcome.passed for outcome in outcomes)


def test_denial_of_application_access_does_not_fail() -> None:
    result = _result("Nao tenho acesso as aplicacoes abertas.")

    outcomes = run_turn_assertions(TurnExpectation(forbid_ungrounded_computer_observation=True), result)

    assert all(outcome.passed for outcome in outcomes)


def test_denial_of_activity_observation_does_not_fail() -> None:
    result = _result("Nao estou a observar a atividade do computador.")

    outcomes = run_turn_assertions(TurnExpectation(forbid_ungrounded_computer_observation=True), result)

    assert all(outcome.passed for outcome in outcomes)


def test_ambiguous_conditional_observation_does_not_fail() -> None:
    result = _result("Posso acompanhar o ecra se ligares uma ferramenta de observacao.")

    outcomes = run_turn_assertions(TurnExpectation(forbid_ungrounded_computer_observation=True), result)

    assert all(outcome.passed for outcome in outcomes)


def test_grounded_screen_claim_passes_with_tool_grounding() -> None:
    result = _result("Estou a observar a janela ativa.")
    result.tools_used = ["get_active_window"]

    outcomes = run_turn_assertions(TurnExpectation(forbid_ungrounded_computer_observation=True), result)

    assert all(outcome.passed for outcome in outcomes)


def test_terminal_mentioned_as_instruction_is_not_observation_claim() -> None:
    result = _result("Podes instalar com pip no terminal.")

    outcomes = run_turn_assertions(TurnExpectation(forbid_ungrounded_computer_observation=True), result)

    assert all(outcome.passed for outcome in outcomes)


def test_claim_about_current_chrome_usage_fails_without_grounding() -> None:
    result = _result("Acho que estas a usar o Google Chrome agora.")

    outcomes = run_turn_assertions(TurnExpectation(forbid_ungrounded_computer_observation=True), result)

    failed = [outcome for outcome in outcomes if outcome.name == "no_ungrounded_computer_observation"]
    assert failed
    assert failed[0].passed is False


def test_unexpected_memory_write_fails_when_case_expects_none() -> None:
    result = _result("Percebo que estejas nervoso.")
    result.memory_write_action = "created"

    outcomes = run_turn_assertions(TurnExpectation(memory_write_action=None), result)

    failed = [outcome for outcome in outcomes if outcome.name == "memory_write_action"]
    assert failed
    assert failed[0].passed is False


def test_emotional_response_too_cold_fails_must_contain_any() -> None:
    outcomes = run_turn_assertions(
        TurnExpectation(must_contain_any=("nervoso", "normal", "percebo")),
        _result("Qual e o exame?"),
    )

    failed = [outcome for outcome in outcomes if outcome.name == "must_contain_any"]
    assert failed
    assert failed[0].passed is False


def test_ptpt_assertion_flags_revisar_and_okay() -> None:
    outcomes = run_turn_assertions(TurnExpectation(), _result("Okay, posso revisar isso."))

    failed = [outcome for outcome in outcomes if outcome.name == "no_brazilian_portuguese"]
    assert failed
    assert failed[0].passed is False


def test_sufficient_short_phrase_request_fails_unnecessary_question() -> None:
    result = _result("Que tipo de revisão queres?")
    result.user_message = "Ajuda-me a escrever uma frase curta para pedir uma revisao."

    outcomes = run_turn_assertions(TurnExpectation(forbid_unnecessary_question_when_sufficient=True), result)

    failed = [outcome for outcome in outcomes if outcome.name == "no_unnecessary_question_when_sufficient"]
    assert failed
    assert failed[0].passed is False


def test_meeting_request_can_still_ask_for_missing_time_context() -> None:
    result = _result("Claro. Para quando?")
    result.user_message = "Quero marcar uma reuniao."

    outcomes = run_turn_assertions(TurnExpectation(forbid_unnecessary_question_when_sufficient=True), result)

    assert all(outcome.passed for outcome in outcomes)


def test_model_quality_case_fails_when_llm_is_not_called() -> None:
    result = _result("Resposta local.")
    result.llm_calls = 0

    outcomes = run_turn_assertions(TurnExpectation(llm_calls_min=1, llm_calls_max=1), result)

    failed = [outcome for outcome in outcomes if outcome.name in {"llm_calls_min", "llm_calls_max"} and not outcome.passed]
    assert failed


def test_provider_mismatch_fails() -> None:
    result = _result("Resposta.")
    result.provider = "ollama"
    result.requested_provider = "anthropic"

    outcomes = run_turn_assertions(TurnExpectation(require_provider_match=True), result)

    failed = [outcome for outcome in outcomes if outcome.name == "provider_matches_request"]
    assert failed
    assert failed[0].passed is False


def test_fallback_fails_when_forbidden() -> None:
    result = _result("Fallback.")
    result.fallback_used = True

    outcomes = run_turn_assertions(TurnExpectation(forbid_fallback=True), result)

    failed = [outcome for outcome in outcomes if outcome.name == "no_fallback"]
    assert failed
    assert failed[0].passed is False


def test_deterministic_response_fails_for_model_quality_case() -> None:
    result = _result("Resposta deterministica.")
    result.response_source = "DETERMINISTIC_HELP"

    outcomes = run_turn_assertions(TurnExpectation(forbid_deterministic_response=True), result)

    failed = [outcome for outcome in outcomes if outcome.name == "no_deterministic_response"]
    assert failed
    assert failed[0].passed is False
