from __future__ import annotations

from evals.human_review import detect_review_reasons
from evals.schemas import TurnResult


def _result(final_response: str, **kwargs) -> TurnResult:
    defaults = dict(
        user_message="x",
        final_response=final_response,
        selected_path="",
        response_source="",
        model=None,
        llm_calls=0,
        llm_call_sources=[],
        tools_used=[],
        selected_memory_ids=[],
        memory_write_action=None,
        grounding_sources=[],
        latency_ms=1.0,
        exception_type=None,
        exception_message=None,
    )
    defaults.update(kwargs)
    return TurnResult(**defaults)


def test_plain_response_needs_no_review() -> None:
    assert detect_review_reasons(_result("Ok, entendido.")) == []


def test_unsupported_memory_claim_flagged() -> None:
    result = _result("Sim, lembro-me bem disso.", unsupported_memory_claim_detected=True)
    assert "unsupported_memory_claim" in detect_review_reasons(result)


def test_unsupported_tool_claim_flagged() -> None:
    result = _result("Já pesquisei isso.", unsupported_tool_claim_detected=True)
    assert "unsupported_tool_claim" in detect_review_reasons(result)


def test_unsupported_entity_claim_without_grounding_flagged() -> None:
    result = _result("A Águas de Coimbra é um cliente importante para nós.")
    assert "unsupported_entity_claim" in detect_review_reasons(result)


def test_unsupported_entity_claim_with_grounding_not_flagged() -> None:
    result = _result(
        "A Águas de Coimbra é um cliente importante para nós.",
        grounding_sources=["PERSISTENT_MEMORY:1"],
    )
    assert "unsupported_entity_claim" not in detect_review_reasons(result)


def test_excessive_enthusiasm_flagged() -> None:
    result = _result("Isso é fantástico!! Que ótima notícia!")
    assert "excessive_enthusiasm" in detect_review_reasons(result)


def test_repeated_question_flagged() -> None:
    result = _result("Qual foi o valor que ele referiu para o protocolo?")
    prior = ["Não me lembro, qual foi o valor que ele referiu para o protocolo?"]
    assert "question_already_answered_in_history" in detect_review_reasons(result, prior)


def test_human_review_never_changes_pass_fail(tmp_path) -> None:
    # Part 6 is explicit: flags are informational, never a gate. This is a
    # documentation-style assertion — detect_review_reasons has no way to
    # affect `passed` because it isn't even called before assertions run.
    result = _result("A Águas de Coimbra é um cliente importante para nós!!!")
    reasons = detect_review_reasons(result)
    assert reasons  # it *is* flagged...
    # ...but nothing about TurnResult itself carries a pass/fail verdict —
    # that lives on TurnEvaluation.passed, set purely from assertions.
    assert not hasattr(result, "passed")
