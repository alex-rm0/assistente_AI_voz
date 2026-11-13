"""The complexity rules that send a non-conversational request to Claude used
to know only Portuguese phrasing. With en-GB as the default language they have
to recognise the same requests in English.
"""
from __future__ import annotations

import pytest

from assistant.model_router import AutomaticRoutingConfig, ModelRouter, ModelRoutingConfig, ModelRoutingInput


def _decide(message: str, *, prompt_chars: int = 0):
    router = ModelRouter(
        ModelRoutingConfig(mode="automatic", automatic=AutomaticRoutingConfig(claude_enabled=True)),
        env={}, anthropic_key_available=lambda: True, paid_calls_allowed=lambda: True,
    )
    # Not a composer source: plain conversation goes to Claude anyway, so this isolates the complexity rules.
    return router.decide(ModelRoutingInput(source="AGENT_FINAL_RESPONSE", user_message=message, prompt_chars=prompt_chars, privacy_class="PERSONAL"))


@pytest.mark.parametrize("message,reason", [
    ("Compare these two laptops for me.", "complex_request"),
    ("Can you give me a comparison of the two options?", "complex_request"),
    ("Explain in detail how the routing works.", "complex_request"),
    ("Walk me through it step by step.", "complex_request"),
    ("Summarise this document.", "complex_request"),
    ("Summarize this document for me, please.", "complex_request"),
    ("Rewrite this paragraph so it reads better.", "complex_request"),
    ("Give me a detailed plan for the week.", "complex_planning"),
    ("Plan my training step by step for the next ten days.", "complex_planning"),
    ("Summarise this in five bullet points: the project started in March and", "structured_summary"),
    ("Give me a summary with the main ideas: first, the budget was cut; second,", "structured_summary"),
    ("Write a detailed professional email about the current state and the next steps.", "professional_writing"),
    ("Draft a detailed report on our progress.", "professional_writing"),
    ("Explain this Python error to me.", "technical_explanation"),
    ("Can you explain what this git command does?", "technical_explanation"),
])
def test_english_requests_reach_claude_for_the_same_reasons(message: str, reason: str) -> None:
    decision = _decide(message)

    assert (decision.provider, decision.reason_code) == ("anthropic", reason)


@pytest.mark.parametrize("message", [
    "What time is it?", "Thanks, that's all.", "Open the browser.", "Tell me a joke.", "Is the meeting today?",
])
def test_simple_english_requests_stay_local(message: str) -> None:
    decision = _decide(message)

    assert (decision.provider, decision.reason_code) == ("ollama", "low_complexity")


def test_portuguese_requests_are_unchanged() -> None:
    assert _decide("Compara estes dois portáteis.").reason_code == "complex_request"
    assert _decide("Faz um plano detalhado para estudar.").reason_code == "complex_planning"
    assert _decide("Que horas são?").reason_code == "low_complexity"
