"""Presence question discipline reached the composer only as a system-message
line, while the user message ended with "Continue the conversation naturally
as Echo." -- the position llama3.1:8b follows most -- so 7 of 9 real
GENERAL_CONVERSATION replies still ended with a question. When the
discipline applies, that closing line is replaced by a short note next to
the user's text; the language anchor stays the very last line.
"""
from __future__ import annotations

import pytest

from assistant.response_composer import ComposerRequest, ResponseComposer
from assistant.response_language import localize

CONTINUE = {"en-GB": "Continue the conversation naturally as Echo.", "pt-PT": "Continua naturalmente a conversa como Echo."}
NOTE = {
    "en-GB": "(Answer without a follow-up question unless one is needed.)",
    "pt-PT": "(Responde sem pergunta de seguimento, a não ser que seja necessária.)",
}
LANGUAGE_ANCHOR = "(Reply in English.)"


class _CapturingLLM:
    def __init__(self) -> None:
        self.user_messages: list[str] = []

    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        self.user_messages.append(user_message)
        return "Ok."


def _payload(language: str, *, discipline: bool) -> str:
    llm = _CapturingLLM()
    ResponseComposer(llm).compose(
        ComposerRequest(
            intent="conversa_normal",
            user_message="Para já estou satisfeito, a interação está muito melhor.",
            response_language=language,
            question_discipline_instruction=(
                localize("composer.question_discipline.suppress_optional_followup", language) if discipline else ""
            ),
        )
    )
    return llm.user_messages[-1]


@pytest.mark.parametrize("language", ["en-GB", "pt-PT"])
def test_discipline_note_replaces_continue_line(language: str) -> None:
    payload = _payload(language, discipline=True)

    assert CONTINUE[language] not in payload
    assert NOTE[language] in payload


def test_note_order_discipline_then_language_anchor_last() -> None:
    lines = [line for line in _payload("en-GB", discipline=True).splitlines() if line.strip()]

    assert lines[-2:] == [NOTE["en-GB"], LANGUAGE_ANCHOR]


def test_pt_pt_discipline_note_is_the_last_line() -> None:
    lines = [line for line in _payload("pt-PT", discipline=True).splitlines() if line.strip()]

    assert lines[-1] == NOTE["pt-PT"]


@pytest.mark.parametrize("language", ["en-GB", "pt-PT"])
def test_payload_unchanged_without_discipline(language: str) -> None:
    payload = _payload(language, discipline=False)

    assert NOTE[language] not in payload
    expected_tail = [CONTINUE[language], LANGUAGE_ANCHOR] if language == "en-GB" else [CONTINUE[language]]
    lines = [line for line in payload.splitlines() if line.strip()]
    assert lines[-len(expected_tail):] == expected_tail
