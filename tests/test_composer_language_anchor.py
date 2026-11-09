"""Policy: response_language is a fixed setting. With en-GB and an English
system prompt (551816b), llama3.1:8b still answered short Portuguese
questions in Portuguese, because the only language instruction sat in the
system message, far from the user's text. With en-GB, compose() and
regenerate() append a short note to the LAST user message of the payload
only; the history passed in and pt-PT stay unchanged.
"""
from __future__ import annotations

from assistant.response_composer import ComposerRequest, ResponseComposer

NOTE = "(Reply in English.)"


class _CapturingLLM:
    def __init__(self) -> None:
        self.user_messages: list[str] = []
        self.histories: list[list[dict[str, str]]] = []

    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        self.user_messages.append(user_message)
        self.histories.append(history)
        return "Paris."


HISTORY = [{"role": "user", "content": "Olá"}, {"role": "assistant", "content": "Hi. How are you?"}]


def _compose(language: str) -> _CapturingLLM:
    llm = _CapturingLLM()
    history = [dict(item) for item in HISTORY]
    ResponseComposer(llm).compose(
        ComposerRequest(
            intent="conversa",
            user_message="Qual é a capital de França?",
            history=history,
            response_language=language,
        )
    )
    assert history == HISTORY
    return llm


def _regenerate(language: str) -> _CapturingLLM:
    llm = _CapturingLLM()
    history = [dict(item) for item in HISTORY]
    ResponseComposer(llm).regenerate("Qual é a capital de França?", history, "emotion_inversion", response_language=language)
    assert history == HISTORY
    return llm


def test_en_gb_compose_anchors_language_at_end_of_user_message() -> None:
    llm = _compose("en-GB")

    assert llm.user_messages[-1].endswith(f"\n\n{NOTE}")
    assert llm.histories[-1] == HISTORY


def test_en_gb_regenerate_anchors_language_at_end_of_user_message() -> None:
    llm = _regenerate("en-GB")

    assert llm.user_messages[-1].endswith(f"\n\n{NOTE}")
    assert llm.histories[-1] == HISTORY


def test_pt_pt_user_message_has_no_language_note() -> None:
    assert NOTE not in _compose("pt-PT").user_messages[-1]
    assert NOTE not in _regenerate("pt-PT").user_messages[-1]
