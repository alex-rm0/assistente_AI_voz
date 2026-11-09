"""KNOWN_ISSUES §1/§7 (decision: response_language is a fixed setting).
With en-GB the composer sent the Portuguese _SYSTEM_PROMPT with the English
instruction appended at the end, and llama3.1:8b answered Portuguese input
in Portuguese. With en-GB the whole system prompt must be English (same
rules); with pt-PT it stays exactly as before.
"""
from __future__ import annotations

from assistant.response_composer import ComposerRequest, ResponseComposer, _SYSTEM_PROMPT
from assistant.response_language import response_language_instruction

PT_MARKERS = ("companheiro digital", "Não inventes factos", "Sê natural e breve")


class _CapturingLLM:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        self.system_prompts.append(system_prompt or "")
        return "Paris."


def _compose(language: str) -> str:
    llm = _CapturingLLM()
    ResponseComposer(llm).compose(
        ComposerRequest(intent="conversa", user_message="Qual é a capital de França?", response_language=language)
    )
    return llm.system_prompts[-1]


def _regenerate(language: str) -> str:
    llm = _CapturingLLM()
    ResponseComposer(llm).regenerate("Qual é a capital de França?", [], "emotion_inversion", response_language=language)
    return llm.system_prompts[-1]


def test_en_gb_compose_system_prompt_is_english() -> None:
    prompt = _compose("en-GB")

    assert not [marker for marker in PT_MARKERS if marker in prompt], prompt
    assert "digital companion" in prompt
    assert "Do not invent facts" in prompt
    assert "Respond in English" in prompt


def test_en_gb_regenerate_system_prompt_is_english() -> None:
    prompt = _regenerate("en-GB")

    assert not [marker for marker in PT_MARKERS if marker in prompt], prompt
    assert "digital companion" in prompt
    assert "Respond in English" in prompt


def test_pt_pt_system_prompts_are_unchanged() -> None:
    expected = f"{_SYSTEM_PROMPT}\n\n{response_language_instruction('pt-PT')}"

    assert _compose("pt-PT") == expected
    assert _regenerate("pt-PT") == expected
