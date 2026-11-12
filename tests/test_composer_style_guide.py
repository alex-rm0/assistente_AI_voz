"""R3 step 4, point 2: the composer carries a Jarvis style guide in English and
Portuguese. The model is never invited to describe its own systems: that only
comes from measured health (assistant/system_health.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.response_composer import ComposerRequest, ResponseComposer
from assistant.tool_registry import ToolRegistry


class _CapturingLLM:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        self.system_prompts.append(system_prompt or "")
        return "Noted."

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "none"}

    def embed(self, text: str):
        return None


def _compose(language: str) -> str:
    llm = _CapturingLLM()
    ResponseComposer(llm).compose(ComposerRequest(intent="conversa", user_message="Hoje estou cansado.", response_language=language))
    return llm.system_prompts[-1]


def _regenerate(language: str) -> str:
    llm = _CapturingLLM()
    ResponseComposer(llm).regenerate("Hoje estou cansado.", [], "emotion_inversion", response_language=language)
    return llm.system_prompts[-1]


@pytest.mark.parametrize("build", [_compose, _regenerate])
def test_english_prompt_has_the_jarvis_style_guide(build) -> None:
    prompt = build("en-GB")

    assert "Jarvis" in prompt
    for trait in ("polite", "slightly formal", "dry", "Short sentences"):
        assert trait in prompt, trait
    assert '"Sir"' in prompt and "very occasionally" in prompt
    assert "Never describe your own systems" in prompt


@pytest.mark.parametrize("build", [_compose, _regenerate])
def test_portuguese_prompt_has_the_jarvis_style_guide_without_sir(build) -> None:
    prompt = build("pt-PT")

    assert "Jarvis" in prompt
    for trait in ("educado", "ligeiramente formal", "humor seco", "Frases curtas"):
        assert trait in prompt, trait
    assert "Nunca descrevas os teus sistemas" in prompt
    # "Sir" is an en-GB touch only: the Portuguese guide forbids it rather than inviting it.
    assert "very occasionally" not in prompt
    assert 'Nunca uses "Sir"' in prompt


def test_engine_no_longer_invites_generic_machine_talk(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (tmp_path / "workspace").mkdir()
    llm = _CapturingLLM()
    memory = LongTermMemory(data, embedder=None)
    memory.set_preference("response_language", "en-GB")
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory,
        tools=ToolRegistry(),
        workspace_path=tmp_path / "workspace",
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )

    for message in ("Hoje estou mesmo cansado.", "I passed my exam today.", "Thanks, that's all for now."):
        engine.respond(message)

    assert llm.system_prompts, "the composer was never called"
    for prompt in llm.system_prompts:
        lowered = prompt.lower()
        assert "machine-like" not in lowered
        assert "systems online" not in lowered and "circuits awake" not in lowered
