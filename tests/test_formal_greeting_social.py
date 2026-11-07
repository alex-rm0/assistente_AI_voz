"""KNOWN_ISSUES §3: "Olá, como está?" (formal) missed the social fast path
and went to GENERAL_CONVERSATION -> RESPONSE_COMPOSER (the LLM, which once
invented activity). is_pure_social_turn matches an exact set of phrases and
only had the informal "como estás". Questions about something ("como está
o projeto?") must stay out of the social path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.conversation import AssistantEngine, is_pure_social_turn
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry

FORMAL = ("olá, como está?", "bom dia, como está?", "como vai?", "hello, how are you?")
NOT_SOCIAL = ("como está o projeto?", "como está o tempo?")


class _FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta inventada"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def _engine(tmp_path: Path) -> AssistantEngine:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AssistantEngine(
        llm=_FakeLLM(),
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=None),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


@pytest.mark.parametrize("message", FORMAL)
def test_formal_greeting_is_social(message: str) -> None:
    assert is_pure_social_turn(message)


@pytest.mark.parametrize("message", FORMAL)
def test_formal_greeting_takes_social_fast_path(tmp_path: Path, message: str) -> None:
    engine = _engine(tmp_path)

    answer = engine.respond(message)

    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("selected_path") == "SOCIAL_PATH", telemetry.get("selected_path")
    assert "resposta inventada" not in answer


@pytest.mark.parametrize("message", NOT_SOCIAL)
def test_questions_about_something_are_not_social(message: str) -> None:
    assert not is_pure_social_turn(message)
