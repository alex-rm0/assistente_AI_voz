"""Stabilization battery scenario 11 follow-up: the acknowledgement of the
FIRST stated criterion said "a bateria também conta" / "the battery
matters too", although nothing else had been stated yet. "também"/"too"
is only right once another criterion already exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry

ALSO_WORDS = ("também", "tambem", " too")


class _FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta inventada"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def _engine(tmp_path: Path, response_language: str) -> AssistantEngine:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    engine = AssistantEngine(
        llm=_FakeLLM(),
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=None),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )
    engine.response_language = response_language
    engine.respond("Ajuda-me a comparar dois portáteis")
    return engine


def _says_also(text: str) -> bool:
    lowered = f" {text.lower()}"
    return any(word in lowered for word in ALSO_WORDS)


@pytest.mark.parametrize("response_language", ["pt-PT", "en-GB"])
@pytest.mark.parametrize("message", ["o critério é a bateria", "quero decidir pela autonomia"])
def test_first_criterion_ack_does_not_say_also(tmp_path: Path, response_language: str, message: str) -> None:
    engine = _engine(tmp_path, response_language)

    answer = engine.respond(message)

    assert "bateria" in answer.lower() or "battery" in answer.lower(), answer
    assert not _says_also(answer), answer


@pytest.mark.parametrize("response_language", ["pt-PT", "en-GB"])
def test_battery_after_another_criterion_may_say_also(tmp_path: Path, response_language: str) -> None:
    engine = _engine(tmp_path, response_language)
    engine.respond("critério: preço")

    answer = engine.respond("o critério é a bateria")

    assert "bateria" in answer.lower() or "battery" in answer.lower(), answer
    assert _says_also(answer), answer
