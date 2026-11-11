"""A vague action request with nothing it can refer to ("Marca isso",
"Send it") gets a fixed, localized clarifying question -- no LLM call. With
the composer the model asked generic questions or claimed it had done the
action ('Ok, vou "marcar" a nossa conversa.'). Comments with a vague
reference ("Faz sentido isso") still go to the composer.
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

QUESTIONS = {
    "pt-PT": {
        "Marca isso": "O que queres que marque?",
        "Envia isso": "O que queres que envie?",
        "Guarda isto": "O que queres que guarde?",
        "Abre aquilo": "O que queres que abra?",
        "Faz isso": "O que queres que faça?",
        "Mark that": "O que queres que marque?",
        "Send it": "O que queres que envie?",
        "Copia isso": "A que te referes?",
    },
    "en-GB": {
        "Marca isso": "What would you like me to mark?",
        "Envia isso": "What would you like me to send?",
        "Guarda isto": "What would you like me to save?",
        "Abre aquilo": "What would you like me to open?",
        "Faz isso": "What would you like me to do?",
        "Mark that": "What would you like me to mark?",
        "Send it": "What would you like me to send?",
        "Copia isso": "What are you referring to?",
    },
}
COMMENTS = ("Isso é interessante", "Faz sentido isso", "Gosto disso", "That makes sense", "Copy that")


class _CountingLLM:
    def __init__(self) -> None:
        self.chat_calls = 0

    def chat(self, user_message, history=None, system_prompt=None, response_format=None, **kwargs):
        self.chat_calls += 1
        return "Resposta do composer."

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def _engine(tmp_path: Path, language: str) -> tuple[AssistantEngine, _CountingLLM]:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    llm = _CountingLLM()
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=None),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )
    engine.response_language = language
    return engine, llm


@pytest.mark.parametrize("language", ["pt-PT", "en-GB"])
@pytest.mark.parametrize("message", list(QUESTIONS["pt-PT"]))
def test_vague_action_request_gets_fixed_question_without_llm(tmp_path: Path, language: str, message: str) -> None:
    engine, llm = _engine(tmp_path, language)

    answer = engine.respond(message)

    assert answer == QUESTIONS[language][message]
    assert llm.chat_calls == 0
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("selected_path") == "VAGUE_ACTION_CLARIFICATION"
    assert telemetry.get("llm_calls") == 0


@pytest.mark.parametrize("message", COMMENTS)
def test_comments_still_go_to_the_composer(tmp_path: Path, message: str) -> None:
    engine, llm = _engine(tmp_path, "pt-PT")

    engine.respond(message)

    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("selected_path") != "VAGUE_ACTION_CLARIFICATION"
