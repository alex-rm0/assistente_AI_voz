"""KNOWN_ISSUES §1 (decision: response_language is a fixed setting). The
final-response guards in AssistantEngine._complete_turn replaced the reply
with Portuguese literals regardless of response_language; only some were
caught afterwards by localize_deterministic_response, and the activity
guard exposed the internal name "Context Observer". Each guard must answer
in the configured language and never mention internal component names.
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


class _FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


# guard -> (user message, generated reply, source, selected_path, blocked source, {language: expected text})
GUARDS = {
    "tool_claim": (
        "Pesquisa voos para Roma",
        "Pesquisei e encontrei três voos baratos.",
        "RESPONSE_COMPOSER",
        "GENERAL_CONVERSATION",
        "TOOL_CLAIM_BLOCKED",
        {
            "pt-PT": "Ainda não consegui fazer essa pesquisa — não tenho essa ferramenta ligada a esta conversa.",
            "en-GB": "I still can't do that research. I don't have that tool connected to this conversation.",
        },
    ),
    "memory_recall_claim": (
        "Lembras-te do exame?",
        "Sim, lembro-me bem do exame.",
        "RESPONSE_COMPOSER",
        "CONVERSATIONAL_REFINEMENT",
        "MEMORY_CLAIM_BLOCKED",
        {
            "pt-PT": "Não tenho essa informação guardada sobre isso.",
            "en-GB": "I don't have that information saved.",
        },
    ),
    "memory_claim": (
        "Gosto de café de manhã.",
        "Tenho guardado que gostas de café.",
        "RESPONSE_COMPOSER",
        "GENERAL_CONVERSATION",
        "MEMORY_CLAIM_BLOCKED",
        {
            "pt-PT": "Não quero inventar memória. Posso consultar a memória real se quiseres.",
            "en-GB": "I don't want to make up a memory. I can check the real memory if you'd like.",
        },
    ),
    "historical_duration_claim": (
        "Trabalho no Echo.",
        "Trabalhas no Echo há três semanas.",
        "RESPONSE_COMPOSER",
        "GENERAL_CONVERSATION",
        "MEMORY_CLAIM_BLOCKED",
        {
            "pt-PT": "Não tenho memória suficiente para confirmar essa duração.",
            "en-GB": "I don't have enough memory to confirm that duration.",
        },
    ),
    "activity_claim": (
        "O que estive a fazer no computador?",
        "Estavas a trabalhar num projeto, se não me engano.",
        "RESPONSE_COMPOSER",
        "GENERAL_CONVERSATION",
        "ACTIVITY_CLAIM_BLOCKED",
        {
            "pt-PT": "Não sei o que estiveste a fazer. Posso dizer-te que janela tens aberta agora, se me perguntares.",
            "en-GB": "I don't know what you were doing. I can tell you which window you have open right now if you ask me.",
        },
    ),
    "workspace_claim": (
        "abre samasung",
        "Samsung está a abrir agora.",
        "AGENT_DIRECT",
        "AGENT",
        "WORKSPACE_CLAIM_BLOCKED",
        {
            "pt-PT": "Não alterei nada no workspace. Preciso de identificar um objeto real antes de agir.",
            "en-GB": "I didn't change anything in the workspace. I need to identify a real object before acting.",
        },
    ),
}


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
    return engine


@pytest.mark.parametrize("response_language", ["pt-PT", "en-GB"])
@pytest.mark.parametrize("guard", sorted(GUARDS))
def test_guard_message_follows_response_language(tmp_path: Path, guard: str, response_language: str) -> None:
    # Each expected text is unique to its guard, so it also identifies which
    # guard fired (the turn trace is cleared when _complete_turn returns).
    user, reply, source, selected_path, _blocked_source, expected = GUARDS[guard]
    engine = _engine(tmp_path, response_language)

    response = engine._complete_turn(user, reply, source, selected_path=selected_path)

    assert response == expected[response_language]
    assert "Context Observer" not in response
