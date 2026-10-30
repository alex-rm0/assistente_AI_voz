"""Stabilization battery scenario 13: "Que janela tenho aberta?" went to
GENERAL_CONVERSATION with tools_used=[] and the model invented an answer.

Cause: assistant.agent's deterministic system-state markers did not cover
the singular "que janela ... aberta" framing nor any English phrasing, so
system_state_tool_intent() said False and the turn never reached the
Context Observer tools. These go through the real engine (no forced agent
path, no LLM routing) and assert a Context Observer tool actually ran.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.agent import system_state_tool_intent
from assistant.conversation import AssistantEngine, _normalize_text
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry

CONTEXT_OBSERVER_TOOLS = (
    "get_current_activity_summary",
    "get_active_window",
    "get_active_application",
    "get_open_windows",
    "get_recent_activity",
    "get_last_context_snapshot",
)

POSITIVE = (
    "que janela tenho aberta?",
    "que janela está aberta?",
    "o que tenho aberto?",
    "que aplicação estou a usar?",
    "que programas tenho abertos?",
    "what window is open?",
    "what am I looking at?",
    "which app am I using?",
)

NEGATIVE = (
    "abre uma janela nova no browser",
    "a janela do quarto está aberta",
    "fecha a janela",
)


class _FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta inventada"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name in CONTEXT_OBSERVER_TOOLS:
        registry.register(name, f"Context Observer: {name}.", ("read:context_observer",))(
            lambda _name=name, **kwargs: f"Resultado real de {_name}: Visual Studio Code."
        )
    return registry


def _engine(tmp_path: Path) -> AssistantEngine:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    llm = _FakeLLM()
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=_registry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


def _tools_used(engine: AssistantEngine) -> list[str]:
    telemetry = engine.get_last_turn_telemetry() or {}
    return [str(tool) for tool in telemetry.get("tools_used", [])]


@pytest.mark.parametrize("phrase", POSITIVE)
def test_window_and_app_questions_have_tool_intent(phrase: str) -> None:
    supported, evidence, _confidence = system_state_tool_intent(_normalize_text(phrase))
    assert supported, phrase
    assert evidence


@pytest.mark.parametrize("phrase", POSITIVE)
def test_window_and_app_questions_use_context_observer_tool(tmp_path: Path, phrase: str) -> None:
    engine = _engine(tmp_path)

    answer = engine.respond(phrase)

    used = _tools_used(engine)
    assert any(tool in CONTEXT_OBSERVER_TOOLS for tool in used), (phrase, used, answer)
    assert "resposta inventada" not in answer


@pytest.mark.parametrize("phrase", NEGATIVE)
def test_unrelated_window_sentences_do_not_route_to_context_observer(phrase: str) -> None:
    supported, _evidence, _confidence = system_state_tool_intent(_normalize_text(phrase))
    assert not supported, phrase
