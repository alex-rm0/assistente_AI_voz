from __future__ import annotations

from datetime import date
from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta"

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path) -> AssistantEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "data"
    llm = FakeLLM()
    memory = LongTermMemory(data, embedder=llm)
    memory.create_task("rever o projeto AssistenteIA", due_date=date.today(), project="AssistenteIA")
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory,
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


def test_startup_greeting_includes_one_proactive_suggestion(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    first = engine.startup_greeting()
    second = engine.startup_greeting()

    assert "Sugestão:" in first
    assert "AssistenteIA" in first
    assert "Sugestão:" not in second


def test_user_can_ask_for_new_suggestion(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    response = engine.respond("Tens alguma sugestão?")

    assert response.startswith("Sugestão:")
