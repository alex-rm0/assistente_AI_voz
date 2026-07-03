from __future__ import annotations

from pathlib import Path

from assistant.context_observer import ContextSnapshot
from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self) -> None:
        self.chat_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        return "resposta"

    def embed(self, text: str):
        return None


class FakeObserver:
    def latest_snapshot(self):
        return ContextSnapshot(
            active_app="Code.exe",
            active_window="AssistenteIA - Visual Studio Code",
            current_project="AssistenteIA",
            observed_at=100.0,
        )

    def latest_summary(self):
        return None

    def activity_summary(self, limit: int = 10):
        return []


def make_engine(tmp_path: Path) -> tuple[AssistantEngine, FakeLLM]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "data"
    llm = FakeLLM()
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        context_observer=FakeObserver(),
    )
    return engine, llm


def test_where_did_we_stop_uses_session_continuity(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    response = engine.respond("onde ficámos?")

    assert "fic" in response.lower()
    assert "AssistenteIA" in response
    assert llm.chat_calls == 0


def test_last_active_project_question_uses_briefing_layer(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    response = engine.respond("qual foi o último projeto ativo?")

    assert response == "Último projeto ativo observado: AssistenteIA"
    assert llm.chat_calls == 0
