from __future__ import annotations

from pathlib import Path

from assistant.context_observer import ContextObserver, ContextSnapshot
from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeMemory:
    def append_pair(self, user_message: str, response: str) -> None:
        pass

    def load(self) -> list[dict[str, str]]:
        return []

    def clear(self) -> None:
        pass


class FakeLLM:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.last_system_prompt = ""

    def choose_tool(self, user_message, tools_description, profile_name=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.last_system_prompt = system_prompt or ""
        return "resposta"

    def embed(self, text: str):
        return None


def test_observed_context_is_passed_to_agent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_path = tmp_path / "data"
    llm = FakeLLM()
    observer = ContextObserver(
        data_path=data_path,
        project_root=tmp_path / "assistenteIA",
        snapshot_provider=lambda: ContextSnapshot(
            active_app="Code.exe",
            active_window="assistenteIA - Visual Studio Code",
            recent_files=("nota.txt",),
            observed_at=100.0,
        ),
    )
    observer.observe_once()
    engine = AssistantEngine(
        llm=llm,
        memory=FakeMemory(),
        long_term_memory=LongTermMemory(data_path, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_system_prompt("Geral"),
        presence_manager=PresenceManager(),
        context_observer=observer,
    )

    engine.respond("ola")

    assert "contexto_observado" in llm.last_system_prompt
    assert "Code.exe" in llm.last_system_prompt
    assert "nota.txt" in llm.last_system_prompt


def test_available_information_summary_is_internal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nota.txt").write_text("nota", encoding="utf-8")
    data_path = tmp_path / "data"
    llm = FakeLLM()
    observer = ContextObserver(
        data_path=data_path,
        project_root=tmp_path / "assistenteIA",
        snapshot_provider=lambda: ContextSnapshot(
            active_app="Code.exe",
            active_window="assistenteIA - Visual Studio Code",
            recent_files=("nota.txt",),
            observed_at=100.0,
        ),
    )
    observer.observe_once()
    engine = AssistantEngine(
        llm=llm,
        memory=FakeMemory(),
        long_term_memory=LongTermMemory(data_path, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_system_prompt("Geral"),
        presence_manager=PresenceManager(),
        context_observer=observer,
    )

    response = engine.respond("que informacao tens?")

    assert "Memória permanente:" in response
    assert "Tarefas:" in response
    assert "Timeline:" in response
    assert "Ficheiros da workspace:" in response
    assert "Contexto observado do computador:" in response
    assert llm.last_system_prompt == ""
