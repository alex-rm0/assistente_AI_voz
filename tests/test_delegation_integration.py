from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeMemory:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def append_pair(self, user_message: str, response: str) -> None:
        self.items.append({"role": "user", "content": user_message})
        self.items.append({"role": "assistant", "content": response})

    def load(self) -> list[dict[str, str]]:
        return list(self.items)

    def clear(self) -> None:
        self.items.clear()


class FakeLLM:
    def __init__(self) -> None:
        self.system_prompt = ""

    def choose_tool(self, user_message, tools_description, profile_name=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta local"

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path) -> AssistantEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    return AssistantEngine(
        llm=llm,
        memory=FakeMemory(),
        long_term_memory=LongTermMemory(tmp_path / "data", embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_system_prompt("Programacao"),
        active_profile_name="Programacao",
        presence_manager=PresenceManager(),
    )


def test_engine_prepares_codex_prompt(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    response = engine.respond("Implementa uma nova funcionalidade neste projeto.")

    assert "melhor resolvido pelo Codex" in response
    assert "Vou preparar o contexto" in response
    assert "Pedido original:" in response


def test_engine_keeps_simple_request_local(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    response = engine.respond("Ola")

    assert response == "Olá! Como estás?"
