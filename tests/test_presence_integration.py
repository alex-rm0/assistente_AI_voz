from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.presence_manager import PresenceManager, PresenceState
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


class FakeLongTermMemory:
    def context_for(self, message: str) -> str:
        return "contexto"

    def remember(self, content: str, category: str = "contexto") -> str:
        return "lembrado"

    def forget(self, query: str) -> str:
        return "esquecido"

    def answer_about(self, query: str) -> str:
        return "nada"

    def search(self, query: str, limit: int = 10):
        return []


class FakeLLM:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.last_system_prompt = ""

    def choose_tool(self, user_message, tools_description, profile_name=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.last_system_prompt = system_prompt or ""
        return "resposta"


def make_engine(tmp_path: Path, presence: PresenceManager, memory: FakeMemory) -> AssistantEngine:
    return AssistantEngine(
        llm=FakeLLM(),
        memory=memory,
        long_term_memory=FakeLongTermMemory(),
        tools=ToolRegistry(),
        workspace_path=tmp_path / "workspace",
        base_system_prompt=get_system_prompt("Geral"),
        presence_manager=presence,
    )


def test_private_mode_does_not_store_conversation(tmp_path: Path) -> None:
    (tmp_path / "workspace").mkdir()
    memory = FakeMemory()
    engine = make_engine(tmp_path, PresenceManager(PresenceState.PRIVATE_MODE), memory)

    response = engine.respond("ola")

    assert response == "Olá! Como estás?"
    assert memory.load() == []


def test_passive_monitoring_does_not_call_agent_or_store_memory(tmp_path: Path) -> None:
    (tmp_path / "workspace").mkdir()
    memory = FakeMemory()
    engine = make_engine(tmp_path, PresenceManager(PresenceState.PASSIVE_MONITORING), memory)

    response = engine.respond("ola")

    assert "PASSIVE_MONITORING" in response
    assert memory.load() == []


def test_active_conversation_passes_memory_context_to_agent(tmp_path: Path) -> None:
    (tmp_path / "workspace").mkdir()
    memory = FakeMemory()
    presence = PresenceManager(PresenceState.ACTIVE_CONVERSATION)
    engine = make_engine(tmp_path, presence, memory)

    response = engine.respond("Ajuda-me a pensar numa arquitetura melhor para este assistente.")

    assert response == "resposta"
    assert "Memoria permanente relevante" in engine.llm.last_system_prompt
    assert "contexto" in engine.llm.last_system_prompt
