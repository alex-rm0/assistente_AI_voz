from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager, PresenceState
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self) -> None:
        self.last_system_prompt = ""
        self.chat_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        self.last_system_prompt = system_prompt or ""
        return "resposta"

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path, presence: PresenceManager | None = None) -> AssistantEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "data"
    llm = FakeLLM()
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=presence or PresenceManager(),
    )


def test_engine_changes_to_passive_monitoring_from_user_message(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    response = engine.respond("fica só a acompanhar")

    assert response == "Entendido, vou ficar em modo observador."
    assert engine.presence.state == PresenceState.PASSIVE_MONITORING
    assert "PASSIVE_MONITORING" in engine.long_term_memory.current_work_context()


def test_engine_can_return_to_active_from_silent_mode(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, PresenceManager(PresenceState.FOCUS_MODE))

    response = engine.respond("volta")

    assert response == "Entendido, volto ao modo conversa."
    assert engine.presence.state == PresenceState.ACTIVE_CONVERSATION


def test_private_mode_command_changes_state_without_storing_conversation_pair(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    response = engine.respond("isto é privado")

    assert response == "Entendido, vou ficar em modo privado e não vou guardar esta conversa."
    assert engine.presence.state == PresenceState.PRIVATE_MODE
    assert engine.history() == []
    assert "PRIVATE_MODE" in engine.long_term_memory.current_work_context()


def test_offline_command_changes_state_before_normal_response(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    response = engine.respond("desliga-te")

    assert response == "Entendido, vou ficar offline."
    assert engine.presence.state == PresenceState.OFFLINE
    assert engine.llm.chat_calls == 0


def test_wake_commands_work_from_all_silent_states(tmp_path: Path) -> None:
    cases = (
        (PresenceState.PASSIVE_MONITORING, "volta ao modo conversa"),
        (PresenceState.FOCUS_MODE, "modo conversa"),
        (PresenceState.PRIVATE_MODE, "acorda"),
        (PresenceState.OFFLINE, "ativa-te"),
    )

    for index, (initial_state, command) in enumerate(cases):
        engine = make_engine(tmp_path / str(index), PresenceManager(initial_state))

        response = engine.respond(command)

        assert response == "Entendido, volto ao modo conversa."
        assert engine.presence.state == PresenceState.ACTIVE_CONVERSATION
        assert "ACTIVE_CONVERSATION" in engine.long_term_memory.current_work_context()


def test_presence_question_uses_presence_manager_not_llm(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, PresenceManager(PresenceState.FOCUS_MODE))

    response = engine.respond("em que modo estás?")

    assert response.startswith("Estou em FOCUS_MODE.")
    assert engine.llm.chat_calls == 0
