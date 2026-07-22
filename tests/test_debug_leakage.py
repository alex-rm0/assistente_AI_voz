from __future__ import annotations

from pathlib import Path

import assistant.conversation as conversation_module
from assistant.agent import Agent, AgentContext
from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry

_FORBIDDEN_MARKERS = ("[DEBUG_AGENT]", "Passos internos", "Razao da escolha", "Razão da escolha")


class FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta direta"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("get_recent_activity", "Mostra atividade recente.", ("read:context_observer",))(
        lambda **kwargs: "Atividade recente: VS Code, Chrome."
    )
    return registry


def make_context() -> AgentContext:
    return AgentContext(system_prompt="", history=[], active_contexts=[], tools_enabled=True)


def _assert_clean(public_response: str) -> None:
    for marker in _FORBIDDEN_MARKERS:
        assert marker not in public_response, f"marcador de debug '{marker}' vazou para a resposta publica"


# --- Part 4: debug never reaches the public response ------------------------


def test_agent_result_response_never_contains_debug_markers_with_tool_use(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = Agent(FakeLLM(), make_registry(), workspace, debug_agent=True)

    result = agent.run("Atividade recente do computador", make_context())

    _assert_clean(result.response)
    # The trace still exists — it just lives in a separate field.
    assert "[DEBUG_AGENT]" in result.debug_trace
    assert "Ferramenta escolhida" in result.debug_trace


def test_agent_debug_trace_is_empty_when_debug_agent_is_off(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = Agent(FakeLLM(), make_registry(), workspace, debug_agent=False)

    result = agent.run("Atividade recente do computador", make_context())

    _assert_clean(result.response)
    assert result.debug_trace == ""


def make_engine(tmp_path: Path, debug_agent: bool) -> AssistantEngine:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    llm = FakeLLM()
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=make_registry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        debug_agent=debug_agent,
        debug_ollama_payload=True,
    )


def _force_agent_path(monkeypatch) -> None:
    # Ordinary phrasing for "atividade recente" gets answered by the
    # Composer shortcut before ever reaching self.agent.run() (see Part 1's
    # _should_answer_without_agent). Forcing every message through the agent
    # branch is the only reliable way to exercise Agent.run()'s tool-using
    # path (and its debug_trace) from the full engine, rather than betting
    # on a phrasing that happens to survive every earlier deterministic
    # shortcut.
    monkeypatch.setattr(conversation_module, "_should_answer_without_agent", lambda strategy: False)


def test_engine_response_never_leaks_debug_even_with_debug_flags_on(tmp_path: Path, monkeypatch, capsys) -> None:
    engine = make_engine(tmp_path, debug_agent=True)
    _force_agent_path(monkeypatch)

    answer = engine.respond("Atividade recente do computador")

    _assert_clean(answer)
    # The trace must still be observable somewhere (terminal, in this case)
    # — Part 4 says never in the UI, not never at all.
    captured = capsys.readouterr()
    assert "[DEBUG_AGENT]" in captured.out


def test_engine_telemetry_carries_the_debug_trace_separately(tmp_path: Path, monkeypatch) -> None:
    engine = make_engine(tmp_path, debug_agent=True)
    _force_agent_path(monkeypatch)

    answer = engine.respond("Atividade recente do computador")
    telemetry = engine.get_last_turn_telemetry() or {}

    _assert_clean(answer)
    assert "[DEBUG_AGENT]" in telemetry.get("agent_debug_trace", "")
