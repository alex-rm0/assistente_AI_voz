from __future__ import annotations

from pathlib import Path

from assistant.agent import Agent, system_state_tool_intent
from assistant.conversation import AssistantEngine, _normalize_text
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self) -> None:
        self.chat_prompts: list[str] = []

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_prompts.append(user_message)
        return "resposta direta"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("get_recent_activity", "Mostra atividade recente.", ("read:context_observer",))(
        lambda **kwargs: "atividade recente: nada"
    )
    return registry


def make_context():
    from assistant.agent import AgentContext

    return AgentContext(system_prompt="", history=[], active_contexts=[], tools_enabled=True)


def make_engine(tmp_path: Path) -> AssistantEngine:
    llm = FakeLLM()
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=make_registry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


# --- Part 3: no evidence span -> no tool ------------------------------------


def test_arm_wrestling_message_has_no_tool_intent_evidence() -> None:
    text = _normalize_text(
        "So ouvi que queriam aumentar, mas tambem nao vou fazer grande braco de ferro."
    )
    supported, evidence_span, confidence = system_state_tool_intent(text)

    assert supported is False
    assert evidence_span == ""
    assert confidence == 0.0


def test_negotiation_message_never_calls_get_recent_activity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace)

    result = agent.run(
        "So ouvi que queriam aumentar, mas tambem nao vou fazer grande braco de ferro.",
        make_context(),
    )

    assert result.tools_used == ()
    assert "get_recent_activity" not in result.response


def test_explicit_activity_question_has_evidence_and_calls_the_tool(tmp_path: Path) -> None:
    text = _normalize_text("O que estive a fazer no computador?")
    supported, evidence_span, confidence = system_state_tool_intent(text)

    assert supported is True
    assert evidence_span == "o que estive a fazer"
    assert confidence == 1.0

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace)

    result = agent.run("O que estive a fazer no computador?", make_context())

    assert "get_recent_activity" in result.tools_used


def test_recent_activity_on_pc_has_evidence() -> None:
    text = _normalize_text("Atividade recente do computador")
    supported, evidence_span, _confidence = system_state_tool_intent(text)

    assert supported is True
    assert evidence_span == "atividade recente"


def test_negotiate_protocol_question_has_no_tool_evidence() -> None:
    text = _normalize_text("Quanto devo negociar no protocolo?")
    supported, evidence_span, confidence = system_state_tool_intent(text)

    assert supported is False
    assert evidence_span == ""
    assert confidence == 0.0


# --- Falha 5: Composer can never claim real activity without a tool -------


def test_composer_cannot_speculate_about_real_activity_without_a_tool(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    blocked = engine._complete_turn(
        "O que estive a fazer no computador?",
        "Estavas a trabalhar num projeto, se não me engano.",
        "RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert blocked == "Não tenho essa informação sobre a tua atividade real — só o Context Observer sabe isso."


def test_ordinary_composer_reply_is_not_touched_by_the_activity_guard(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    reply = engine._complete_turn(
        "Como está o tempo hoje?",
        "Não tenho acesso a essa informação em tempo real.",
        "RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert reply == "Não tenho acesso a essa informação em tempo real."


def test_activity_claim_guard_does_not_block_when_a_tool_actually_ran(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    reply = engine._complete_turn(
        "O que estive a fazer no computador?",
        "Estavas a trabalhar num projeto, se não me engano.",
        "AGENT_DIRECT",
        selected_path="AGENT",
        tools_used=("get_recent_activity",),
        technical=True,
    )

    assert reply == "Estavas a trabalhar num projeto, se não me engano."
