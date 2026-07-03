from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self) -> None:
        self.last_system_prompt = ""

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.last_system_prompt = system_prompt or ""
        return "resposta"

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path) -> AssistantEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    data = tmp_path / "data"
    llm = FakeLLM()
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


def test_language_policy_answer_comes_from_internal_state(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    response = engine.respond("Qual é a tua língua base?")

    assert response == "A minha língua base é português de Portugal, mas posso falar inglês quando pedires."
    assert engine.llm.last_system_prompt == ""


def test_user_can_switch_current_language_preference(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    english_response = engine.respond("Prefiro inglês.")

    assert "base language remains Portuguese from Portugal" in english_response
    assert engine.long_term_memory.get_preference("idioma_base") == "pt-PT"
    assert engine.long_term_memory.get_preference("idioma_atual") == "en"

    portuguese_response = engine.respond("Volta ao português de Portugal.")

    assert portuguese_response == "Claro. Volto ao português de Portugal."
    assert engine.long_term_memory.get_preference("idioma_atual") == "pt-PT"


def test_language_instruction_is_sent_to_llm(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("Olá")

    assert "idioma_base = pt-PT" in engine.llm.last_system_prompt
    assert "idioma_atual = pt-PT" in engine.llm.last_system_prompt
    assert "aplicações" in engine.llm.last_system_prompt
    assert "ficheiros" in engine.llm.last_system_prompt
