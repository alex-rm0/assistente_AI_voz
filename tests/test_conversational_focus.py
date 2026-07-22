from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.personal_model import PersonalModel
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FocusLLM:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.choose_tool_calls = 0
        self.embed_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        self.choose_tool_calls += 1
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        if "Boa ideia. Antes de" in user_message or "preferência principal" in user_message:
            return "Antes de organizarmos isso, queres uma coisa simples e tranquila ou algo mais combinado ao detalhe?"
        return "Resposta direta."

    def embed(self, text: str):
        self.embed_calls += 1
        return None


def make_engine(tmp_path: Path) -> tuple[AssistantEngine, FocusLLM]:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FocusLLM()
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        personal_model=PersonalModel(data),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )
    return engine, llm


def assert_short_focused_answer(answer: str) -> None:
    lowered = answer.lower()
    assert answer.count("?") <= 1
    assert sum(answer.count(mark) for mark in ".!?") <= 2
    assert "você" not in lowered
    assert "sua" not in lowered
    assert "seus" not in lowered
    assert "estou aqui para ajudar" not in lowered
    assert "ferramenta" not in lowered


def test_arrived_home_work_resistance_asks_low_effort_question(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    answer = engine.respond("Cheguei agora a casa, estava a pensar em ir trabalhar mas não sei se tenho muita vontade.")

    assert answer == "O que te está a travar: cansaço ou falta de vontade?"
    assert_short_focused_answer(answer)
    assert "cheguei" not in answer.lower()
    assert "casa" not in answer.lower()
    assert "sentir falta" not in answer.lower()
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0


def test_document_restart_resistance_focuses_on_restart_not_content(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    answer = engine.respond("Tenho um documento quase pronto, mas não me apetece voltar a pegar nele.")

    assert answer == "O verdadeiro obstáculo parece ser voltares a pegar nele, não o documento em si."
    assert_short_focused_answer(answer)
    assert "rever o documento" not in answer.lower()
    assert llm.chat_calls == 0


def test_overloaded_day_focuses_on_cognitive_load(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    answer = engine.respond("Hoje tive reuniões, fui ao treino e ainda queria estudar, mas já não consigo pensar.")

    assert answer == "Parece mais excesso de carga do que falta de vontade."
    assert_short_focused_answer(answer)
    lowered = answer.lower()
    assert "reuniões" not in lowered
    assert "treino" not in lowered
    assert "estudar" not in lowered
    assert llm.chat_calls == 0


def test_casual_beach_share_does_not_start_planning(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    answer = engine.respond("Vou à praia com amigos no fim de semana.")

    assert answer == "Parece um bom fim de semana. Praia e amigos combinam bem."
    assert_short_focused_answer(answer)
    assert "plano" not in answer.lower()
    assert "organizar" not in answer.lower()
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0


def test_explicit_beach_planning_asks_relevant_context(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    answer = engine.respond("Ajuda-me a organizar uma ida à praia com amigos.")

    assert "simples e tranquila" in answer
    assert "detalhe" in answer
    assert_short_focused_answer(answer)
    assert llm.chat_calls == 1
    assert llm.choose_tool_calls == 0
