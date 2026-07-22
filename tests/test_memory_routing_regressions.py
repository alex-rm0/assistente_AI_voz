from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.memory_recall import extract_academic_event_candidate
from assistant.presence_manager import PresenceManager
from assistant.tool_registry import ToolRegistry


class RoutingLLM:
    system_prompt = ""

    def __init__(self) -> None:
        self.chat_call_count = 0

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_call_count += 1
        return "Estratégias Algorítmicas estuda formas eficientes de resolver problemas."

    def choose_tool(self, *args, **kwargs):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, *args, **kwargs):
        return None


def make_engine(tmp_path: Path) -> tuple[AssistantEngine, RoutingLLM, LongTermMemory]:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = RoutingLLM()
    memory = LongTermMemory(data, embedder=llm)
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory,
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt="",
        presence_manager=PresenceManager(),
    )
    return engine, llm, memory


def seed_exam(memory: LongTermMemory):
    fact, _action, _reason = memory.remember_structured_fact_with_trace(
        "academic_event",
        {
            "event": "exame",
            "discipline": "Estratégias Algorítmicas",
            "date_reference": "para a semana",
            "status": "upcoming",
        },
        confidence=0.9,
        source="user_statement",
    )
    return fact


def test_general_knowledge_query_is_not_memory_command(tmp_path: Path) -> None:
    engine, llm, _memory = make_engine(tmp_path)

    response = engine.respond("O que sabes sobre Estratégias Algorítmicas?")

    assert "Estratégias Algorítmicas" in response
    assert engine._last_selected_path == "GENERAL_KNOWLEDGE_QUERY"
    assert llm.chat_call_count == 1


def test_memory_inventory_uses_persistent_data_without_llm(tmp_path: Path) -> None:
    engine, llm, memory = make_engine(tmp_path)
    seed_exam(memory)

    response = engine.respond("O que tens na memória?")

    assert "exame de Estratégias Algorítmicas" in response
    assert engine._last_selected_path == "MEMORY_INVENTORY"
    assert llm.chat_call_count == 0


def test_exam_recall_questions_do_not_write_or_change_status(tmp_path: Path) -> None:
    engine, llm, memory = make_engine(tmp_path)
    fact = seed_exam(memory)
    before = memory.find_structured_facts(fact_type="academic_event")[0]

    response = engine.respond("Verifica que exame vou ter para a semana.")
    after = memory.find_structured_facts(fact_type="academic_event")[0]

    assert "Estratégias Algorítmicas" in response
    assert engine._last_selected_path == "MEMORY_RECALL"
    assert llm.chat_call_count == 0
    assert after.id == fact.id
    assert after.status == before.status == "upcoming"
    assert after.status_history == before.status_history
    assert after.updated_at == before.updated_at


def test_memory_recall_continuation_stays_grounded(tmp_path: Path) -> None:
    engine, llm, memory = make_engine(tmp_path)
    seed_exam(memory)

    first = engine.respond("Sobre o meu exame.")
    second = engine.respond("Não sabes?")
    third = engine.respond("Supostamente guardaste isso.")

    assert "Estratégias Algorítmicas" in first
    assert "Estratégias Algorítmicas" in second
    assert "Estratégias Algorítmicas" in third
    assert llm.chat_call_count == 0


def test_status_extraction_requires_explicit_evidence() -> None:
    assert extract_academic_event_candidate("Vou ter o exame para a semana.")["status"] == "upcoming"
    assert extract_academic_event_candidate("Já fiz o exame.")["status"] == "completed"
    assert extract_academic_event_candidate("Passei no exame.")["status"] == "passed"
    assert extract_academic_event_candidate("Chumbei no exame.")["status"] == "failed"
    assert extract_academic_event_candidate("Vais conseguir responder à pergunta.") == {}
