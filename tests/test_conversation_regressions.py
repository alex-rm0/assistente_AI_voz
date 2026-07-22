from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class RaisingLLM:
    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.call_count += 1
        raise AssertionError("o LLM nao devia ser necessario para este caso")

    def choose_tool(self, *args, **kwargs):
        raise AssertionError("choose_tool nao devia ser chamado")

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path) -> tuple[AssistantEngine, RaisingLLM]:
    llm = RaisingLLM()
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )
    engine.debug_ollama_payload = True
    return engine, llm


def test_exam_emotion_gets_warm_brief_response_without_memory_write(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    response = engine.respond("Estou um bocado nervoso para um exame.")
    telemetry = engine.get_last_turn_telemetry()

    assert "nervoso" in response.lower() or "normal" in response.lower() or "percebo" in response.lower()
    assert response.count("?") <= 1
    assert len(response.split()) <= 25
    assert telemetry["memory_write_action"] is None
    assert engine.long_term_memory.find_structured_facts(fact_type="academic_event") == []
    assert llm.call_count == 0


def test_short_phrase_request_gets_direct_usable_sentence(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    response = engine.respond("Ajuda-me a escrever uma frase curta para pedir uma revisao.")

    assert "por favor" in response.lower()
    assert response.count("?") == 0
    assert len(response.split()) <= 25
    assert "queres que" not in response.lower()
    assert llm.call_count == 0


def test_more_short_phrase_requests_do_not_ask_for_context(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    examples = (
        "Escreve uma frase curta para pedir desculpa.",
        "Da-me uma frase curta para pedir confirmacao.",
        "Escreve uma frase formal para pedir resposta.",
    )

    for message in examples:
        response = engine.respond(message)
        assert response.strip()
        assert len(response.split()) <= 25
        assert "que tipo" not in response.lower()
        assert "podes dizer" not in response.lower()

    assert llm.call_count == 0


def test_current_session_continuity_uses_recent_conversation(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    engine.respond("Estamos a corrigir o sistema de memoria do Echo.")
    engine.respond("Ja corrigimos a escrita passiva, falta o recall.")
    llm.call_count = 0
    response = engine.respond("Resume onde ficamos.")

    normalized = response.lower()
    assert "escrita passiva" in normalized
    assert "recall" in normalized
    assert "nao tenho contexto" not in normalized
    assert llm.call_count == 0


def test_session_continuity_without_context_is_honest(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    response = engine.respond("Onde ficamos?")

    assert "nao tenho" in response.lower() or "não tenho" in response.lower()
    assert llm.call_count == 0


def test_pyside6_module_error_gets_direct_grounded_help(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    response = engine.respond("Tenho um erro em Python: ModuleNotFoundError: No module named PySide6.")

    assert "PySide6" in response
    assert "pip install PySide6" in response
    assert "acompanhar" not in response.lower()
    assert llm.call_count == 0


def test_travel_planning_asks_preference_before_suggesting_places(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    response = engine.respond("Queria planear umas ferias no Norte de Portugal.")

    assert "descansar" in response.lower()
    assert response.count("?") == 1
    assert "Porto" not in response
    assert llm.call_count == 0
