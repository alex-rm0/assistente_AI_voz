from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine, _extract_research_query
from assistant.presence_manager import PresenceManager
from assistant.tool_registry import ToolRegistry
from assistant.ui_event_adapter import UIEventAdapter


class DummyMemory:
    def __init__(self):
        self.items = []

    def load(self):
        return list(self.items)

    def add(self, role, content):
        self.items.append({"role": role, "content": content})

    def append_pair(self, user_message, assistant_message):
        self.add("user", user_message)
        self.add("assistant", assistant_message)

    def clear(self):
        self.items.clear()


class DummyLongTermMemory:
    def get_preference(self, key, default=None):
        return default

    def set_preference(self, key, value):
        pass


class MemoryExtractionMustNotRun(DummyLongTermMemory):
    def remember_structured_fact_with_trace(self, *args, **kwargs):
        raise AssertionError("Research commands must not be stored as personal tasks.")


class DummyLLM:
    system_prompt = ""
    chat_call_count = 0

    def chat(self, *args, **kwargs):
        raise AssertionError("Research routing should not call the LLM")


def make_engine(tools: ToolRegistry | None = None) -> AssistantEngine:
    return AssistantEngine(
        llm=DummyLLM(),
        memory=DummyMemory(),
        long_term_memory=DummyLongTermMemory(),
        tools=tools or ToolRegistry(),
        workspace_path=Path("workspace"),
        base_system_prompt="",
        presence_manager=PresenceManager(),
    )


def make_engine_without_memory_extraction(tools: ToolRegistry | None = None) -> AssistantEngine:
    return AssistantEngine(
        llm=DummyLLM(),
        memory=DummyMemory(),
        long_term_memory=MemoryExtractionMustNotRun(),
        tools=tools or ToolRegistry(),
        workspace_path=Path("workspace"),
        base_system_prompt="",
        presence_manager=PresenceManager(),
    )


def test_explicit_research_queries_are_detected() -> None:
    assert _extract_research_query("Pesquisa sobre Picasso.") == "Picasso"
    assert _extract_research_query("Quero que faças uma pesquisa sobre Picasso.") == "Picasso"
    assert _extract_research_query("Pesquisa informação geral sobre Picasso.") == "Picasso"
    assert _extract_research_query("Pesquisa na internet sobre Picasso.") == "Picasso"
    assert _extract_research_query("Quero que investigues Picasso.") == "Picasso"
    assert _extract_research_query("Encontra fontes sobre Picasso.") == "Picasso"
    assert _extract_research_query("Verifica online sobre Picasso.") == "Picasso"
    assert _extract_research_query("Estou bem. Pesquisa informação geral sobre Picasso.") == "Picasso"
    assert _extract_research_query("Tudo certo. Consegues pesquisar na internet sobre Picasso?") == "Picasso"


def test_preference_about_research_is_not_command() -> None:
    assert _extract_research_query("Gosto de pesquisar sobre Picasso.") == ""


def test_research_without_tool_is_honest_and_emits_event() -> None:
    engine = make_engine()

    response = engine.respond("Pesquisa sobre Picasso.")
    events = engine.consume_ui_events()

    assert response == "Ainda não tenho uma ferramenta de pesquisa ligada."
    assert "research_unavailable" in "\n".join(events)
    assert "research_results_ready" not in "\n".join(events)
    assert "Picasso" in "\n".join(events)


def test_research_command_does_not_create_personal_task() -> None:
    engine = make_engine_without_memory_extraction()

    response = engine.respond("Quero que faças uma pesquisa sobre Picasso.")

    assert response == "Ainda não tenho uma ferramenta de pesquisa ligada."


def test_research_with_tool_emits_structured_events() -> None:
    registry = ToolRegistry()

    @registry.register(
        name="web_search",
        description="Pesquisa real de teste.",
        permissions=("web_search",),
    )
    def web_search(query: str) -> str:
        assert query == "Picasso"
        return "Fonte de teste sobre Picasso."

    engine = make_engine(tools=registry)

    response = engine.respond("Pesquisa sobre Picasso.")
    events = "\n".join(engine.consume_ui_events())

    assert response == "Fonte de teste sobre Picasso."
    assert "research_started" in events
    assert "research_results_ready" in events
    assert "research_completed" in events
    assert "Fonte de teste sobre Picasso." in events


def test_clear_conversation_resets_history_and_transient_context() -> None:
    engine = make_engine()
    engine.respond("Pesquisa sobre Picasso.")
    engine._active_memory_topic = "academic_event"
    engine._active_memory_entity_id = "1"
    engine._active_memory_recall_ttl = 2

    engine.clear_conversation()
    events = "\n".join(engine.consume_ui_events())

    assert engine.history() == []
    assert engine._active_operation_topic == ""
    assert engine._active_memory_topic == ""
    assert "conversation_cleared" in events


def test_research_followup_keeps_active_topic() -> None:
    registry = ToolRegistry()

    @registry.register(
        name="web_search",
        description="Pesquisa real de teste.",
        permissions=("web_search",),
    )
    def web_search(query: str) -> str:
        assert query == "Picasso"
        return "Visão geral de teste sobre Picasso."

    engine = make_engine(tools=registry)
    engine._active_operation_type = "research"
    engine._active_operation_topic = "Picasso"
    engine._active_operation_id = "research-test"
    engine._active_operation_ttl = 2

    response = engine.respond("Só informação geral.")
    events = "\n".join(engine.consume_ui_events())

    assert response == "Visão geral de teste sobre Picasso."
    assert "research-test" in events
    assert "research_results_ready" in events


def test_ui_event_adapter_preserves_utf8_text() -> None:
    payload = UIEventAdapter.serialize(
        "research_results_ready",
        {
            "topic": "Estratégias Algorítmicas",
            "summary": "Informação sobre João, memória, aplicações e ecrã.",
            "results": [{"source": "Fundação Calouste Gulbenkian"}],
        },
    )

    assert "Estratégias Algorítmicas" in payload
    assert "João" in payload
    assert "Fundação" in payload
    assert "\\u00" not in payload
