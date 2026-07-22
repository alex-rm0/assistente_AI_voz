from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from assistant.cognition.context_builder import ContextBuilder
from assistant.cognition.intent_engine import IntentResult


@dataclass
class MemoryRecord:
    content: str


class FakePersonalModel:
    def facts_about(self, query: str = ""):
        return ["O Alexandre costuma começar por estruturar a arquitetura."]


class FakeSessionManager:
    def facts_for_last_session(self):
        return ["Da última vez trabalhámos no Response Composer."]

    def planner_context(self):
        return "Última sessão: Response Composer."


class FakeLongTermMemory:
    def search(self, query: str, limit: int = 5):
        return [MemoryRecord("Memória relacionada com estudo.")]

    def pending_tasks(self, limit: int = 5, show_details: bool = False):
        return "Tens uma tarefa pendente."


def test_context_builder_collects_relevant_identity_context() -> None:
    intent = IntentResult("explorar_identidade", 0.9, "interpretar identidade", "pergunta sobre o utilizador")

    context = ContextBuilder(
        personal_model=FakePersonalModel(),
        long_term_memory=FakeLongTermMemory(),
        session_manager=FakeSessionManager(),
        now=datetime(2026, 7, 10, 9, 30),
    ).build("O que sabes sobre mim?", intent, enabled_sources={"personal_model"})

    assert context.personal_facts == ["O Alexandre costuma começar por estruturar a arquitetura."]
    assert context.memory_facts == []
    assert "2026-07-10" in context.time_context


def test_context_builder_collects_session_context_for_continuity() -> None:
    intent = IntentResult("retomar_contexto", 0.9, "retomar", "pergunta onde ficou")

    context = ContextBuilder(session_manager=FakeSessionManager()).build(
        "Onde ficámos?",
        intent,
        enabled_sources={"session"},
    )

    assert context.session_facts == ["Da última vez trabalhámos no Response Composer."]


def test_context_builder_does_not_collect_sources_without_permission() -> None:
    intent = IntentResult("conversa_normal", 0.55, "conversar", "pedido geral")

    context = ContextBuilder(
        personal_model=FakePersonalModel(),
        long_term_memory=FakeLongTermMemory(),
        session_manager=FakeSessionManager(),
    ).build("Olá, como estás?", intent, enabled_sources=set())

    assert context.personal_facts == []
    assert context.session_facts == []
    assert context.memory_facts == []
    assert context.pending_tasks == ""
