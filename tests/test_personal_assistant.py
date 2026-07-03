from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from assistant.context_observer import ContextSnapshot
from assistant.long_term_memory import LongTermMemory
from assistant.personal_assistant import (
    generate_context_summary,
    generate_daily_briefing,
    generate_greeting,
    generate_session_resume,
    generate_task_summary,
)


class FakeEmbedder:
    def embed(self, text: str):
        return None


class FakeObserver:
    def latest_snapshot(self):
        return ContextSnapshot(
            active_app="Code.exe",
            active_window="AssistenteIA - Visual Studio Code",
            current_project="AssistenteIA",
            observed_at=100.0,
        )

    def latest_summary(self):
        return None


def test_greeting_rules_without_tasks(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())

    assert generate_greeting(memory, now=datetime(2026, 6, 8, 8, 0)).startswith("Bom dia, Alexandre.")
    assert generate_greeting(memory, now=datetime(2026, 6, 8, 15, 0)).startswith("Boa tarde, Alexandre.")
    assert generate_greeting(memory, now=datetime(2026, 6, 8, 22, 0)).startswith("Olá Alexandre.")


def test_greeting_shows_summary_when_today_has_tasks(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.create_task("responder ao João", due_date=date(2026, 6, 8))

    result = generate_greeting(memory, now=datetime(2026, 6, 8, 8, 0))

    assert "Bom dia, Alexandre." in result
    assert "Para hoje, tens" in result
    assert "responder ao João" in result
    assert "Queres ver?" not in result


def test_task_summary_mentions_today_and_overdue_tasks(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.create_task("tarefa atrasada", due_date=date(2026, 6, 7))
    memory.create_task("tarefa de hoje", due_date=date(2026, 6, 8))

    result = generate_task_summary(memory, today=date(2026, 6, 8))

    assert "Tens em atraso" in result
    assert "Para hoje, tens" in result
    assert "tarefa atrasada" in result
    assert "tarefa de hoje" in result


def test_session_resume_naturalizes_presence_events(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.remember_timeline_event(
        "Mudanca automatica de modo de presenca: ACTIVE_CONVERSATION -> OFFLINE.",
        event_date=date.today() - timedelta(days=1),
    )

    result = generate_session_resume(memory)

    assert "testar os modos de presença" in result
    assert "ACTIVE_CONVERSATION -> OFFLINE" not in result


def test_context_summary_uses_observed_project() -> None:
    result = generate_context_summary(FakeObserver())

    assert "projeto AssistenteIA" in result
    assert "aplicação ativa" in result


def test_daily_briefing_sounds_personal(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.create_task("rever o projeto AssistenteIA", due_date=date(2026, 6, 8))

    result = generate_daily_briefing(memory, FakeObserver(), today=date(2026, 6, 8))

    assert "Aqui está o ponto de situação para hoje" in result
    assert "Para hoje, tens" in result
    assert "projeto que parece mais presente" in result
    assert "data:" not in result
    assert "projeto:" not in result
    assert "prioridade:" not in result
    assert "estado:" not in result
