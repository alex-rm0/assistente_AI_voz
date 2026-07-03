from __future__ import annotations

from datetime import date
from pathlib import Path

from assistant.briefing import (
    NO_DATA_MESSAGE,
    generate_daily_briefing,
    generate_session_continuity_summary,
    get_last_active_project,
    summarize_yesterday,
)
from assistant.context_observer import ContextSnapshot, ContextSummary
from assistant.long_term_memory import LongTermMemory


class FakeEmbedder:
    def embed(self, text: str):
        return None


class FakeObserver:
    def __init__(
        self,
        snapshot: ContextSnapshot | None = None,
        summary: ContextSummary | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.summary = summary

    def latest_snapshot(self):
        return self.snapshot

    def latest_summary(self):
        return self.summary

    def activity_summary(self, limit: int = 10):
        return []


def test_daily_briefing_with_tasks(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.create_task("terminar o relatório hoje", due_date=date(2026, 6, 7), project="AssistenteIA")

    result = generate_daily_briefing(memory, today=date(2026, 6, 7))

    assert "Resumo para hoje" in result
    assert "Tarefas pendentes:" in result
    assert "terminar o relatório hoje" in result
    assert "Inferências prováveis:" in result


def test_daily_briefing_without_tasks_uses_observed_context(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    observer = FakeObserver(
        snapshot=ContextSnapshot(
            active_app="Code.exe",
            active_window="AssistenteIA - Visual Studio Code",
            current_project="AssistenteIA",
            observed_at=100.0,
        )
    )

    result = generate_daily_briefing(memory, observer, today=date(2026, 6, 7))

    assert "Contexto observado:" in result
    assert "AssistenteIA" in result
    assert "Não encontrei tarefas pendentes relevantes." in result


def test_session_continuity_with_last_active_project(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    observer = FakeObserver(
        snapshot=ContextSnapshot(
            active_app="Code.exe",
            active_window="AssistenteIA - Visual Studio Code",
            current_project="AssistenteIA",
            observed_at=100.0,
        )
    )

    result = generate_session_continuity_summary(memory, observer, today=date(2026, 6, 7))

    assert "Continuidade da sessão" in result
    assert "Último projeto ativo observado: AssistenteIA" in result
    assert "Parece que ficámos ligados ao projeto AssistenteIA." in result


def test_session_continuity_without_data(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())

    result = generate_session_continuity_summary(memory, today=date(2026, 6, 7))

    assert result == NO_DATA_MESSAGE


def test_last_active_project_without_observer_data_is_clear() -> None:
    assert (
        get_last_active_project(None)
        == "Ainda não tenho dados suficientes para saber qual foi o último projeto ativo."
    )


def test_yesterday_summary_uses_timeline(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.remember_timeline_event(
        "Ontem trabalhámos no projeto AssistenteIA.",
        event_date=date(2026, 6, 6),
    )

    result = summarize_yesterday(memory, today=date(2026, 6, 7))

    assert "Resumo de ontem" in result
    assert "AssistenteIA" in result
