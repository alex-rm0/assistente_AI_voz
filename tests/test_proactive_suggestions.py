from __future__ import annotations

from datetime import date
from pathlib import Path

from assistant.context_observer import ContextSnapshot, WindowInfo
from assistant.long_term_memory import LongTermMemory
from assistant.proactive_suggestions import (
    generate_proactive_suggestions,
    next_proactive_suggestion,
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
            open_windows=(
                WindowInfo("AssistenteIA - Visual Studio Code", "Code.exe", 1, True),
                WindowInfo("Codex - AssistenteIA", "Codex.exe", 2, False),
            ),
            observed_at=100.0,
        )


def test_suggests_today_task_related_to_project(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.create_task("rever o projeto AssistenteIA", due_date=date(2026, 6, 8), project="AssistenteIA")

    suggestions = generate_proactive_suggestions(memory, today=date(2026, 6, 8))

    assert any("tarefa para hoje" in suggestion.message for suggestion in suggestions)
    assert any("AssistenteIA" in suggestion.message for suggestion in suggestions)


def test_suggests_vscode_and_codex_same_project(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())

    suggestions = generate_proactive_suggestions(memory, FakeObserver(), today=date(2026, 6, 8))

    assert suggestions[0].message == "Tens o VS Code e o Codex abertos no mesmo projeto: AssistenteIA."


def test_suggestion_is_not_repeated_on_same_day(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.create_task("rever o projeto AssistenteIA", due_date=date(2026, 6, 8), project="AssistenteIA")

    first = next_proactive_suggestion(memory, today=date(2026, 6, 8))
    second = next_proactive_suggestion(memory, today=date(2026, 6, 8))

    assert first
    assert second == ""


def test_suggestion_can_reappear_on_next_day(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.create_task("rever o projeto AssistenteIA", due_date=date(2026, 6, 8), project="AssistenteIA")

    first = next_proactive_suggestion(memory, today=date(2026, 6, 8))
    next_day = next_proactive_suggestion(memory, today=date(2026, 6, 9))

    assert first
    assert next_day
