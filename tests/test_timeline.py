from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from assistant.long_term_memory import LongTermMemory, extract_people, extract_project, infer_event_date


def test_infer_event_date_relative_expressions() -> None:
    today = date(2026, 6, 7)

    assert infer_event_date("Ontem trabalhamos no projeto AssistenteIA.", today) == date(2026, 6, 6)
    assert infer_event_date("Na semana passada falamos sobre ferias.", today) == date(2026, 5, 31)
    assert infer_event_date("Ha tres meses comecaste este projeto.", today) == date(2026, 3, 7)


def test_extract_project_and_people() -> None:
    content = "Ontem estivemos a trabalhar no projeto AssistenteIA com Alexandre."

    assert extract_project(content) == "AssistenteIA"
    assert extract_people(content) == ["Alexandre"]


def test_remember_timeline_event_persists_project_and_people(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path)
    event_date = date.today() - timedelta(days=1)

    result = memory.remember_timeline_event(
        "Ontem estivemos a trabalhar no projeto AssistenteIA com Alexandre.",
        event_date=event_date,
    )
    answer = memory.timeline_for_date(event_date)

    assert "Registei na timeline" in result
    assert "AssistenteIA" in answer
    assert "Alexandre" in answer


def test_project_start_returns_earliest_project_event(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path)
    memory.remember_timeline_event("Ontem continuamos o projeto AssistenteIA.", date(2026, 6, 6))
    memory.remember_timeline_event("Ha tres meses comecamos o projeto AssistenteIA.", date(2026, 3, 7))

    answer = memory.project_start("AssistenteIA")

    assert "2026-03-07" in answer
    assert "AssistenteIA" in answer


def test_project_start_works_with_this_project_reference(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path)
    memory.remember_timeline_event("Ha tres meses comecaste este projeto.", date(2026, 3, 7))

    answer = memory.project_start()

    assert "2026-03-07" in answer
    assert "este projeto" in answer


def test_current_work_context_uses_recent_events(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path)
    memory.remember_timeline_event("Ontem estivemos no projeto AssistenteIA.", date.today() - timedelta(days=1))

    answer = memory.current_work_context()

    assert "AssistenteIA" in answer
