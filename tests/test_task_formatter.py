from __future__ import annotations

from assistant.long_term_memory import TaskRecord
from assistant.task_formatter import format_task_for_assistant, format_tasks_panel


def test_format_task_for_assistant_hides_raw_fields_by_default() -> None:
    task = TaskRecord(
        id=1,
        title="continuar o desenvolvimento e rever a qualidade da linguagem das respostas",
        due_date="2026-06-08",
        project="AssistenteIA",
        priority="alta",
        status="pending",
    )

    result = format_task_for_assistant(task)

    assert result == (
        "uma tarefa relacionada com o projeto AssistenteIA: "
        "continuar o desenvolvimento e rever a qualidade da linguagem das respostas"
    )
    assert "data:" not in result
    assert "projeto:" not in result
    assert "prioridade:" not in result
    assert "estado:" not in result


def test_format_task_for_assistant_can_show_details() -> None:
    task = TaskRecord(
        id=1,
        title="rever linguagem",
        due_date="2026-06-08",
        project="AssistenteIA",
        priority="alta",
        status="pending",
    )

    result = format_task_for_assistant(task, show_details=True)

    assert "data: 2026-06-08" in result
    assert "projeto: AssistenteIA" in result
    assert "prioridade: alta" in result
    assert "estado: pendente" in result


def test_format_tasks_panel_is_compact() -> None:
    tasks = (
        TaskRecord(id=1, title="rever projeto AssistenteIA", project="AssistenteIA"),
    )

    result = format_tasks_panel(tasks)

    assert result.startswith("- uma tarefa relacionada com o projeto AssistenteIA")
    assert "data:" not in result
