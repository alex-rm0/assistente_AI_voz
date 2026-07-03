from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from assistant.long_term_memory import LongTermMemory, infer_task_due_date, infer_task_priority


def test_infer_task_due_date() -> None:
    today = date(2026, 6, 7)

    assert infer_task_due_date("terminar o relatorio hoje", today) == today
    assert infer_task_due_date("terminar o relatorio amanha", today) == today + timedelta(days=1)
    assert infer_task_due_date("rever isto proxima semana", today) == today + timedelta(days=7)
    assert infer_task_due_date("entregar o relatorio sexta-feira", today) == date(2026, 6, 12)


def test_infer_task_priority() -> None:
    assert infer_task_priority("responder urgente") == "alta"
    assert infer_task_priority("quando der rever isto") == "baixa"
    assert infer_task_priority("rever projeto") == "normal"


def test_create_task_with_project_and_due_date(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path)

    result = memory.create_task("terminar o relatorio do projeto AssistenteIA amanha")

    assert "Guardei a tarefa" in result
    assert "AssistenteIA" in result
    assert (date.today() + timedelta(days=1)).isoformat() in result


def test_tasks_for_today(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path)
    memory.create_task("terminar o relatorio", due_date=date.today())
    memory.create_task("comprar material", due_date=date.today() + timedelta(days=1))

    result = memory.tasks_for_today()

    assert "terminar o relatorio" in result
    assert "comprar material" not in result


def test_pending_tasks(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path)
    memory.create_task("terminar o relatorio")

    result = memory.pending_tasks()

    assert "terminar o relatorio" in result


def test_tasks_for_week(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path)
    today = date.today()
    memory.create_task("tarefa desta semana", due_date=today + timedelta(days=2))
    memory.create_task("tarefa futura", due_date=today + timedelta(days=10))

    result = memory.tasks_for_week(today)

    assert "tarefa desta semana" in result
    assert "tarefa futura" not in result


def test_complete_postpone_and_cancel_tasks(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path)
    memory.create_task("responder ao Joao", priority="alta")

    postpone_result = memory.postpone_task("responder ao Joao", due_date=date(2026, 6, 8))
    assert "2026-06-08" in postpone_result
    assert "prioridade:" not in memory.pending_tasks()
    assert "prioridade: alta" in memory.pending_tasks(show_details=True)

    complete_result = memory.complete_task("responder ao Joao")
    assert "concluida" in complete_result
    assert "responder ao Joao" not in memory.pending_tasks()

    memory.create_task("cancelar teste")
    cancel_result = memory.cancel_task("cancelar teste")
    assert "cancelada" in cancel_result
    assert "cancelar teste" not in memory.pending_tasks()
