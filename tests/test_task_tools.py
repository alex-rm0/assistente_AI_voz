from __future__ import annotations

from pathlib import Path

from assistant.long_term_memory import LongTermMemory
from assistant.tools import cancel_task, complete_task


class FakeEmbedder:
    def embed(self, text: str):
        return None


def make_memory(tmp_path: Path) -> LongTermMemory:
    return LongTermMemory(tmp_path / "data", embedder=FakeEmbedder())


def test_complete_single_pending_task_updates_database(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    memory.create_task("terminar o relatorio")

    result = complete_task(long_term_memory=memory)

    assert "Marquei essa tarefa" in result
    assert memory.pending_task_count() == 0
    assert "terminar o relatorio" not in memory.pending_tasks()


def test_cancel_single_pending_task_updates_database(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    memory.create_task("responder ao Joao")

    result = cancel_task(long_term_memory=memory)

    assert "Cancelei essa tarefa" in result
    assert memory.pending_task_count() == 0
    assert "responder ao Joao" not in memory.pending_tasks()


def test_complete_ambiguous_task_asks_when_multiple_pending(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    memory.create_task("terminar o relatorio")
    memory.create_task("responder ao Joao")

    result = complete_task(long_term_memory=memory, query="essa tarefa")

    assert "tarefas pendentes" in result
    assert memory.pending_task_count() == 2


def test_cancel_ambiguous_task_asks_when_multiple_pending(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    memory.create_task("terminar o relatorio")
    memory.create_task("responder ao Joao")

    result = cancel_task(long_term_memory=memory, query="essa tarefa")

    assert "tarefas pendentes" in result
    assert memory.pending_task_count() == 2
