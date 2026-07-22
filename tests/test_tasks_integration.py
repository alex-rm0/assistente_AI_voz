from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeMemory:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def append_pair(self, user_message: str, response: str) -> None:
        self.items.append({"role": "user", "content": user_message})
        self.items.append({"role": "assistant", "content": response})

    def load(self) -> list[dict[str, str]]:
        return list(self.items)

    def clear(self) -> None:
        self.items.clear()


class FakeLLM:
    def __init__(self) -> None:
        self.system_prompt = ""

    def choose_tool(self, user_message, tools_description, profile_name=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta"

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path) -> AssistantEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    return AssistantEngine(
        llm=llm,
        memory=FakeMemory(),
        long_term_memory=LongTermMemory(tmp_path / "data", embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_system_prompt("Geral"),
        presence_manager=PresenceManager(),
    )


def test_conversation_creates_task(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    result = engine.respond("Lembra-me de terminar o relatorio.")

    assert "Guardei a tarefa" in result
    assert "terminar o relatorio" in result


def test_conversation_creates_tomorrow_reminder_from_this(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.respond("Preciso de rever o relatorio.")

    result = engine.respond("Lembra-me disto amanha.")

    assert "Guardei a tarefa" in result
    assert "Preciso de rever o relatorio" in result
    assert (date.today() + timedelta(days=1)).isoformat() in result


def test_conversation_lists_today_tasks(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.respond("Lembra-me de terminar o relatorio hoje.")

    result = engine.respond("O que tenho para fazer hoje?")

    # The structured-memory path now normalizes the spelling before speaking
    # it back ("relatorio" -> "relatório"), per the raw/canonical
    # normalization requirement — the DB keeps the original, the spoken
    # answer never does.
    assert "terminar o relatório" in result


def test_conversation_creates_task_from_natural_language(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    result = engine.respond("Tenho de entregar o relatorio sexta-feira.")

    assert "Guardei a tarefa" in result
    assert "entregar o relatorio sexta-feira" in result


def test_conversation_lists_week_tasks(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.respond("Adiciona uma tarefa para rever o projeto AssistenteIA amanhã.")

    result = engine.respond("Que tarefas tenho esta semana?")

    assert "rever o projeto AssistenteIA" in result


def test_conversation_updates_task_status(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.respond("Lembra-me de responder ao Joao amanhã.")

    postponed = engine.respond("Adia esta tarefa para amanhã.")
    assert "Adiei a tarefa" in postponed

    completed = engine.respond("Marca esta tarefa como concluída.")
    assert "Marquei essa tarefa" in completed
    assert engine.pending_task_count() == 0
    assert "responder ao Joao" not in engine.pending_tasks_summary()

    engine.respond("Lembra-me de rever o projeto AssistenteIA.")
    cancelled = engine.respond("Cancela esta tarefa.")
    assert "Cancelei essa tarefa" in cancelled
    assert engine.pending_task_count() == 0
    assert "rever o projeto AssistenteIA" not in engine.pending_tasks_summary()


def test_conversation_completes_single_task_from_natural_phrase(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.respond("Lembra-me de rever o painel de tarefas.")

    result = engine.respond("Ja terminei.")

    assert "Marquei essa tarefa" in result
    assert engine.pending_task_count() == 0
    assert "rever o painel de tarefas" not in engine.pending_tasks_summary()


def test_conversation_cancels_single_task_from_natural_phrase(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.respond("Lembra-me de ligar ao Joao.")

    result = engine.respond("Retira esse lembrete.")

    assert "Cancelei essa tarefa" in result
    assert engine.pending_task_count() == 0
    assert "ligar ao Joao" not in engine.pending_tasks_summary()


def test_conversation_asks_which_task_when_multiple_pending(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.respond("Lembra-me de rever o README.")
    engine.respond("Lembra-me de testar o painel.")

    result = engine.respond("Limpa essa tarefa.")

    assert "tarefas pendentes" in result
    assert engine.pending_task_count() == 2
