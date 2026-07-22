from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.planner import plan_user_request
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.session_manager import SessionManager
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta"

    def embed(self, text: str):
        return None


class FakeContextObserver:
    def latest_snapshot(self):
        return SimpleNamespace(
            current_project="AssistenteIA",
            recently_modified_files=("assistant/planner.py", "README.md"),
            vscode_sessions=(),
        )

    def latest_summary(self):
        return SimpleNamespace(
            summary="Entre as 09:00 e as 10:00 o Alexandre trabalhou no AssistenteIA, sobretudo no Planner."
        )


def test_session_manager_creates_session(tmp_path) -> None:
    manager = SessionManager(tmp_path)

    manager.start_session(datetime(2026, 7, 8, 9, 0, 0))

    assert manager.current_started_at == datetime(2026, 7, 8, 9, 0, 0)


def test_session_manager_closes_and_persists_summary(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    manager.start_session(datetime(2026, 7, 8, 9, 0, 0))
    manager.record_message_pair("Vamos continuar o projeto AssistenteIA", "Vamos trabalhar no Planner.")

    summary = manager.end_session(
        FakeContextObserver(),
        now=datetime(2026, 7, 8, 10, 0, 0),
        reason="teste",
    )

    assert summary is not None
    assert summary.main_project == "AssistenteIA"
    assert "Planner" in summary.main_activity
    assert "assistant/planner.py" in summary.files_touched


def test_session_manager_recovers_latest_session(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    manager.start_session(datetime(2026, 7, 8, 9, 0, 0))
    manager.record_message_pair("Implementar Planner", "O proximo passo e criar Session Manager.")
    manager.end_session(FakeContextObserver(), now=datetime(2026, 7, 8, 10, 0, 0))

    latest = manager.latest_summary()

    assert latest is not None
    assert latest.main_project == "AssistenteIA"
    assert "Session Manager" in latest.next_suggested_step


def test_session_manager_generates_human_summary(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    manager.start_session(datetime(2026, 7, 8, 9, 0, 0))
    manager.record_message_pair("Implementar Planner", "Decidimos criar o Session Manager como proximo passo.")
    manager.end_session(FakeContextObserver(), now=datetime(2026, 7, 8, 10, 0, 0))

    answer = manager.answer_last_session()

    assert "Da última vez" in answer
    assert "próximo passo" in answer
    assert "sessao iniciada" not in answer.lower()
    assert "motivo de fecho" not in answer.lower()


def test_session_manager_today_summary(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    manager.start_session(datetime.combine(date.today(), datetime.min.time()))
    manager.record_message_pair("Implementar Planner", "O Planner ficou pronto.")
    manager.end_session(FakeContextObserver(), now=datetime.now())

    answer = manager.answer_today()

    assert "Hoje" in answer
    assert "AssistenteIA" in answer


def test_session_manager_closes_after_idle(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    manager.start_session(datetime(2026, 7, 8, 9, 0, 0))
    manager.last_interaction_at = datetime(2026, 7, 8, 9, 0, 0)

    summary = manager.end_if_inactive(
        60,
        FakeContextObserver(),
        now=datetime(2026, 7, 8, 9, 2, 0),
    )

    assert summary is not None
    assert manager.current_started_at is None


def test_planner_uses_last_session_summary_for_project() -> None:
    result = plan_user_request(
        "Vamos continuar",
        last_session_summary="Ultima sessao: projeto=AssistenteIA; atividade=Planner; proximo_passo=Session Manager",
    )

    assert result.intent == "retomar projeto"
    assert result.related_project == "AssistenteIA"


def test_session_close_is_promoted_to_long_term_memory_timeline(tmp_path: Path) -> None:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    long_term_memory = LongTermMemory(data, embedder=FakeLLM())
    session_manager = SessionManager(data)
    engine = AssistantEngine(
        llm=FakeLLM(),
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=long_term_memory,
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        context_observer=FakeContextObserver(),
        session_manager=session_manager,
    )

    session_manager.record_message_pair("Implementar Session Manager", "Decidimos testar a continuidade.")
    engine.close_session("teste")

    timeline = long_term_memory.timeline_for_date(date.today())

    assert "Resumo de sessao" in timeline
    assert "AssistenteIA" in timeline
