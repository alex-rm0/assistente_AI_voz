"""KNOWN_ISSUES §5: every app start/close wrote a "Resumo de sessao" memory
into long_term_memory even when the session had no content (~203 of 218
memories), drowning search. A session with no decisions, no tasks changed
and no tools must not create a long-term memory (observer files do not
count); the summary is still persisted in session_manager.sqlite.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.session_manager import SessionManager
from assistant.tool_registry import ToolRegistry


class _FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


class _Observer:
    def __init__(self, files: tuple[str, ...] = ()) -> None:
        self.files = files

    def latest_snapshot(self):
        return SimpleNamespace(current_project="", recently_modified_files=self.files, vscode_sessions=())

    def latest_summary(self):
        return None


def _engine(tmp_path: Path, observer: _Observer) -> tuple[AssistantEngine, SessionManager, Path]:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    session_manager = SessionManager(data)
    engine = AssistantEngine(
        llm=_FakeLLM(),
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=_FakeLLM()),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        context_observer=observer,
        session_manager=session_manager,
    )
    return engine, session_manager, data


def _session_memories(data: Path) -> list[str]:
    with sqlite3.connect(data / "long_term_memory.sqlite") as connection:
        rows = connection.execute("SELECT content FROM memories").fetchall()
    return [row[0] for row in rows if row[0].startswith("Resumo de sessao")]


def test_empty_session_is_not_promoted_to_long_term_memory(tmp_path: Path) -> None:
    engine, session_manager, data = _engine(tmp_path, _Observer())
    session_manager.start_session()

    engine.close_session("fecho da aplicacao")

    assert _session_memories(data) == []
    assert session_manager.latest_summary() is not None


def test_trivial_conversation_session_is_not_promoted(tmp_path: Path) -> None:
    engine, session_manager, data = _engine(tmp_path, _Observer())
    session_manager.record_message_pair("Olá", "Olá! Como estás?")

    engine.close_session("fecho da aplicacao")

    assert _session_memories(data) == []
    assert session_manager.latest_summary() is not None


def test_observer_recent_files_alone_do_not_promote(tmp_path: Path) -> None:
    # files_touched is the Context Observer's recently modified files in the
    # project root (.pytest_cache, requirements.txt...), not what the
    # session did, so it does not count as content.
    observer = _Observer((".pytest_cache/v/cache/nodeids", "requirements.txt"))
    engine, session_manager, data = _engine(tmp_path, observer)
    session_manager.record_message_pair("Olá", "Olá! Como estás?")

    engine.close_session("fecho da aplicacao")

    assert _session_memories(data) == []
    assert session_manager.latest_summary().files_touched


@pytest.mark.parametrize(
    ("user", "assistant", "tool"),
    [
        ("Cria uma tarefa para rever o Planner", "Criei a tarefa.", ""),
        ("Olá", "Olá!", "get_active_window"),
        ("Implementar Planner", "Decidimos criar o Session Manager.", ""),
    ],
    ids=["tasks", "tools", "decisions"],
)
def test_session_with_content_is_still_promoted(tmp_path: Path, user: str, assistant: str, tool: str) -> None:
    engine, session_manager, data = _engine(tmp_path, _Observer(("assistant/planner.py",)))
    session_manager.record_message_pair(user, assistant)
    if tool:
        session_manager.record_tool_used(tool)

    engine.close_session("fecho da aplicacao")

    assert len(_session_memories(data)) == 1
