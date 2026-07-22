from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from assistant.session_reflection import (
    SessionReflection,
    format_next_step,
    format_session_reflection,
    format_startup_session_hint,
    format_today_reflection,
    reflect_session,
)


DEFAULT_SESSION_DB = "session_manager.sqlite"
NO_SESSION_DATA = "Ainda nao tenho uma sessao anterior guardada."


@dataclass(frozen=True)
class SessionSummary:
    id: int
    started_at: str
    ended_at: str
    main_project: str
    main_activity: str
    summary: str
    files_touched: str
    tasks_changed: str
    tools_used: str
    decisions_taken: str
    next_suggested_step: str


class SessionManager:
    """Tracks useful work-session summaries without storing raw activity logs."""

    def __init__(self, data_path: Path, db_file: str = DEFAULT_SESSION_DB) -> None:
        self.data_path = data_path.resolve()
        self.db_path = (self.data_path / db_file).resolve()
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.current_started_at: datetime | None = None
        self.last_interaction_at: datetime | None = None
        self.messages: list[str] = []
        self.tools_used: set[str] = set()
        self.tasks_changed: set[str] = set()
        self.decisions: list[str] = []

    def start_session(self, now: datetime | None = None) -> None:
        if self.current_started_at is not None:
            return
        self.current_started_at = now or datetime.now()
        self.last_interaction_at = self.current_started_at
        self.messages = []
        self.tools_used = set()
        self.tasks_changed = set()
        self.decisions = []

    def end_session(
        self,
        context_observer: Any | None = None,
        now: datetime | None = None,
        reason: str = "",
    ) -> SessionSummary | None:
        if self.current_started_at is None:
            return None
        ended_at = now or datetime.now()
        summary = self._build_summary(context_observer, ended_at, reason)
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO session_summaries (
                    started_at, ended_at, main_project, main_activity, summary,
                    files_touched, tasks_changed, tools_used, decisions_taken, next_suggested_step
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.started_at,
                    summary.ended_at,
                    summary.main_project,
                    summary.main_activity,
                    summary.summary,
                    summary.files_touched,
                    summary.tasks_changed,
                    summary.tools_used,
                    summary.decisions_taken,
                    summary.next_suggested_step,
                ),
            )
            saved = SessionSummary(
                id=int(cursor.lastrowid),
                started_at=summary.started_at,
                ended_at=summary.ended_at,
                main_project=summary.main_project,
                main_activity=summary.main_activity,
                summary=summary.summary,
                files_touched=summary.files_touched,
                tasks_changed=summary.tasks_changed,
                tools_used=summary.tools_used,
                decisions_taken=summary.decisions_taken,
                next_suggested_step=summary.next_suggested_step,
            )
        self.current_started_at = None
        self.last_interaction_at = None
        self.messages = []
        self.tools_used = set()
        self.tasks_changed = set()
        self.decisions = []
        return saved

    def record_message_pair(self, user_message: str, assistant_response: str) -> None:
        if self.current_started_at is None:
            self.start_session()
        self.last_interaction_at = datetime.now()
        combined = f"Utilizador: {user_message.strip()} | Assistente: {assistant_response.strip()}"
        if len(combined) > 500:
            combined = combined[:497] + "..."
        if _is_relevant_message(combined):
            self.messages.append(combined)
            self.messages = self.messages[-8:]
        self._track_task_change(combined)
        self._track_tool_use(combined)
        self._track_decision(combined)

    def record_tool_used(self, tool_name: str) -> None:
        if tool_name:
            self.tools_used.add(tool_name)

    def end_if_inactive(
        self,
        max_idle_seconds: float,
        context_observer: Any | None = None,
        now: datetime | None = None,
    ) -> SessionSummary | None:
        if self.current_started_at is None or self.last_interaction_at is None:
            return None
        current = now or datetime.now()
        idle_seconds = (current - self.last_interaction_at).total_seconds()
        if idle_seconds < max_idle_seconds:
            return None
        return self.end_session(context_observer, now=current, reason="inatividade")

    def latest_summary(self) -> SessionSummary | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, started_at, ended_at, main_project, main_activity, summary,
                       files_touched, tasks_changed, tools_used, decisions_taken, next_suggested_step
                FROM session_summaries
                ORDER BY ended_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        return _summary_from_row(row) if row else None

    def summaries_for_date(self, day: date) -> list[SessionSummary]:
        prefix = day.isoformat()
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, started_at, ended_at, main_project, main_activity, summary,
                       files_touched, tasks_changed, tools_used, decisions_taken, next_suggested_step
                FROM session_summaries
                WHERE started_at LIKE ? OR ended_at LIKE ?
                ORDER BY started_at ASC, id ASC
                """,
                (f"{prefix}%", f"{prefix}%"),
            ).fetchall()
        return [_summary_from_row(row) for row in rows]

    def answer_last_session(self) -> str:
        summary = self.latest_summary()
        if summary is None:
            return NO_SESSION_DATA
        return format_session_reflection(summary)

    def answer_today(self, today: date | None = None) -> str:
        summaries = self.summaries_for_date(today or date.today())
        if not summaries:
            return "Ainda não tenho sessões úteis guardadas para hoje."
        return format_today_reflection(summaries)

    def answer_changes_since_last_time(self) -> str:
        summary = self.latest_summary()
        if summary is None:
            return NO_SESSION_DATA
        facts = []
        if summary.files_touched:
            facts.append(f"ficheiros tocados: {summary.files_touched}")
        if summary.tasks_changed:
            facts.append(f"tarefas alteradas: {summary.tasks_changed}")
        if summary.tools_used:
            facts.append(f"ferramentas usadas: {summary.tools_used}")
        if summary.decisions_taken:
            facts.append(f"decisoes tomadas: {summary.decisions_taken}")
        if not facts:
            return "Na ultima sessao nao tenho alteracoes concretas suficientes para destacar."
        return "Desde a ultima sessao, tenho estes factos guardados:\n- " + "\n- ".join(facts)

    def answer_next_step(self) -> str:
        summary = self.latest_summary()
        if summary is None:
            return NO_SESSION_DATA
        return format_next_step(summary)

    def facts_for_last_session(self) -> list[str]:
        summary = self.latest_summary()
        if summary is None:
            return []
        return _reflection_facts(reflect_session(summary))

    def facts_for_today(self, today: date | None = None) -> list[str]:
        facts: list[str] = []
        for summary in self.summaries_for_date(today or date.today()):
            facts.extend(_reflection_facts(reflect_session(summary)))
        return facts

    def facts_for_next_step(self) -> list[str]:
        summary = self.latest_summary()
        if summary is None:
            return []
        next_step = reflect_session(summary).next_step
        return [f"O próximo passo parecia ser {next_step}."] if next_step else []

    def facts_for_changes_since_last_time(self) -> list[str]:
        summary = self.latest_summary()
        if summary is None:
            return []
        reflection = reflect_session(summary)
        facts = []
        if reflection.what_happened:
            facts.append(f"Desde a última vez, {reflection.what_happened}.")
        if reflection.decisions:
            facts.append(f"Ficou registado que {reflection.decisions}.")
        if reflection.pending:
            facts.append(f"Ficou pendente: {reflection.pending}.")
        return facts

    def planner_context(self) -> str:
        summary = self.latest_summary()
        if summary is None:
            return ""
        return (
            f"Ultima sessao: projeto={summary.main_project or 'desconhecido'}; "
            f"atividade={summary.main_activity or 'sem atividade principal'}; "
            f"resumo={summary.summary}; decisoes={summary.decisions_taken}; "
            f"proximo_passo={summary.next_suggested_step}"
        )

    def startup_hint(self, user_name: str = "Alexandre") -> str:
        summary = self.latest_summary()
        if summary is None:
            return ""
        return format_startup_session_hint(summary, user_name=user_name)

    def _build_summary(self, context_observer: Any | None, ended_at: datetime, reason: str) -> SessionSummary:
        snapshot = _latest_snapshot(context_observer)
        latest_context_summary = _latest_context_summary(context_observer)
        main_project = _project_from_snapshot(snapshot) or _project_from_text(self.messages) or _project_from_text([latest_context_summary])
        files_touched = _files_from_snapshot(snapshot)
        tools_used = ", ".join(sorted(self.tools_used))
        tasks_changed = "; ".join(sorted(self.tasks_changed))
        decisions_taken = "; ".join(self.decisions[-5:])
        main_activity = _main_activity(main_project, self.messages, latest_context_summary)
        next_step = _next_step(main_project, self.messages)
        facts = [
            f"sessao iniciada em {self.current_started_at.isoformat(timespec='seconds') if self.current_started_at else ''}",
            f"sessao terminada em {ended_at.isoformat(timespec='seconds')}",
        ]
        if reason:
            facts.append(f"motivo de fecho: {reason}")
        if latest_context_summary:
            facts.append(f"contexto observado: {latest_context_summary}")
        if self.messages:
            facts.append("mensagens relevantes: " + " | ".join(self.messages[-4:]))
        summary_text = ". ".join(item for item in facts if item).strip()
        return SessionSummary(
            id=0,
            started_at=self.current_started_at.isoformat(timespec="seconds") if self.current_started_at else "",
            ended_at=ended_at.isoformat(timespec="seconds"),
            main_project=main_project,
            main_activity=main_activity,
            summary=summary_text,
            files_touched=files_touched,
            tasks_changed=tasks_changed,
            tools_used=tools_used,
            decisions_taken=decisions_taken,
            next_suggested_step=next_step,
        )

    def _track_task_change(self, text: str) -> None:
        normalized = _normalize(text)
        if any(word in normalized for word in ("tarefa", "lembrete", "concluida", "cancelada", "adiada")):
            self.tasks_changed.add(_shorten(text, 160))

    def _track_tool_use(self, text: str) -> None:
        normalized = _normalize(text)
        markers = {
            "abri o url": "open_url",
            "abri o projeto": "open_project",
            "abri '": "open_file_or_folder",
            "criei o ficheiro": "create_workspace_file",
            "ficheiros na pasta workspace": "list_workspace_files",
        }
        for marker, tool_name in markers.items():
            if marker in normalized:
                self.tools_used.add(tool_name)

    def _track_decision(self, text: str) -> None:
        normalized = _normalize(text)
        if any(word in normalized for word in ("decidimos", "proximo passo", "sugeri", "planeei")):
            self.decisions.append(_shorten(_extract_decision_text(text), 180))
            self.decisions = self.decisions[-5:]

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    main_project TEXT NOT NULL DEFAULT '',
                    main_activity TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL,
                    files_touched TEXT NOT NULL DEFAULT '',
                    tasks_changed TEXT NOT NULL DEFAULT '',
                    tools_used TEXT NOT NULL DEFAULT '',
                    decisions_taken TEXT NOT NULL DEFAULT '',
                    next_suggested_step TEXT NOT NULL DEFAULT ''
                )
                """
            )
            _ensure_column(connection, "session_summaries", "decisions_taken", "TEXT NOT NULL DEFAULT ''")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_summaries_ended_at
                ON session_summaries(ended_at)
                """
            )


def _reflection_facts(reflection: SessionReflection) -> list[str]:
    facts = []
    if reflection.what_happened:
        facts.append(f"Da última vez, {reflection.what_happened}.")
    if reflection.decisions:
        facts.append(f"Ficou registado que {reflection.decisions}.")
    if reflection.pending:
        facts.append(f"Ficou pendente: {reflection.pending}.")
    if reflection.next_step:
        facts.append(f"O próximo passo parecia ser {reflection.next_step}.")
    return facts


def _summary_from_row(row: tuple) -> SessionSummary:
    return SessionSummary(
        id=int(row[0]),
        started_at=str(row[1]),
        ended_at=str(row[2]),
        main_project=str(row[3] or ""),
        main_activity=str(row[4] or ""),
        summary=str(row[5] or ""),
        files_touched=str(row[6] or ""),
        tasks_changed=str(row[7] or ""),
        tools_used=str(row[8] or ""),
        decisions_taken=str(row[9] or ""),
        next_suggested_step=str(row[10] or ""),
    )


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _latest_snapshot(context_observer: Any | None) -> Any | None:
    if context_observer is None:
        return None
    latest_snapshot = getattr(context_observer, "latest_snapshot", None)
    if latest_snapshot is None:
        return None
    try:
        return latest_snapshot()
    except Exception:
        return None


def _latest_context_summary(context_observer: Any | None) -> str:
    if context_observer is None:
        return ""
    latest_summary = getattr(context_observer, "latest_summary", None)
    if latest_summary is None:
        return ""
    try:
        value = latest_summary()
    except Exception:
        return ""
    return str(getattr(value, "summary", "") or "")


def _project_from_snapshot(snapshot: Any | None) -> str:
    if snapshot is None:
        return ""
    project = str(getattr(snapshot, "current_project", "") or "")
    if project:
        return project
    for session in getattr(snapshot, "vscode_sessions", ()) or ():
        folder = str(getattr(session, "folder", "") or "")
        if folder:
            return folder
    return ""


def _files_from_snapshot(snapshot: Any | None) -> str:
    if snapshot is None:
        return ""
    files = list(getattr(snapshot, "recently_modified_files", ()) or ())
    return ", ".join(str(item) for item in files[:10])


def _project_from_text(items: list[str]) -> str:
    combined = _normalize("\n".join(items))
    if "assistenteia" in combined or "assistente ia" in combined:
        return "AssistenteIA"
    return ""


def _main_activity(project: str, messages: list[str], context_summary: str) -> str:
    text = _normalize("\n".join([context_summary, *messages]))
    if "personal model" in text:
        return "Personal Model"
    if "session reflection" in text:
        return "Session Reflection"
    if "planner" in text:
        return "desenvolvimento do Planner"
    if "session manager" in text or "sessao" in text:
        return "continuidade de sessoes"
    if "voice" in text or "microfone" in text:
        return "Voice Input"
    if project:
        return f"trabalho no projeto {project}"
    return "atividade de trabalho local"


def _next_step(project: str, messages: list[str]) -> str:
    raw_text = " ".join(messages)
    explicit = _extract_explicit_next_step(raw_text)
    if explicit:
        return explicit
    text = _normalize(raw_text)
    if "planner" in text:
        return "criar ou estabilizar o Session Manager"
    if "session manager" in text:
        return "testar a continuidade entre sessoes"
    if project:
        return f"retomar o projeto {project} pelo ponto seguinte"
    return "rever o contexto anterior e escolher o proximo passo"


def _extract_explicit_next_step(text: str) -> str:
    import re

    patterns = (
        r"pr[oó]ximo passo (?:e|é|era|seria)\s+([^|.]+)",
        r"ficou pendente\s+([^|.]+)",
        r"devemos continuar por\s+([^|.]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _shorten(match.group(1).strip(" .:"), 180)
    return ""


def _extract_decision_text(text: str) -> str:
    compact = " ".join(text.split())
    if "|" in compact:
        compact = compact.split("|", 1)[-1].strip()
    compact = compact.replace("Assistente:", "").replace("Utilizador:", "").strip()
    return compact.strip(" .:")


def _is_relevant_message(text: str) -> bool:
    normalized = _normalize(text)
    return any(
        word in normalized
        for word in (
            "projeto",
            "planner",
            "session",
            "sessao",
            "tarefa",
            "desktop",
            "voice",
            "microfone",
            "ficheiro",
            "decid",
            "proximo passo",
        )
    )


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _normalize(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
