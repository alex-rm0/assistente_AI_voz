from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from datetime import date, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from assistant.task_formatter import format_task_collection_for_assistant, format_tasks_panel


DEFAULT_MEMORY_DB = "long_term_memory.sqlite"


class MemoryCategory(str, Enum):
    USER_PROFILE = "perfil_utilizador"
    PROJECTS = "projetos"
    CONVERSATIONS = "conversas"
    PREFERENCES = "preferencias"
    TASKS = "tarefas"
    RELATIONSHIPS = "relacoes"


class Embedder(Protocol):
    def embed(self, text: str) -> list[float] | None:
        ...


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    category: str
    content: str
    score: float = 0.0


@dataclass(frozen=True)
class TimelineEvent:
    id: int
    event_date: str
    content: str
    project: str = ""
    people: str = ""


@dataclass(frozen=True)
class TaskRecord:
    id: int
    title: str
    description: str = ""
    due_date: str = ""
    project: str = ""
    status: str = "pending"
    priority: str = "normal"


class LongTermMemory:
    """Permanent SQLite memory, separate from transient conversation history."""

    def __init__(self, data_path: Path, db_file: str = DEFAULT_MEMORY_DB, embedder: Embedder | None = None) -> None:
        self.data_path = data_path.resolve()
        self.db_path = (self.data_path / db_file).resolve()
        self.embedder = embedder
        self._ensure_inside_data()
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def remember(self, content: str, category: str | None = None) -> str:
        text = content.strip()
        if not text:
            return "Diz-me o que queres que eu memorize."

        memory_category = normalize_category(category) if category else classify_memory(text)
        embedding = self._embed_json(text)

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO memories (category, content, embedding)
                VALUES (?, ?, ?)
                """,
                (memory_category, text, embedding),
            )

        return f"Memorizei isto em {memory_category}: {text}"

    def forget(self, query: str) -> str:
        text = query.strip()
        if not text:
            return "Diz-me o que queres que eu esqueca."

        matches = self.search(text, limit=5)
        if not matches:
            return f"Nao encontrei memorias sobre '{text}'."

        best = matches[0]
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DELETE FROM memories WHERE id = ?", (best.id,))

        return f"Esqueci esta memoria: {best.content}"

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        text = query.strip()
        if not text:
            return []

        # Use Ollama embeddings when possible; otherwise fall back to a small
        # text search so memory still works offline or with unsupported models.
        records = self._all_records()
        query_embedding = self._embed(text)
        if query_embedding:
            scored = [
                MemoryRecord(record.id, record.category, record.content, _cosine_similarity(query_embedding, embedding))
                for record, embedding in records
                if embedding
            ]
            scored = [record for record in scored if record.score > 0]
            scored.sort(key=lambda record: record.score, reverse=True)
            if scored:
                return scored[:limit]

        return self._text_search(text, limit)

    def answer_about(self, query: str) -> str:
        matches = self.search(query, limit=5)
        if not matches:
            return f"Nao tenho memoria permanente sobre '{query}'."

        lines = [f"- [{record.category}] {record.content}" for record in matches]
        return "Sei isto na memoria permanente:\n" + "\n".join(lines)

    def remember_timeline_event(self, content: str, event_date: date | None = None) -> str:
        text = content.strip()
        if not text:
            return "Diz-me que evento queres registar na timeline."

        resolved_date = event_date or infer_event_date(text)
        project = extract_project(text)
        people = ", ".join(extract_people(text))

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO timeline_events (event_date, content, project, people)
                VALUES (?, ?, ?, ?)
                """,
                (resolved_date.isoformat(), text, project, people),
            )

        details = []
        if project:
            details.append(f"projeto: {project}")
        if people:
            details.append(f"pessoas: {people}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"Registei na timeline em {resolved_date.isoformat()}: {text}{suffix}"

    def remember_context_summary(
        self,
        summary: str,
        event_date: date | None = None,
        project: str = "",
    ) -> str:
        text = summary.strip()
        if not text:
            return "Resumo de contexto vazio."

        embedding = self._embed_json(text)
        resolved_date = event_date or date.today()
        project_name = project.strip() or extract_project(text)

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO memories (category, content, embedding)
                VALUES (?, ?, ?)
                """,
                (MemoryCategory.PROJECTS.value, text, embedding),
            )
            connection.execute(
                """
                INSERT INTO timeline_events (event_date, content, project, people)
                VALUES (?, ?, ?, '')
                """,
                (resolved_date.isoformat(), text, project_name),
            )

        return f"Guardei o resumo de contexto: {text}"

    def timeline_for_date(self, event_date: date) -> str:
        events = self._timeline_events_for_range(event_date, event_date)
        if not events:
            return f"Nao tenho eventos registados para {event_date.isoformat()}."
        return _format_timeline_events(f"Eventos de {event_date.isoformat()}:", events)

    def timeline_for_period(self, start_date: date, end_date: date) -> str:
        events = self._timeline_events_for_range(start_date, end_date)
        if not events:
            return f"Nao tenho eventos registados entre {start_date.isoformat()} e {end_date.isoformat()}."
        return _format_timeline_events(
            f"Eventos entre {start_date.isoformat()} e {end_date.isoformat()}:",
            events,
        )

    def current_work_context(self) -> str:
        events = self._timeline_events(limit=5)
        if not events:
            return "Ainda nao tenho eventos suficientes para saber em que estavamos a trabalhar."
        return _format_timeline_events("Ultimos temas/projetos em que trabalhamos:", events)

    def project_start(self, project: str | None = None) -> str:
        project_name = (project or "").strip()
        with sqlite3.connect(self.db_path) as connection:
            if project_name:
                row = connection.execute(
                    """
                    SELECT id, event_date, content, project, people
                    FROM timeline_events
                    WHERE LOWER(project) LIKE ?
                    ORDER BY event_date ASC, id ASC
                    LIMIT 1
                    """,
                    (f"%{project_name.lower()}%",),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT id, event_date, content, project, people
                    FROM timeline_events
                    WHERE project != ''
                    ORDER BY event_date ASC, id ASC
                    LIMIT 1
                    """
                ).fetchone()

        if row is None:
            return "Ainda nao tenho um inicio de projeto registado na timeline."

        event = _timeline_event_from_row(row)
        project_label = event.project or project_name or "este projeto"
        return f"Comecamos {project_label} em {event.event_date}: {event.content}"

    def create_task(
        self,
        title: str,
        description: str = "",
        due_date: date | None = None,
        project: str | None = None,
        priority: str = "normal",
    ) -> str:
        task_title = title.strip()
        if not task_title:
            return "Diz-me qual e a tarefa que queres guardar."

        task_due_date = due_date or infer_task_due_date(task_title)
        task_project = (project or extract_project(task_title)).strip()
        task_description = description.strip()
        task_priority = normalize_task_priority(priority or infer_task_priority(task_title))
        embedding = self._embed_json(task_title)

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO tasks (title, description, due_date, project, status, priority, embedding)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    task_title,
                    task_description,
                    task_due_date.isoformat() if task_due_date else "",
                    task_project,
                    task_priority,
                    embedding,
                ),
            )
            connection.execute(
                """
                INSERT INTO memories (category, content, embedding)
                VALUES (?, ?, ?)
                """,
                (
                    MemoryCategory.TASKS.value,
                    task_title,
                    embedding,
                ),
            )

        details = []
        if task_due_date:
            details.append(f"para {task_due_date.isoformat()}")
        if task_project:
            details.append(f"ligada ao projeto {task_project}")
        if task_priority != "normal":
            details.append(f"com prioridade {task_priority}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"Guardei a tarefa: {task_title}{suffix}"

    def tasks_for_date(self, due_date: date, show_details: bool = False) -> str:
        tasks = self._tasks_for_date(due_date)
        if not tasks:
            return f"Nao tens tarefas registadas para {due_date.isoformat()}."
        return format_task_collection_for_assistant(
            tasks,
            "Para esse dia, tens" if due_date != date.today() else "Para hoje, tens",
            show_details=show_details,
        )

    def tasks_for_today(self, show_details: bool = False) -> str:
        return self.tasks_for_date(date.today(), show_details=show_details)

    def overdue_tasks(self, today: date | None = None, show_details: bool = False) -> str:
        current = today or date.today()
        tasks = self._overdue_tasks(current)
        if not tasks:
            return "Nao tens tarefas atrasadas."
        return format_task_collection_for_assistant(
            tasks,
            "Tens em atraso",
            show_details=show_details,
        )

    def tasks_for_week(self, start_date: date | None = None, show_details: bool = False) -> str:
        start = start_date or date.today()
        end = start + timedelta(days=6)
        tasks = self._tasks_for_range(start, end)
        if not tasks:
            return f"Nao tens tarefas registadas entre {start.isoformat()} e {end.isoformat()}."
        return format_task_collection_for_assistant(
            tasks,
            "Esta semana, tens",
            show_details=show_details,
        )

    def pending_tasks(self, limit: int = 10, show_details: bool = False) -> str:
        tasks = self._pending_tasks(limit=limit)
        if not tasks:
            return "Nao tens tarefas pendentes registadas."
        return format_task_collection_for_assistant(
            tasks,
            "Tens pendente",
            show_details=show_details,
        )

    def complete_task(self, query: str = "") -> str:
        return self._update_task_status(query, "completed")

    def cancel_task(self, query: str = "") -> str:
        return self._update_task_status(query, "cancelled")

    def postpone_task(self, query: str = "", due_date: date | None = None) -> str:
        task = self._find_pending_task(query)
        inferred_due_date = due_date or infer_task_due_date(query)
        if task is None and inferred_due_date is not None:
            task = self._find_pending_task("")
        if task is None:
            return "Nao encontrei uma tarefa pendente para adiar."

        new_due_date = inferred_due_date or (date.today() + timedelta(days=1))
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE tasks SET due_date = ? WHERE id = ?",
                (new_due_date.isoformat(), task.id),
            )
        return f"Adiei a tarefa '{task.title}' para {new_due_date.isoformat()}."

    def task_panel_summary(self, limit: int = 8, show_details: bool = False) -> str:
        tasks = self._pending_tasks(limit=limit)
        if not tasks:
            return "Sem tarefas pendentes."
        return format_tasks_panel(tasks, show_details=show_details)

    def pending_task_count(self) -> int:
        return len(self._pending_tasks(limit=200))

    def set_preference(self, key: str, value: str) -> None:
        preference_key = key.strip()
        preference_value = value.strip()
        if not preference_key:
            return

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO preferences (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (preference_key, preference_value),
            )

    def get_preference(self, key: str, default: str = "") -> str:
        preference_key = key.strip()
        if not preference_key:
            return default

        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT value FROM preferences WHERE key = ?",
                (preference_key,),
            ).fetchone()

        if row is None:
            return default
        return row[0] or default

    def context_for(self, query: str, limit: int = 5) -> str:
        matches = self.search(query, limit=limit)
        timeline = self._timeline_context_for(query, limit=3)
        tasks = self._task_context_for(query, limit=3)
        if not matches and not timeline and not tasks:
            return ""

        lines = [f"- [{record.category}] {record.content}" for record in matches]
        lines.extend(f"- [timeline:{event.event_date}] {event.content}" for event in timeline)
        lines.extend(
            f"- [tarefa:{task.due_date or 'sem_data'}] {task.title}"
            for task in tasks
        )
        return "\n".join(lines)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_category
                ON memories(category)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_created_at
                ON memories(created_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS timeline_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_date TEXT NOT NULL,
                    content TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT '',
                    people TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timeline_event_date
                ON timeline_events(event_date)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timeline_project
                ON timeline_events(project)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    due_date TEXT NOT NULL DEFAULT '',
                    project TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    embedding TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._ensure_task_columns(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_due_date
                ON tasks(due_date)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_project
                ON tasks(project)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._migrate_categories(connection)

    def _ensure_task_columns(self, connection: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "description" not in columns:
            connection.execute("ALTER TABLE tasks ADD COLUMN description TEXT NOT NULL DEFAULT ''")
        if "priority" not in columns:
            connection.execute("ALTER TABLE tasks ADD COLUMN priority TEXT NOT NULL DEFAULT 'normal'")

    def _migrate_categories(self, connection: sqlite3.Connection) -> None:
        legacy_categories = {
            "preferencia": MemoryCategory.PREFERENCES.value,
            "projeto": MemoryCategory.PROJECTS.value,
            "projecto": MemoryCategory.PROJECTS.value,
            "contexto": MemoryCategory.CONVERSATIONS.value,
        }
        for old_category, new_category in legacy_categories.items():
            connection.execute(
                "UPDATE memories SET category = ? WHERE category = ?",
                (new_category, old_category),
            )

    def _all_records(self) -> list[tuple[MemoryRecord, list[float] | None]]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT id, category, content, embedding FROM memories ORDER BY created_at DESC"
            ).fetchall()

        records: list[tuple[MemoryRecord, list[float] | None]] = []
        for row in rows:
            embedding = _loads_embedding(row[3])
            records.append((MemoryRecord(id=row[0], category=row[1], content=row[2]), embedding))
        return records

    def _text_search(self, query: str, limit: int) -> list[MemoryRecord]:
        terms = _terms(query)
        if not terms:
            return []

        scored: list[MemoryRecord] = []
        for record, _embedding in self._all_records():
            content = record.content.lower()
            category = record.category.lower()
            score = sum(1 for term in terms if term in content or term in category)
            if score > 0:
                scored.append(MemoryRecord(record.id, record.category, record.content, float(score)))

        scored.sort(key=lambda record: record.score, reverse=True)
        return scored[:limit]

    def _timeline_events_for_range(self, start_date: date, end_date: date) -> list[TimelineEvent]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, event_date, content, project, people
                FROM timeline_events
                WHERE event_date BETWEEN ? AND ?
                ORDER BY event_date ASC, id ASC
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
        return [_timeline_event_from_row(row) for row in rows]

    def _timeline_events(self, limit: int = 10) -> list[TimelineEvent]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, event_date, content, project, people
                FROM timeline_events
                ORDER BY event_date DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_timeline_event_from_row(row) for row in rows]

    def _timeline_context_for(self, query: str, limit: int = 3) -> list[TimelineEvent]:
        terms = _terms(query)
        if not terms:
            return []

        scored: list[tuple[int, TimelineEvent]] = []
        for event in self._timeline_events(limit=50):
            haystack = _normalize_text(" ".join((event.content, event.project, event.people)))
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, event))

        scored.sort(key=lambda item: (item[0], item[1].event_date), reverse=True)
        return [event for _score, event in scored[:limit]]

    def _tasks_for_date(self, due_date: date) -> list[TaskRecord]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, title, description, due_date, project, status, priority
                FROM tasks
                WHERE due_date = ? AND status = 'pending'
                ORDER BY id ASC
                """,
                (due_date.isoformat(),),
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def _pending_tasks(self, limit: int = 10) -> list[TaskRecord]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, title, description, due_date, project, status, priority
                FROM tasks
                WHERE status = 'pending'
                ORDER BY
                    CASE priority WHEN 'alta' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,
                    CASE WHEN due_date = '' THEN 1 ELSE 0 END,
                    due_date ASC,
                    id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def _tasks_for_range(self, start_date: date, end_date: date) -> list[TaskRecord]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, title, description, due_date, project, status, priority
                FROM tasks
                WHERE due_date BETWEEN ? AND ? AND status = 'pending'
                ORDER BY due_date ASC,
                    CASE priority WHEN 'alta' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,
                    id ASC
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def _overdue_tasks(self, today: date) -> list[TaskRecord]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, title, description, due_date, project, status, priority
                FROM tasks
                WHERE due_date != '' AND due_date < ? AND status = 'pending'
                ORDER BY due_date ASC,
                    CASE priority WHEN 'alta' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,
                    id ASC
                """,
                (today.isoformat(),),
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def _task_context_for(self, query: str, limit: int = 3) -> list[TaskRecord]:
        terms = _terms(query)
        if not terms:
            return []

        scored: list[tuple[int, TaskRecord]] = []
        for task in self._pending_tasks(limit=50):
            haystack = _normalize_text(" ".join((task.title, task.project, task.due_date)))
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, task))

        scored.sort(key=lambda item: (item[0], item[1].due_date), reverse=True)
        return [task for _score, task in scored[:limit]]

    def _find_pending_task(self, query: str = "") -> TaskRecord | None:
        tasks = self._pending_tasks(limit=50)
        if not tasks:
            return None

        terms = _terms(query)
        if not terms or _normalize_text(query) in {"esta tarefa", "a tarefa", "tarefa", ""}:
            return tasks[0]

        scored: list[tuple[int, TaskRecord]] = []
        for task in tasks:
            haystack = _normalize_text(" ".join((task.title, task.description, task.project, task.due_date)))
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, task))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _update_task_status(self, query: str, status: str) -> str:
        task = self._find_pending_task(query)
        if task is None:
            return "Nao encontrei uma tarefa pendente para atualizar."

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE tasks SET status = ? WHERE id = ?",
                (status, task.id),
            )

        label = _task_status_label(status)
        return f"Marquei a tarefa '{task.title}' como {label}."

    def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        return self.embedder.embed(text)

    def _embed_json(self, text: str) -> str | None:
        embedding = self._embed(text)
        if not embedding:
            return None
        return json.dumps(embedding, ensure_ascii=True)

    def _ensure_inside_data(self) -> None:
        if self.db_path != self.data_path and self.data_path not in self.db_path.parents:
            raise ValueError("Long-term memory database must stay inside the data folder.")


def classify_memory(content: str) -> str:
    text = _normalize_text(content)
    if any(
        phrase in text
        for phrase in (
            "chamo-me",
            "chamo me",
            "o meu nome e",
            "meu nome e",
            "sou ",
            "moro ",
            "trabalho ",
        )
    ):
        return MemoryCategory.USER_PROFILE.value

    if any(word in text for word in ("prefiro", "preferencia", "gosto", "nao gosto", "adoro", "detesto")):
        return MemoryCategory.PREFERENCES.value

    if any(word in text for word in ("habito", "costumo", "rotina", "todos os dias", "normalmente")):
        return MemoryCategory.PREFERENCES.value

    if any(word in text for word in ("interesse", "interesses", "interesso", "quero aprender")):
        return MemoryCategory.PREFERENCES.value

    if any(word in text for word in ("projeto", "projecto", "assistenteia", "app", "aplicacao")):
        return MemoryCategory.PROJECTS.value

    if any(word in text for word in ("tarefa", "pendente", "tenho de", "preciso de fazer", "lembrar de fazer")):
        return MemoryCategory.TASKS.value

    if any(
        word in text
        for word in (
            "mae",
            "pai",
            "filho",
            "filha",
            "esposa",
            "marido",
            "amigo",
            "amiga",
            "colega",
            "pessoa importante",
        )
    ):
        return MemoryCategory.RELATIONSHIPS.value

    return MemoryCategory.CONVERSATIONS.value


def normalize_category(category: str | MemoryCategory) -> str:
    if isinstance(category, MemoryCategory):
        return category.value

    text = _normalize_text(category)
    aliases = {
        "perfil": MemoryCategory.USER_PROFILE,
        "perfil_utilizador": MemoryCategory.USER_PROFILE,
        "utilizador": MemoryCategory.USER_PROFILE,
        "usuario": MemoryCategory.USER_PROFILE,
        "projeto": MemoryCategory.PROJECTS,
        "projecto": MemoryCategory.PROJECTS,
        "projetos": MemoryCategory.PROJECTS,
        "conversa": MemoryCategory.CONVERSATIONS,
        "conversas": MemoryCategory.CONVERSATIONS,
        "contexto": MemoryCategory.CONVERSATIONS,
        "preferencia": MemoryCategory.PREFERENCES,
        "preferencias": MemoryCategory.PREFERENCES,
        "tarefa": MemoryCategory.TASKS,
        "tarefas": MemoryCategory.TASKS,
        "relacao": MemoryCategory.RELATIONSHIPS,
        "relacoes": MemoryCategory.RELATIONSHIPS,
    }
    return aliases.get(text, MemoryCategory.CONVERSATIONS).value


def infer_event_date(content: str, today: date | None = None) -> date:
    current = today or date.today()
    text = _normalize_text(content)

    if "ontem" in text:
        return current - timedelta(days=1)
    if "anteontem" in text:
        return current - timedelta(days=2)
    if "semana passada" in text or "na semana passada" in text:
        return current - timedelta(days=7)

    months_match = re_search_numbered_period(text, "mes")
    if months_match is not None:
        return _subtract_months(current, months_match)

    days_match = re_search_numbered_period(text, "dia")
    if days_match is not None:
        return current - timedelta(days=days_match)

    weeks_match = re_search_numbered_period(text, "semana")
    if weeks_match is not None:
        return current - timedelta(weeks=weeks_match)

    return current


def infer_task_due_date(content: str, today: date | None = None) -> date | None:
    current = today or date.today()
    text = _normalize_text(content)

    if "amanha" in text:
        return current + timedelta(days=1)
    if "hoje" in text:
        return current
    if "depois de amanha" in text:
        return current + timedelta(days=2)
    weekday = _weekday_from_text(text)
    if weekday is not None:
        days_until = (weekday - current.weekday()) % 7
        if days_until == 0:
            days_until = 7
        return current + timedelta(days=days_until)
    if "proxima semana" in text or "próxima semana" in text:
        return current + timedelta(days=7)

    days_match = re_search_numbered_period(text, "dia")
    if days_match is not None:
        return current + timedelta(days=days_match)

    weeks_match = re_search_numbered_period(text, "semana")
    if weeks_match is not None:
        return current + timedelta(weeks=weeks_match)

    months_match = re_search_numbered_period(text, "mes")
    if months_match is not None:
        return _add_months(current, months_match)

    return None


def infer_task_priority(content: str) -> str:
    text = _normalize_text(content)
    if any(word in text for word in ("urgente", "importante", "prioridade alta", "alta prioridade")):
        return "alta"
    if any(phrase in text for phrase in ("prioridade baixa", "baixa prioridade", "quando der")):
        return "baixa"
    return "normal"


def normalize_task_priority(priority: str) -> str:
    text = _normalize_text(priority)
    if text in {"alta", "high", "urgente"}:
        return "alta"
    if text in {"baixa", "low"}:
        return "baixa"
    return "normal"


def _weekday_from_text(text: str) -> int | None:
    weekdays = {
        "segunda": 0,
        "segunda-feira": 0,
        "terca": 1,
        "terca-feira": 1,
        "terça": 1,
        "terça-feira": 1,
        "quarta": 2,
        "quarta-feira": 2,
        "quinta": 3,
        "quinta-feira": 3,
        "sexta": 4,
        "sexta-feira": 4,
        "sabado": 5,
        "sábado": 5,
        "domingo": 6,
    }
    for marker, weekday in weekdays.items():
        if marker in text:
            return weekday
    return None


def extract_project(content: str) -> str:
    normalized = _normalize_text(content)
    match = _PROJECT_PATTERN.search(content)
    if not match:
        if "assistenteia" in normalized:
            return "AssistenteIA"
        if "este projeto" in normalized or "este projecto" in normalized:
            return "este projeto"
        return ""
    project = match.group(1).strip(" .,!?:;")
    project = re.split(r"\s+com\s+", project, maxsplit=1, flags=re.IGNORECASE)[0]
    project = re.split(
        r"\s+(?:hoje|amanh\S*|depois de amanh\S*|esta semana|sexta-feira|segunda-feira|terca-feira|ter\S*a-feira|quarta-feira|quinta-feira|sabado|s\S*bado|domingo)\b",
        project,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return project.strip(" .,!?:;")


def extract_people(content: str) -> list[str]:
    people: list[str] = []
    for match in _PERSON_PATTERN.finditer(content):
        name = match.group(1).strip(" .,!?:;")
        if name and name not in people:
            people.append(name)
    return people


def re_search_numbered_period(text: str, period: str) -> int | None:
    number_words = {
        "um": 1,
        "uma": 1,
        "dois": 2,
        "duas": 2,
        "tres": 3,
        "quatro": 4,
        "cinco": 5,
        "seis": 6,
        "sete": 7,
        "oito": 8,
        "nove": 9,
        "dez": 10,
        "onze": 11,
        "doze": 12,
    }
    pattern = rf"ha\s+(\d+|{'|'.join(number_words)})\s+{period}"
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1)
    if value.isdigit():
        return int(value)
    return number_words.get(value)


def _subtract_months(current: date, months: int) -> date:
    month = current.month - months
    year = current.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(current.day, _days_in_month(year, month))
    return date(year, month, day)


def _add_months(current: date, months: int) -> date:
    month = current.month + months
    year = current.year
    while month > 12:
        month -= 12
        year += 1
    day = min(current.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def _timeline_event_from_row(row: tuple) -> TimelineEvent:
    return TimelineEvent(
        id=row[0],
        event_date=row[1],
        content=row[2],
        project=row[3] or "",
        people=row[4] or "",
    )


def _task_from_row(row: tuple) -> TaskRecord:
    return TaskRecord(
        id=row[0],
        title=row[1],
        description=row[2] or "",
        due_date=row[3] or "",
        project=row[4] or "",
        status=row[5] or "pending",
        priority=row[6] or "normal",
    )


def _format_timeline_events(title: str, events: list[TimelineEvent]) -> str:
    lines = [title]
    for event in events:
        details = []
        if event.project:
            details.append(f"projeto: {event.project}")
        if event.people:
            details.append(f"pessoas: {event.people}")
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {event.event_date}: {event.content}{suffix}")
    return "\n".join(lines)


def _format_tasks(title: str, tasks: list[TaskRecord]) -> str:
    lines = [title]
    for task in tasks:
        details = []
        if task.due_date:
            details.append(f"data: {task.due_date}")
        if task.project:
            details.append(f"projeto: {task.project}")
        if task.priority:
            details.append(f"prioridade: {task.priority}")
        if task.status:
            details.append(f"estado: {_task_status_label(task.status)}")
        suffix = f" ({'; '.join(details)})" if details else ""
        description = f" - {task.description}" if task.description else ""
        lines.append(f"- {task.title}{description}{suffix}")
    return "\n".join(lines)


def _task_status_label(status: str) -> str:
    labels = {
        "pending": "pendente",
        "completed": "concluida",
        "cancelled": "cancelada",
    }
    return labels.get(status, status)


def _loads_embedding(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [float(item) for item in data if isinstance(item, (int, float))]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [term.strip(".,;:!?()[]{}\"'") for term in normalized.split() if len(term.strip()) >= 3]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


_PROJECT_PATTERN = re.compile(
    r"\b(?:projeto|projecto)\s+([^.,;:!?]+)",
    re.IGNORECASE,
)
_PERSON_PATTERN = re.compile(
    r"\b(?:com|pessoa importante e|pessoa importante é)\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç -]{1,60})",
)
