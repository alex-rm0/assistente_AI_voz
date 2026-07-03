from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class ProactiveSuggestion:
    key: str
    message: str
    score: int = 0


def generate_proactive_suggestions(
    long_term_memory: Any,
    context_observer: Any | None = None,
    today: date | None = None,
    limit: int = 3,
) -> list[ProactiveSuggestion]:
    current_date = today or date.today()
    suggestions: list[ProactiveSuggestion] = []

    today_tasks = _safe_call(long_term_memory, "tasks_for_date", current_date)
    if _has_tasks(today_tasks):
        project = _project_from_text(today_tasks)
        if project:
            suggestions.append(
                ProactiveSuggestion(
                    key=f"task_today_project:{project.lower()}",
                    message=f"Tens uma tarefa para hoje relacionada com o projeto {project}.",
                    score=90,
                )
            )
        else:
            suggestions.append(
                ProactiveSuggestion(
                    key="task_today",
                    message="Tens uma tarefa marcada para hoje.",
                    score=80,
                )
            )

    overdue_tasks = _safe_call(long_term_memory, "overdue_tasks", current_date)
    if _has_tasks(overdue_tasks):
        project = _project_from_text(overdue_tasks)
        if project:
            suggestions.append(
                ProactiveSuggestion(
                    key=f"overdue_task_project:{project.lower()}",
                    message=f"Tens uma tarefa atrasada relacionada com o projeto {project}.",
                    score=85,
                )
            )
        else:
            suggestions.append(
                ProactiveSuggestion(
                    key="overdue_task",
                    message="Tens uma tarefa atrasada que talvez valha a pena fechar hoje.",
                    score=75,
                )
            )

    timeline = _safe_call(long_term_memory, "current_work_context")
    if "context observer" in timeline.lower():
        suggestions.append(
            ProactiveSuggestion(
                key="timeline_context_observer",
                message="Ontem estiveste a trabalhar no Context Observer. Pode valer a pena retomar por aí.",
                score=70,
            )
        )
    elif "assistenteia" in timeline.lower():
        suggestions.append(
            ProactiveSuggestion(
                key="timeline_assistenteia",
                message="Parece que tens retomado trabalho no AssistenteIA recentemente.",
                score=60,
            )
        )

    snapshot = _safe_observer_call(context_observer, "latest_snapshot")
    if snapshot is not None:
        project = getattr(snapshot, "current_project", "") or _project_from_windows(snapshot)
        apps = _observed_apps(snapshot)
        if project and {"VS Code", "Codex"}.issubset(apps):
            suggestions.append(
                ProactiveSuggestion(
                    key=f"vscode_codex_same_project:{project.lower()}",
                    message=f"Tens o VS Code e o Codex abertos no mesmo projeto: {project}.",
                    score=95,
                )
            )
        elif project:
            suggestions.append(
                ProactiveSuggestion(
                    key=f"resumed_project:{project.lower()}",
                    message=f"Parece que retomaste o trabalho no projeto {project}.",
                    score=65,
                )
            )

    return _unique_suggestions(suggestions)[:limit]


def next_proactive_suggestion(
    long_term_memory: Any,
    context_observer: Any | None = None,
    today: date | None = None,
) -> str:
    current_date = today or date.today()
    for suggestion in generate_proactive_suggestions(long_term_memory, context_observer, current_date):
        if _was_suggestion_shown(long_term_memory, suggestion.key, current_date):
            continue
        _mark_suggestion_shown(long_term_memory, suggestion.key, current_date)
        return suggestion.message
    return ""


def _was_suggestion_shown(long_term_memory: Any, key: str, current_date: date) -> bool:
    getter = getattr(long_term_memory, "get_preference", None)
    if getter is None:
        return False
    return getter(_preference_key(key, current_date), "") == "shown"


def _mark_suggestion_shown(long_term_memory: Any, key: str, current_date: date) -> None:
    setter = getattr(long_term_memory, "set_preference", None)
    if setter is None:
        return
    setter(_preference_key(key, current_date), "shown")


def _preference_key(key: str, current_date: date) -> str:
    return f"proactive_suggestion:{current_date.isoformat()}:{key}"


def _has_tasks(value: str) -> bool:
    text = value.strip().lower()
    return bool(text) and "nao tens tarefas" not in text and "não tens tarefas" not in text


def _project_from_text(text: str) -> str:
    marker = "projeto:"
    for line in text.splitlines():
        lower = line.lower()
        if marker not in lower:
            continue
        start = lower.find(marker) + len(marker)
        value = line[start:].strip(" ).;")
        if value:
            return value
    if "assistenteia" in text.lower():
        return "AssistenteIA"
    return ""


def _project_from_windows(snapshot: Any) -> str:
    for window in getattr(snapshot, "open_windows", ()) or ():
        title = getattr(window, "title", "")
        if "assistenteia" in title.lower():
            return "AssistenteIA"
    return ""


def _observed_apps(snapshot: Any) -> set[str]:
    apps: set[str] = set()
    active_app = getattr(snapshot, "active_app", "")
    apps.add(_friendly_app(active_app))
    for window in getattr(snapshot, "open_windows", ()) or ():
        apps.add(_friendly_app(getattr(window, "process_name", "") or getattr(window, "title", "")))
    return {app for app in apps if app}


def _friendly_app(value: str) -> str:
    text = value.lower()
    if "code.exe" in text or "visual studio code" in text:
        return "VS Code"
    if "codex" in text:
        return "Codex"
    if "chrome" in text:
        return "Chrome"
    return value


def _unique_suggestions(suggestions: list[ProactiveSuggestion]) -> list[ProactiveSuggestion]:
    by_key: dict[str, ProactiveSuggestion] = {}
    for suggestion in sorted(suggestions, key=lambda item: item.score, reverse=True):
        by_key.setdefault(suggestion.key, suggestion)
    return list(by_key.values())


def _safe_call(target: Any, method_name: str, *args: Any) -> str:
    method = getattr(target, method_name, None)
    if method is None:
        return ""
    try:
        result = method(*args)
    except Exception:
        return ""
    return str(result or "").strip()


def _safe_observer_call(target: Any | None, method_name: str, *args: Any) -> Any | None:
    if target is None:
        return None
    method = getattr(target, method_name, None)
    if method is None:
        return None
    try:
        return method(*args)
    except Exception:
        return None
