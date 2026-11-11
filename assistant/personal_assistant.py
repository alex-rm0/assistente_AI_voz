from __future__ import annotations

from datetime import date, datetime
from typing import Any

from assistant.briefing import get_last_active_project
from assistant.response_language import localize


NO_USEFUL_CONTEXT = "Ainda não tenho contexto suficiente para te dar um resumo útil."


def generate_greeting(
    long_term_memory: Any | None = None,
    context_observer: Any | None = None,
    user_name: str = "Alexandre",
    now: datetime | None = None,
    response_language: str = "pt-PT",
) -> str:
    current = now or datetime.now()
    greeting = _greeting_for_hour(current.hour, user_name, response_language)
    if long_term_memory is None:
        return f"{greeting}."

    today_tasks = _today_heading(_clean_empty_tasks(_safe_call(long_term_memory, "tasks_for_date", current.date(), response_language=response_language)))
    overdue_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "overdue_tasks", current.date(), response_language=response_language))
    if not today_tasks and not overdue_tasks:
        return f"{greeting}."

    briefing = generate_daily_briefing(long_term_memory, context_observer, current.date(), user_name=user_name, response_language=response_language)
    return f"{greeting}.\n\n{briefing}"


def generate_daily_briefing(
    long_term_memory: Any,
    context_observer: Any | None = None,
    today: date | None = None,
    user_name: str = "Alexandre",
    response_language: str = "pt-PT",
) -> str:
    current_date = today or date.today()
    today_tasks = _today_heading(_clean_empty_tasks(_safe_call(long_term_memory, "tasks_for_date", current_date, response_language=response_language)))
    overdue_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "overdue_tasks", current_date, response_language=response_language))
    pending_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "pending_tasks", response_language=response_language))
    project = _project_name(context_observer)

    lines = [localize("personal.briefing", response_language)]

    task_summary = generate_task_summary(
        today_tasks=today_tasks,
        overdue_tasks=overdue_tasks,
        pending_tasks=pending_tasks,
        response_language=response_language,
    )
    lines.append(task_summary)

    if project:
        lines.append(localize("personal.project", response_language, project=project))

    context_summary = generate_context_summary(context_observer, response_language)
    if context_summary:
        lines.append(context_summary)

    if len(lines) == 1:
        return f"Não encontrei tarefas ou contexto relevante para hoje, {user_name}."
    return "\n\n".join(lines)


def generate_task_summary(
    long_term_memory: Any | None = None,
    today: date | None = None,
    *,
    today_tasks: str = "",
    overdue_tasks: str = "",
    pending_tasks: str = "",
    response_language: str = "pt-PT",
) -> str:
    current_date = today or date.today()
    if long_term_memory is not None:
        today_tasks = _today_heading(_clean_empty_tasks(_safe_call(long_term_memory, "tasks_for_date", current_date, response_language=response_language)))
        overdue_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "overdue_tasks", current_date, response_language=response_language))
        pending_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "pending_tasks", response_language=response_language))

    parts: list[str] = []
    if overdue_tasks:
        parts.append(overdue_tasks)
    if today_tasks:
        parts.append(today_tasks)
    if not today_tasks and not overdue_tasks and pending_tasks:
        parts.append(pending_tasks)

    if not parts:
        return localize("personal.no_tasks", response_language)
    return "\n".join(parts)


def generate_session_resume(
    long_term_memory: Any,
    context_observer: Any | None = None,
    today: date | None = None,
    response_language: str = "pt-PT",
) -> str:
    current_date = today or date.today()
    timeline = _clean_empty_timeline(_safe_call(long_term_memory, "current_work_context"))
    project = _project_name(context_observer)
    pending_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "pending_tasks", response_language=response_language))

    lines: list[str] = []
    if project:
        lines.append(localize("personal.resume_project", response_language, project=project))
    if timeline:
        lines.append(_naturalize_timeline(timeline, response_language))
    if pending_tasks:
        lines.append(generate_task_summary(pending_tasks=pending_tasks, today=current_date, response_language=response_language))

    if not lines:
        return localize("personal.no_context", response_language)
    return "\n\n".join(lines)


def generate_context_summary(context_observer: Any | None = None, response_language: str = "pt-PT") -> str:
    if context_observer is None:
        return ""

    snapshot = _safe_observer_call(context_observer, "latest_snapshot")
    summary = _safe_observer_call(context_observer, "latest_summary")
    if snapshot is None and summary is None:
        return ""

    parts: list[str] = []
    if snapshot is not None:
        active_app = getattr(snapshot, "active_app", "")
        active_window = getattr(snapshot, "active_window", "")
        current_project = getattr(snapshot, "current_project", "")
        if current_project:
            parts.append(localize("personal.observed_project", response_language, project=current_project))
        if active_app:
            parts.append(localize("personal.observed_app", response_language, app=active_app))
        if active_window:
            parts.append(localize("personal.observed_window", response_language, window=active_window))
    if summary is not None and getattr(summary, "summary", ""):
        parts.append(localize("personal.recent", response_language, summary=getattr(summary, "summary")))

    if not parts:
        return ""
    return localize("personal.observed", response_language) + "; ".join(parts) + "."


def _greeting_for_hour(hour: int, user_name: str, response_language: str = "pt-PT") -> str:
    if 6 <= hour <= 11:
        return localize("personal.morning", response_language, name=user_name)
    if 12 <= hour <= 19:
        return localize("personal.afternoon", response_language, name=user_name)
    return localize("personal.hello", response_language, name=user_name)


def _project_name(context_observer: Any | None) -> str:
    if context_observer is None:
        return ""
    value = get_last_active_project(context_observer)
    marker = "Último projeto ativo observado: "
    if value.startswith(marker):
        return value[len(marker) :].strip()
    return ""


def _naturalize_task_block(block: str) -> str:
    tasks = _bullet_items(block)
    if not tasks:
        return ""
    return "\n".join(f"- {item}" for item in tasks[:5])


def _naturalize_timeline(timeline: str, response_language: str = "pt-PT") -> str:
    text = timeline.replace("Mudanca automatica de modo de presenca", "teste dos modos de presença")
    text = text.replace("ACTIVE_CONVERSATION -> OFFLINE", "conversa ativa para offline")
    text = text.replace("PASSIVE_MONITORING", "modo observador")
    text = text.replace("FOCUS_MODE", "modo foco")
    text = text.replace("PRIVATE_MODE", "modo privado")
    text = text.replace("ACTIVE_CONVERSATION", "modo conversa")
    text = text.replace("OFFLINE", "offline")
    if "teste dos modos de presença" in text:
        return localize("personal.presence_tests", response_language)
    return localize("personal.timeline", response_language) + "\n" + "\n".join(_bullet_items(text)[:5])


def _bullet_items(block: str) -> list[str]:
    items: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:])
    return items


def _count_bullets(block: str) -> int:
    return max(1, len(_bullet_items(block)))


def _clean_empty_tasks(value: str) -> str:
    text = value.strip()
    lower = text.lower()
    if not text:
        return ""
    empty_markers = (
        "nao tens tarefas",
        "não tens tarefas",
        "sem tarefas pendentes",
        "no pending tasks",
        "you have no tasks",
        "no overdue tasks",
    )
    if any(marker in lower for marker in empty_markers):
        return ""
    return text


def _today_heading(value: str) -> str:
    return value.replace("Para esse dia, tens", "Para hoje, tens", 1)


def _clean_empty_timeline(value: str) -> str:
    text = value.strip()
    lower = text.lower()
    if not text:
        return ""
    if "ainda nao tenho" in lower or "ainda não tenho" in lower:
        return ""
    return text


def _safe_call(target: Any, method_name: str, *args: Any, response_language: str = "pt-PT") -> str:
    method = getattr(target, method_name, None)
    if method is None:
        return ""
    try:
        import inspect
        kwargs = {"response_language": response_language} if "response_language" in inspect.signature(method).parameters else {}
        result = method(*args, **kwargs)
    except Exception:
        return ""
    return str(result or "").strip()


def _safe_observer_call(target: Any, method_name: str, *args: Any) -> Any | None:
    method = getattr(target, method_name, None)
    if method is None:
        return None
    try:
        return method(*args)
    except Exception:
        return None
