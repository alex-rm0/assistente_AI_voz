from __future__ import annotations

from datetime import date, datetime
from typing import Any

from assistant.briefing import get_last_active_project


NO_USEFUL_CONTEXT = "Ainda não tenho contexto suficiente para te dar um resumo útil."


def generate_greeting(
    long_term_memory: Any | None = None,
    context_observer: Any | None = None,
    user_name: str = "Alexandre",
    now: datetime | None = None,
) -> str:
    current = now or datetime.now()
    greeting = _greeting_for_hour(current.hour, user_name)
    if long_term_memory is None:
        return f"{greeting}."

    today_tasks = _today_heading(_clean_empty_tasks(_safe_call(long_term_memory, "tasks_for_date", current.date())))
    overdue_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "overdue_tasks", current.date()))
    if not today_tasks and not overdue_tasks:
        return f"{greeting}."

    briefing = generate_daily_briefing(long_term_memory, context_observer, current.date(), user_name=user_name)
    return f"{greeting}.\n\n{briefing}"


def generate_daily_briefing(
    long_term_memory: Any,
    context_observer: Any | None = None,
    today: date | None = None,
    user_name: str = "Alexandre",
) -> str:
    current_date = today or date.today()
    today_tasks = _today_heading(_clean_empty_tasks(_safe_call(long_term_memory, "tasks_for_date", current_date)))
    overdue_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "overdue_tasks", current_date))
    pending_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "pending_tasks"))
    project = _project_name(context_observer)

    lines = ["Aqui está o ponto de situação para hoje:"]

    task_summary = generate_task_summary(
        today_tasks=today_tasks,
        overdue_tasks=overdue_tasks,
        pending_tasks=pending_tasks,
    )
    lines.append(task_summary)

    if project:
        lines.append(f"O projeto que parece mais presente neste momento é {project}.")

    context_summary = generate_context_summary(context_observer)
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
) -> str:
    current_date = today or date.today()
    if long_term_memory is not None:
        today_tasks = _today_heading(_clean_empty_tasks(_safe_call(long_term_memory, "tasks_for_date", current_date)))
        overdue_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "overdue_tasks", current_date))
        pending_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "pending_tasks"))

    parts: list[str] = []
    if overdue_tasks:
        parts.append(overdue_tasks)
    if today_tasks:
        parts.append(today_tasks)
    if not today_tasks and not overdue_tasks and pending_tasks:
        parts.append(pending_tasks)

    if not parts:
        return "Não encontrei tarefas pendentes relevantes."
    return "\n".join(parts)


def generate_session_resume(
    long_term_memory: Any,
    context_observer: Any | None = None,
    today: date | None = None,
) -> str:
    current_date = today or date.today()
    timeline = _clean_empty_timeline(_safe_call(long_term_memory, "current_work_context"))
    project = _project_name(context_observer)
    pending_tasks = _clean_empty_tasks(_safe_call(long_term_memory, "pending_tasks"))

    lines: list[str] = []
    if project:
        lines.append(f"Parece que ficámos mais ligados ao projeto {project}.")
    if timeline:
        lines.append(_naturalize_timeline(timeline))
    if pending_tasks:
        lines.append(generate_task_summary(pending_tasks=pending_tasks, today=current_date))

    if not lines:
        return NO_USEFUL_CONTEXT
    return "\n\n".join(lines)


def generate_context_summary(context_observer: Any | None = None) -> str:
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
            parts.append(f"estás a trabalhar no projeto {current_project}")
        if active_app:
            parts.append(f"tens {active_app} como aplicação ativa")
        if active_window:
            parts.append(f"a janela atual é {active_window}")
    if summary is not None and getattr(summary, "summary", ""):
        parts.append(f"a atividade recente indica: {getattr(summary, 'summary')}")

    if not parts:
        return ""
    return "Do contexto observado, " + "; ".join(parts) + "."


def _greeting_for_hour(hour: int, user_name: str) -> str:
    if 6 <= hour <= 11:
        return f"Bom dia, {user_name}"
    if 12 <= hour <= 19:
        return f"Boa tarde, {user_name}"
    return f"Olá {user_name}"


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


def _naturalize_timeline(timeline: str) -> str:
    text = timeline.replace("Mudanca automatica de modo de presenca", "teste dos modos de presença")
    text = text.replace("ACTIVE_CONVERSATION -> OFFLINE", "conversa ativa para offline")
    text = text.replace("PASSIVE_MONITORING", "modo observador")
    text = text.replace("FOCUS_MODE", "modo foco")
    text = text.replace("PRIVATE_MODE", "modo privado")
    text = text.replace("ACTIVE_CONVERSATION", "modo conversa")
    text = text.replace("OFFLINE", "offline")
    if "teste dos modos de presença" in text:
        return "Recentemente estiveste a testar os modos de presença do assistente."
    return "Pelo histórico recente, estivemos a trabalhar nestes pontos:\n" + "\n".join(_bullet_items(text)[:5])


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


def _safe_call(target: Any, method_name: str, *args: Any) -> str:
    method = getattr(target, method_name, None)
    if method is None:
        return ""
    try:
        result = method(*args)
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
