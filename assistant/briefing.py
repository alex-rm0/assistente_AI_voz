from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


NO_DATA_MESSAGE = "Ainda não tenho dados suficientes para preparar um resumo útil."


@dataclass(frozen=True)
class BriefingData:
    today_tasks: str = ""
    pending_tasks: str = ""
    recent_timeline: str = ""
    yesterday_timeline: str = ""
    observed_context: str = ""
    recent_activity: str = ""
    last_project: str = ""


def generate_startup_briefing_prompt(user_name: str = "Alexandre") -> str:
    return f"Bom dia, {user_name}. Tenho um resumo preparado para hoje. Queres ver?"


def generate_daily_briefing(
    long_term_memory: Any,
    context_observer: Any | None = None,
    today: date | None = None,
) -> str:
    current_date = today or date.today()
    data = _collect_briefing_data(long_term_memory, context_observer, current_date)

    facts = _compact_items(
        data.observed_context,
        data.recent_timeline,
        data.recent_activity,
    )
    tasks = _compact_items(data.today_tasks, data.pending_tasks)
    inferences = _daily_inferences(data)

    if not facts and not tasks and not inferences:
        return NO_DATA_MESSAGE

    return _format_sections(
        title="Resumo para hoje",
        facts=facts,
        tasks=tasks,
        inferences=inferences,
    )


def generate_session_continuity_summary(
    long_term_memory: Any,
    context_observer: Any | None = None,
    today: date | None = None,
) -> str:
    current_date = today or date.today()
    data = _collect_briefing_data(long_term_memory, context_observer, current_date)

    facts = _compact_items(
        _last_project_fact(data.last_project),
        data.observed_context,
        data.recent_timeline,
    )
    tasks = _compact_items(data.pending_tasks, data.today_tasks)
    inferences = _continuity_inferences(data)

    if not facts and not tasks and not inferences:
        return NO_DATA_MESSAGE

    return _format_sections(
        title="Continuidade da sessão",
        facts=facts,
        tasks=tasks,
        inferences=inferences,
    )


def summarize_yesterday(long_term_memory: Any, today: date | None = None) -> str:
    current_date = today or date.today()
    yesterday = current_date - timedelta(days=1)
    timeline = _safe_call(long_term_memory, "timeline_for_date", yesterday)
    if _is_empty_timeline(timeline):
        return f"Ainda não tenho eventos registados para ontem ({yesterday.isoformat()})."
    return _format_sections(
        title="Resumo de ontem",
        facts=[timeline],
        tasks=[],
        inferences=[],
    )


def get_last_active_project(context_observer: Any | None = None) -> str:
    project = _last_project_from_observer(context_observer)
    if not project:
        return "Ainda não tenho dados suficientes para saber qual foi o último projeto ativo."
    return f"Último projeto ativo observado: {project}"


def _collect_briefing_data(
    long_term_memory: Any,
    context_observer: Any | None,
    current_date: date,
) -> BriefingData:
    return BriefingData(
        today_tasks=_without_empty_task_message(_safe_call(long_term_memory, "tasks_for_date", current_date)),
        pending_tasks=_without_empty_task_message(_safe_call(long_term_memory, "pending_tasks")),
        recent_timeline=_without_empty_timeline_message(_safe_call(long_term_memory, "current_work_context")),
        yesterday_timeline=_without_empty_timeline_message(
            _safe_call(long_term_memory, "timeline_for_date", current_date - timedelta(days=1))
        ),
        observed_context=_observed_context_fact(context_observer),
        recent_activity=_recent_activity_fact(context_observer),
        last_project=_last_project_from_observer(context_observer),
    )


def _observed_context_fact(context_observer: Any | None) -> str:
    if context_observer is None:
        return ""
    snapshot = _safe_observer_call(context_observer, "latest_snapshot")
    if snapshot is None:
        return ""

    parts: list[str] = []
    active_app = getattr(snapshot, "active_app", "")
    active_window = getattr(snapshot, "active_window", "")
    current_project = getattr(snapshot, "current_project", "")
    if active_app:
        parts.append(f"aplicação ativa: {active_app}")
    if active_window:
        parts.append(f"janela ativa: {active_window}")
    if current_project:
        parts.append(f"projeto observado: {current_project}")
    if not parts:
        return ""
    return "Contexto observado: " + "; ".join(parts) + "."


def _recent_activity_fact(context_observer: Any | None) -> str:
    if context_observer is None:
        return ""
    latest_summary = _safe_observer_call(context_observer, "latest_summary")
    if latest_summary is not None and getattr(latest_summary, "summary", ""):
        return "Atividade recente: " + getattr(latest_summary, "summary")

    activity_summary = _safe_observer_call(context_observer, "activity_summary", 3)
    if not activity_summary:
        return ""

    lines = []
    for active_app, active_window, current_project, total_seconds in activity_summary:
        minutes = max(1, round(float(total_seconds) / 60))
        project = f" no projeto {current_project}" if current_project else ""
        window = f" ({active_window})" if active_window else ""
        lines.append(f"{active_app}{window}{project}: cerca de {minutes} min")
    return "Atividade recente: " + "; ".join(lines)


def _last_project_from_observer(context_observer: Any | None) -> str:
    if context_observer is None:
        return ""
    snapshot = _safe_observer_call(context_observer, "latest_snapshot")
    if snapshot is not None and getattr(snapshot, "current_project", ""):
        return getattr(snapshot, "current_project")
    latest_summary = _safe_observer_call(context_observer, "latest_summary")
    if latest_summary is not None and getattr(latest_summary, "project", ""):
        return getattr(latest_summary, "project")
    return ""


def _last_project_fact(project: str) -> str:
    if not project:
        return ""
    return f"Último projeto ativo observado: {project}"


def _daily_inferences(data: BriefingData) -> list[str]:
    inferences: list[str] = []
    if data.last_project:
        inferences.append(f"Parece que o projeto mais relevante para retomar é {data.last_project}.")
    if data.pending_tasks or data.today_tasks:
        inferences.append("Provavelmente vale a pena começar pelas tarefas pendentes ou com data de hoje.")
    return inferences


def _continuity_inferences(data: BriefingData) -> list[str]:
    inferences: list[str] = []
    if data.last_project:
        inferences.append(f"Parece que ficámos ligados ao projeto {data.last_project}.")
    if data.recent_activity:
        inferences.append("Provavelmente a atividade recente ajuda a retomar a sessão anterior.")
    return inferences


def _format_sections(
    title: str,
    facts: list[str],
    tasks: list[str],
    inferences: list[str],
) -> str:
    lines = [title]
    lines.append("")
    lines.append("Factos observados:")
    lines.extend(f"- {item}" for item in facts) if facts else lines.append("- Sem factos observados suficientes.")
    lines.append("")
    lines.append("Tarefas pendentes:")
    lines.extend(f"- {item}" for item in tasks) if tasks else lines.append("- Não encontrei tarefas pendentes relevantes.")
    lines.append("")
    lines.append("Inferências prováveis:")
    lines.extend(f"- {item}" for item in inferences) if inferences else lines.append("- Sem inferências suficientes.")
    return "\n".join(lines)


def _compact_items(*items: str) -> list[str]:
    return [item.strip() for item in items if item and item.strip()]


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


def _without_empty_task_message(value: str) -> str:
    text = value.strip()
    lower = text.lower()
    if not text or "nao tens tarefas" in lower or "não tens tarefas" in lower:
        return ""
    return text


def _without_empty_timeline_message(value: str) -> str:
    text = value.strip()
    lower = text.lower()
    if (
        not text
        or "ainda nao tenho eventos" in lower
        or "ainda não tenho eventos" in lower
        or "ainda nao tenho eventos suficientes" in lower
        or "ainda não tenho eventos suficientes" in lower
    ):
        return ""
    return text


def _is_empty_timeline(value: str) -> bool:
    return not _without_empty_timeline_message(value)
