from __future__ import annotations

from typing import Protocol

from assistant.response_language import localize


class TaskLike(Protocol):
    title: str
    description: str
    due_date: str
    project: str
    status: str
    priority: str


def format_task_for_assistant(task: TaskLike, show_details: bool = False, response_language: str = "pt-PT") -> str:
    title = _short_task_title(task.title)
    if show_details:
        return _format_task_details(task, title, response_language)

    # Phase 7.7 (section 7): the framing sentence around the task is
    # Echo-authored and localizes; task.title/description themselves are
    # the user's OWN reminder text (parsed, not translated -- section 18).
    project = (task.project or _infer_project(task.title)).strip()
    description = _natural_description(task.description or task.title, project)
    if project:
        return localize("task.reminder.project_framing", response_language, project=project, description=description)
    return description


def format_tasks_panel(
    tasks: list[TaskLike] | tuple[TaskLike, ...], show_details: bool = False, response_language: str = "pt-PT"
) -> str:
    if not tasks:
        return localize("task.panel.empty", response_language)

    lines = []
    for task in tasks:
        if show_details:
            lines.append(f"- {format_task_for_assistant(task, show_details=True, response_language=response_language)}")
        else:
            lines.append(f"- {format_task_for_assistant(task, response_language=response_language)}")
    return "\n".join(lines)


def format_task_collection_for_assistant(
    tasks: list[TaskLike] | tuple[TaskLike, ...],
    heading: str,
    show_details: bool = False,
    response_language: str = "pt-PT",
) -> str:
    if not tasks:
        return ""

    count = len(tasks)
    if show_details:
        return f"{heading}\n{format_tasks_panel(tasks, show_details=True, response_language=response_language)}"

    if count == 1:
        return f"{heading} {format_task_for_assistant(tasks[0], response_language=response_language)}."
    return (
        f"{localize('task.reminder.count_heading', response_language, heading=heading, count=count)}\n"
        f"{format_tasks_panel(tasks, response_language=response_language)}"
    )


def _format_task_details(task: TaskLike, title: str, response_language: str = "pt-PT") -> str:
    details = [title]
    if task.description:
        details.append(f"{localize('task.reminder.detail.description', response_language)} {task.description}")
    if task.due_date:
        details.append(f"{localize('task.reminder.detail.date', response_language)} {task.due_date}")
    if task.project:
        details.append(f"{localize('task.reminder.detail.project', response_language)} {task.project}")
    if task.priority:
        details.append(f"{localize('task.reminder.detail.priority', response_language)} {task.priority}")
    if task.status:
        details.append(f"{localize('task.reminder.detail.status', response_language)} {_status_label(task.status, response_language)}")
    return " | ".join(details)


def _natural_description(text: str, project: str = "") -> str:
    cleaned = _strip_noise(text)
    cleaned = _strip_due_date_words(cleaned)
    if project:
        cleaned = cleaned.replace(f"do projeto {project}", "")
    cleaned = " ".join(cleaned.split()).strip(" .:-")
    if not cleaned:
        return _short_task_title(text)
    return cleaned


def _short_task_title(text: str) -> str:
    cleaned = _strip_noise(text)
    cleaned = _strip_due_date_words(cleaned)
    cleaned = " ".join(cleaned.split()).strip(" .:-")
    if len(cleaned) <= 80:
        return cleaned
    return cleaned[:77].rstrip() + "..."


def _strip_noise(text: str) -> str:
    cleaned = text.strip()
    prefixes = (
        "lembra-me de ",
        "lembra me de ",
        "tenho de ",
        "tenho que ",
        "adiciona uma tarefa para ",
        "adiciona uma tarefa ",
        "cria uma tarefa para ",
        "cria uma tarefa ",
    )
    lowered = cleaned.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return cleaned[len(prefix) :]
    return cleaned


def _strip_due_date_words(text: str) -> str:
    import re

    return re.sub(
        r"\s+(?:amanh\S*|depois de amanh\S*|segunda-feira|terca-feira|ter\S*a-feira|quarta-feira|quinta-feira|sexta-feira|sabado|s\S*bado|domingo)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )


def _infer_project(text: str) -> str:
    lowered = text.lower()
    if "assistenteia" in lowered:
        return "AssistenteIA"
    return ""


def _status_label(status: str, response_language: str = "pt-PT") -> str:
    keys = {
        "pending": "task.status.pending",
        "completed": "task.status.completed",
        "cancelled": "task.status.cancelled",
    }
    key = keys.get(status)
    return localize(key, response_language) if key else status
