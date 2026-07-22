from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SessionLike(Protocol):
    main_project: str
    main_activity: str
    summary: str
    files_touched: str
    tasks_changed: str
    tools_used: str
    decisions_taken: str
    next_suggested_step: str


GENERIC_ACTIVITIES = {
    "",
    "atividade de trabalho local",
    "sem atividade principal",
    "atividade principal ainda pouco clara",
}

GENERIC_NEXT_STEPS = {
    "",
    "rever o contexto anterior e escolher o proximo passo",
    "rever o contexto anterior e escolher o próximo passo",
}


@dataclass(frozen=True)
class SessionReflection:
    what_happened: str = ""
    decisions: str = ""
    pending: str = ""
    next_step: str = ""

    @property
    def has_useful_context(self) -> bool:
        return any((self.what_happened, self.decisions, self.pending, self.next_step))

    @property
    def has_clear_next_step(self) -> bool:
        return bool(self.next_step)


def reflect_session(summary: SessionLike) -> SessionReflection:
    project = (summary.main_project or "").strip()
    activity = _clean_activity(summary.main_activity)
    decisions = _clean_human_text(summary.decisions_taken)
    pending = _clean_human_text(summary.tasks_changed)
    next_step = _clean_next_step(summary.next_suggested_step)

    what_happened = ""
    if project and activity:
        what_happened = f"estivemos a trabalhar no {project}, sobretudo em {activity}"
    elif project:
        what_happened = f"estivemos a trabalhar no {project}"
    elif activity:
        what_happened = f"estivemos sobretudo em {activity}"
    else:
        what_happened = _infer_from_summary(summary.summary)

    return SessionReflection(
        what_happened=what_happened,
        decisions=decisions,
        pending=pending,
        next_step=next_step,
    )


def format_session_reflection(summary: SessionLike) -> str:
    reflection = reflect_session(summary)
    if not reflection.has_useful_context:
        return "Ainda não tenho contexto suficiente para reconstruir a última sessão com utilidade."

    lines: list[str] = []
    if reflection.what_happened:
        lines.append(f"Da última vez, {reflection.what_happened}.")
    if reflection.decisions:
        lines.append(f"Ficou registado que {reflection.decisions}.")
    if reflection.pending:
        lines.append(f"Ficou pendente: {reflection.pending}.")
    if reflection.next_step:
        lines.append(f"O próximo passo parecia ser {reflection.next_step}.")
    return " ".join(lines)


def format_today_reflection(summaries: list[SessionLike]) -> str:
    reflections = [reflect_session(summary) for summary in summaries]
    useful = [reflection for reflection in reflections if reflection.has_useful_context]
    if not useful:
        return "Ainda não tenho sessões úteis guardadas para hoje."

    lines = ["Hoje, pelo que tenho guardado:"]
    for reflection in useful:
        if reflection.what_happened:
            lines.append(f"- {reflection.what_happened}.")
        elif reflection.next_step:
            lines.append(f"- ficou como próximo passo {reflection.next_step}.")
    return "\n".join(lines)


def format_next_step(summary: SessionLike) -> str:
    next_step = reflect_session(summary).next_step
    if not next_step:
        return "Ainda não tenho um próximo passo claro guardado."
    return f"O próximo passo parecia ser {next_step}."


def format_startup_session_hint(summary: SessionLike, user_name: str = "Alexandre") -> str:
    reflection = reflect_session(summary)
    if not reflection.has_clear_next_step:
        return ""
    if reflection.what_happened:
        return f"Olá. Ficámos em {reflection.what_happened}. Queres pegar por {reflection.next_step}?"
    return f"Olá. Tínhamos este próximo passo em aberto: {reflection.next_step}. Queres pegar por aí?"


def _clean_activity(value: str) -> str:
    text = _clean_human_text(value)
    if text.lower() in GENERIC_ACTIVITIES:
        return ""
    return text


def _clean_next_step(value: str) -> str:
    text = _clean_human_text(value)
    if text.lower() in GENERIC_NEXT_STEPS:
        return ""
    return text


def _clean_human_text(value: str) -> str:
    text = " ".join(str(value or "").replace("Utilizador:", "").replace("Assistente:", "").split())
    text = text.strip(" .;:")
    return text


def _infer_from_summary(value: str) -> str:
    text = _clean_human_text(value)
    if not text:
        return ""
    technical_markers = (
        "sessao iniciada",
        "sessao terminada",
        "motivo de fecho",
        "session_summaries",
    )
    chunks = [chunk.strip() for chunk in text.split(".") if chunk.strip()]
    human_chunks = [
        chunk
        for chunk in chunks
        if not any(marker in chunk.lower() for marker in technical_markers)
    ]
    if not human_chunks:
        return ""
    return human_chunks[-1].strip(" .")
