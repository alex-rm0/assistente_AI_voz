from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.context_observer import ContextSnapshot


@dataclass(frozen=True)
class ContextReasoningResult:
    main_activity: str
    main_project: str
    relevant_applications: tuple[str, ...]
    possible_goals: tuple[str, ...]
    suggestions: tuple[str, ...]
    evidence: tuple[str, ...]
    current_observations: tuple[str, ...] = ()
    recent_activity: tuple[str, ...] = ()
    recently_modified_files: tuple[str, ...] = ()
    inferences: tuple[str, ...] = ()

    def format(self) -> str:
        lines: list[str] = []

        if self.current_observations:
            lines.append("Contexto atual observado agora:")
            lines.extend(f"- {item}" for item in self.current_observations)

        if self.recent_activity:
            if lines:
                lines.append("")
            lines.append("Atividade recente:")
            lines.extend(f"- {item}" for item in self.recent_activity)

        if self.recently_modified_files:
            if lines:
                lines.append("")
            lines.append("Ficheiros recentemente modificados:")
            lines.extend(f"- {filename}" for filename in self.recently_modified_files)

        if self.inferences:
            if lines:
                lines.append("")
            lines.append("Inferências prováveis:")
            lines.extend(f"- {item}" for item in self.inferences)

        if self.possible_goals:
            if lines:
                lines.append("")
            lines.append("Possíveis objetivos:")
            lines.extend(f"- {goal}" for goal in self.possible_goals)

        if self.suggestions:
            if lines:
                lines.append("")
            lines.append("Sugestões opcionais:")
            lines.extend(f"- {suggestion}" for suggestion in self.suggestions)

        if self.evidence:
            if lines:
                lines.append("")
            lines.append("Evidências observadas:")
            lines.extend(f"- {item}" for item in self.evidence)

        if not lines:
            return "Ainda não tenho dados suficientes para concluir a tua atividade principal."
        return "\n".join(lines)


def reason_about_context(
    snapshot: ContextSnapshot | None,
    active_contexts: list[str] | tuple[str, ...] | None = None,
    relevant_memory: str = "",
    pending_tasks: str = "",
) -> ContextReasoningResult:
    if snapshot is None:
        return ContextReasoningResult("", "", (), (), (), ())

    project = _main_project(snapshot)
    applications = _relevant_applications(snapshot)
    modified_files = _modified_files(snapshot)
    current_observations = _current_observations(snapshot, project, applications)
    recent_activity = _recent_activity(relevant_memory, pending_tasks)
    evidence = _evidence(snapshot, project, modified_files)
    main_activity = _main_activity(snapshot, project, applications, modified_files)
    inferences = _inferences(main_activity, project, applications, active_contexts)
    possible_goals = _possible_goals(project, modified_files, relevant_memory, pending_tasks)
    suggestions = _suggestions(snapshot, project, possible_goals)

    return ContextReasoningResult(
        main_activity=main_activity,
        main_project=project,
        relevant_applications=tuple(applications),
        possible_goals=tuple(possible_goals),
        suggestions=tuple(suggestions),
        evidence=tuple(evidence),
        current_observations=tuple(current_observations),
        recent_activity=tuple(recent_activity),
        recently_modified_files=tuple(modified_files[:8]),
        inferences=tuple(inferences),
    )


def _current_observations(snapshot: ContextSnapshot, project: str, applications: list[str]) -> list[str]:
    observations: list[str] = []
    if snapshot.active_window:
        observations.append(f"Janela ativa: {snapshot.active_window}")
    if snapshot.active_app:
        observations.append(f"Aplicação ativa: {snapshot.active_app}")
    if project:
        observations.append(f"Projeto observado: {project}")
    if applications:
        observations.append("Aplicações relevantes observadas: " + ", ".join(applications[:6]))
    if snapshot.open_windows:
        observations.append(f"Janelas abertas observadas: {len(snapshot.open_windows)}")
    return observations


def _recent_activity(relevant_memory: str, pending_tasks: str) -> list[str]:
    activity: list[str] = []
    memory = relevant_memory.strip()
    tasks = pending_tasks.strip()
    if memory:
        activity.append("Memória relevante encontrada, mas não é estado atual observado.")
    if tasks and "nao tens tarefas" not in _normalize(tasks) and "não tens tarefas" not in _normalize(tasks):
        activity.append("Existem tarefas pendentes relacionadas que podem influenciar o contexto.")
    return activity


def _evidence(snapshot: ContextSnapshot, project: str, modified_files: list[str]) -> list[str]:
    evidence: list[str] = []
    if snapshot.active_window:
        evidence.append(f"Janela ativa observada: {snapshot.active_window}")
    if snapshot.active_app:
        evidence.append(f"Aplicação ativa observada: {snapshot.active_app}")
    if project:
        evidence.append(f"Projeto identificado por observação: {project}")
    if snapshot.vscode_sessions:
        evidence.append("Sessão de VS Code observada" + (f" em {project}" if project else ""))
    if snapshot.git_repositories:
        evidence.append("Repositório Git observado")
    if modified_files:
        evidence.append("Ficheiros modificados recentemente: " + ", ".join(modified_files[:5]))
    return evidence


def _main_project(snapshot: ContextSnapshot) -> str:
    if snapshot.current_project:
        return snapshot.current_project
    for session in snapshot.vscode_sessions:
        if session.folder:
            return session.folder
    for repo in snapshot.git_repositories:
        if repo.path:
            return repo.path
    return ""


def _relevant_applications(snapshot: ContextSnapshot) -> list[str]:
    names: list[str] = []
    for source in (snapshot.active_app,):
        name = _friendly_app(source)
        if name and name not in names:
            names.append(name)
    for window in snapshot.open_windows:
        name = _friendly_app(window.process_name or window.title)
        if name and name not in names and _is_relevant_app(name):
            names.append(name)
    if snapshot.vscode_sessions and "VS Code" not in names:
        names.append("VS Code")
    if snapshot.git_repositories and "Git" not in names:
        names.append("Git")
    return names


def _main_activity(
    snapshot: ContextSnapshot,
    project: str,
    applications: list[str],
    modified_files: list[str],
) -> str:
    has_development = any(app in applications for app in ("VS Code", "Codex", "Git", "Terminal"))
    has_modified_files = bool(modified_files)
    if project and (has_development or has_modified_files):
        return f"Parece que estás a trabalhar no desenvolvimento do {project}."
    if project:
        return f"Parece que estás a trabalhar no projeto {project}."
    if applications:
        return "Parece que estás a usar " + ", ".join(applications[:3]) + "."
    if snapshot.active_window or snapshot.active_app:
        return "Parece que tens uma atividade ativa, mas ainda não consigo identificar o objetivo."
    return ""


def _inferences(
    main_activity: str,
    project: str,
    applications: list[str],
    active_contexts: list[str] | tuple[str, ...] | None,
) -> list[str]:
    inferences: list[str] = []
    if main_activity:
        inferences.append(main_activity)
    contexts = set(active_contexts or ())
    if project and "TECH_CONTEXT" in contexts:
        inferences.append(f"Provavelmente o pedido está ligado a trabalho técnico no projeto {project}.")
    if "PRODUCTIVITY_CONTEXT" in contexts:
        inferences.append("Provavelmente o contexto também envolve organização ou tarefas.")
    if applications and not main_activity:
        inferences.append("Parece que as aplicações abertas são relevantes para a tua atividade atual.")
    return _unique(inferences)


def _possible_goals(
    project: str,
    modified_files: list[str],
    relevant_memory: str,
    pending_tasks: str,
) -> list[str]:
    goals: list[str] = []
    if project and modified_files:
        goals.append(f"Continuar alterações recentes no projeto {project}.")
    if any("test" in file.lower() for file in modified_files):
        goals.append("Validar ou melhorar testes.")
    if any(file.lower().endswith((".md", ".txt")) for file in modified_files):
        goals.append("Atualizar documentação ou notas do projeto.")
    if "tarefa" in _normalize(pending_tasks) and project:
        goals.append(f"Avançar tarefas pendentes relacionadas com {project}.")
    if relevant_memory and project and project.lower() in relevant_memory.lower():
        goals.append(f"Retomar contexto recorrente sobre {project}.")
    return _unique(goals)


def _suggestions(snapshot: ContextSnapshot, project: str, possible_goals: list[str]) -> list[str]:
    suggestions: list[str] = []
    if project and possible_goals:
        suggestions.append("Posso resumir os ficheiros modificados recentemente.")
    if snapshot.git_repositories:
        suggestions.append("Posso ajudar a rever o estado do repositório ou preparar próximos passos.")
    return suggestions[:2]


def _modified_files(snapshot: ContextSnapshot) -> list[str]:
    files = list(snapshot.recently_modified_files)
    for repo in snapshot.git_repositories:
        files.extend(repo.modified_files)
    cleaned: list[str] = []
    for filename in files:
        value = Path(filename).as_posix()
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def _friendly_app(value: str) -> str:
    text = value.lower()
    if not text:
        return ""
    mapping = {
        "code.exe": "VS Code",
        "codex": "Codex",
        "codex.exe": "Codex",
        "git.exe": "Git",
        "windowsterminal.exe": "Terminal",
        "powershell.exe": "Terminal",
        "cmd.exe": "Terminal",
        "chrome.exe": "Chrome",
        "msedge.exe": "Edge",
        "firefox.exe": "Firefox",
    }
    for marker, label in mapping.items():
        if marker in text:
            return label
    return value


def _is_relevant_app(name: str) -> bool:
    return name in {"VS Code", "Codex", "Git", "Terminal", "Chrome", "Edge", "Firefox"}


def _unique(items: list[str]) -> list[str]:
    unique: list[str] = []
    for item in items:
        if item and item not in unique:
            unique.append(item)
    return unique


def _normalize(text: str) -> str:
    return (
        text.lower()
        .replace("ã", "a")
        .replace("á", "a")
        .replace("à", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("õ", "o")
        .replace("ô", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )
