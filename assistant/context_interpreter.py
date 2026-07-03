from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.context_observer import ContextSnapshot, WindowInfo


NO_INTERPRETABLE_CONTEXT = (
    "Consigo tentar observar o computador, mas ainda não tenho dados suficientes. "
    "Experimenta mudar de janela ou aguardar alguns segundos."
)

NOISE_PROCESSES = {
    "program manager",
    "textinputhost.exe",
    "applicationframehost.exe",
    "nvidia overlay.exe",
    "shellexperiencehost.exe",
    "searchhost.exe",
}

APP_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Desenvolvimento": (
        "code.exe",
        "code - insiders.exe",
        "codex",
        "codex.exe",
        "git",
        "git.exe",
        "windowsterminal.exe",
        "terminal",
        "powershell.exe",
        "cmd.exe",
    ),
    "Comunicação": (
        "whatsapp.exe",
        "teams.exe",
        "outlook.exe",
        "discord.exe",
    ),
    "Produtividade": (
        "winword.exe",
        "excel.exe",
        "powerpnt.exe",
        "onenote.exe",
        "notepad.exe",
    ),
    "Navegação": (
        "chrome.exe",
        "msedge.exe",
        "firefox.exe",
    ),
    "Sistema": (
        "systemsettings.exe",
        "explorer.exe",
        "nvidia overlay.exe",
    ),
}


def interpret_snapshot(snapshot: ContextSnapshot | None, debug_context: bool = False) -> str:
    if snapshot is None:
        return NO_INTERPRETABLE_CONTEXT

    useful_windows = [_window for _window in snapshot.open_windows if _is_useful_window(_window)]
    project = _project_name(snapshot)
    categories = _group_windows(useful_windows)
    modified_files = _modified_files(snapshot)

    parts: list[str] = []
    if project:
        parts.append(f"Estás a trabalhar no projeto {project}.")

    development = categories.get("Desenvolvimento", [])
    navigation = categories.get("Navegação", [])
    communication = categories.get("Comunicação", [])
    productivity = categories.get("Produtividade", [])

    details: list[str] = []
    vscode = _vscode_detail(snapshot)
    if vscode:
        details.append(vscode)
    if _has_app(development, "Codex"):
        details.append("tens o Codex aberto")
    browser_detail = _browser_detail(navigation)
    if browser_detail:
        details.append(browser_detail)
    if communication:
        details.append("tens ferramentas de comunicação abertas: " + _join(_unique_app_names(communication)))
    if productivity:
        details.append("tens ferramentas de produtividade abertas: " + _join(_unique_app_names(productivity)))

    if details:
        parts.append(_sentence_from_details(details))

    if modified_files:
        parts.append("Os ficheiros modificados recentemente foram " + _join(modified_files[:5]) + ".")

    if not parts:
        active = snapshot.active_window or snapshot.active_app
        if active and active.strip().lower() not in NOISE_PROCESSES:
            parts.append(f"Tenho apenas um sinal parcial: a janela/aplicação ativa parece ser {active}.")
        else:
            return NO_INTERPRETABLE_CONTEXT

    if debug_context:
        parts.append("")
        parts.append("[DEBUG_CONTEXT]")
        parts.append(_raw_snapshot_debug(snapshot, useful_windows, categories))

    return " ".join(part for part in parts if part).strip()


def _project_name(snapshot: ContextSnapshot) -> str:
    if snapshot.current_project:
        return snapshot.current_project
    for session in snapshot.vscode_sessions:
        if session.folder:
            return session.folder
    for repo in snapshot.git_repositories:
        if repo.path:
            return repo.path
    return ""


def _is_useful_window(window: WindowInfo) -> bool:
    title = window.title.strip()
    process = window.process_name.strip().lower()
    if not title:
        return False
    if title.lower() in NOISE_PROCESSES or process in NOISE_PROCESSES:
        return False
    return True


def _group_windows(windows: list[WindowInfo]) -> dict[str, list[WindowInfo]]:
    grouped: dict[str, list[WindowInfo]] = defaultdict(list)
    for window in windows:
        grouped[_category_for(window)].append(window)
    return dict(grouped)


def _category_for(window: WindowInfo) -> str:
    process = window.process_name.lower()
    title = window.title.lower()
    for category, markers in APP_CATEGORIES.items():
        if any(marker in process or marker in title for marker in markers):
            return category
    return "Outras"


def _vscode_detail(snapshot: ContextSnapshot) -> str:
    if snapshot.vscode_sessions:
        folders = [session.folder for session in snapshot.vscode_sessions if session.folder]
        if folders:
            return f"tens o VS Code aberto no repositório {_join_unique(folders)}"
        return "tens o VS Code aberto"
    if "code" in snapshot.active_app.lower() or "visual studio code" in snapshot.active_window.lower():
        return "tens o VS Code aberto"
    return ""


def _browser_detail(windows: list[WindowInfo]) -> str:
    if not windows:
        return ""
    titles = [window.title for window in windows if window.title]
    project_titles = [
        title for title in titles if any(word in title.lower() for word in ("assistenteia", "projeto", "projecto", "codex"))
    ]
    if project_titles:
        return "tens o browser aberto com contexto relacionado com o projeto"
    return "tens o browser aberto"


def _modified_files(snapshot: ContextSnapshot) -> list[str]:
    files = list(snapshot.recently_modified_files)
    for repo in snapshot.git_repositories:
        files.extend(repo.modified_files)
    cleaned: list[str] = []
    for file_name in files:
        name = Path(file_name).as_posix()
        if name not in cleaned:
            cleaned.append(name)
    return cleaned


def _has_app(windows: list[WindowInfo], label: str) -> bool:
    target = label.lower()
    return any(target in window.title.lower() or target in window.process_name.lower() for window in windows)


def _unique_app_names(windows: list[WindowInfo]) -> list[str]:
    names: list[str] = []
    for window in windows:
        name = _friendly_app_name(window)
        if name not in names:
            names.append(name)
    return names


def _friendly_app_name(window: WindowInfo) -> str:
    process = window.process_name.lower()
    mapping = {
        "whatsapp.exe": "WhatsApp",
        "teams.exe": "Teams",
        "outlook.exe": "Outlook",
        "discord.exe": "Discord",
        "winword.exe": "Word",
        "excel.exe": "Excel",
        "powerpnt.exe": "PowerPoint",
        "onenote.exe": "OneNote",
        "notepad.exe": "Bloco de Notas",
        "chrome.exe": "Chrome",
        "msedge.exe": "Edge",
        "firefox.exe": "Firefox",
    }
    return mapping.get(process, window.process_name or window.title)


def _sentence_from_details(details: list[str]) -> str:
    sentence = _join(details)
    return sentence[0].upper() + sentence[1:] + "."


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " e " + items[-1]


def _join_unique(items: list[str]) -> str:
    unique: list[str] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return _join(unique)


def _raw_snapshot_debug(
    snapshot: ContextSnapshot,
    useful_windows: list[WindowInfo],
    categories: dict[str, list[WindowInfo]],
) -> str:
    lines = [
        f"Aplicação ativa: {snapshot.active_app}",
        f"Janela ativa: {snapshot.active_window}",
        f"Projeto atual: {snapshot.current_project}",
        f"Janelas úteis: {len(useful_windows)}",
        "Categorias: "
        + ", ".join(f"{category}={len(windows)}" for category, windows in sorted(categories.items())),
        "Ficheiros modificados: " + ", ".join(_modified_files(snapshot)[:10]),
    ]
    return "\n".join(lines)
