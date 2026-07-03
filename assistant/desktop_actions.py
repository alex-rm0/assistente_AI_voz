from __future__ import annotations

import os
import subprocess
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Protocol
from urllib.parse import urlparse


ALLOWED_APPLICATIONS = {
    "chrome": {
        "label": "Chrome",
        "aliases": {"chrome", "browser", "navegador"},
        "executables": ("chrome.exe", "chrome"),
    },
    "vscode": {
        "label": "VS Code",
        "aliases": {"vs code", "vscode", "visual studio code", "codigo", "código", "code"},
        "executables": ("code.cmd", "code.exe", "code"),
    },
    "outlook": {
        "label": "Outlook",
        "aliases": {"outlook", "mail", "email", "correio"},
        "executables": ("outlook.exe", "olk.exe"),
        "uri": "mailto:",
    },
    "teams": {
        "label": "Teams",
        "aliases": {"teams", "microsoft teams"},
        "executables": ("teams.exe", "ms-teams.exe"),
        "uri": "msteams:",
    },
    "discord": {
        "label": "Discord",
        "aliases": {"discord"},
        "executables": ("discord.exe",),
        "uri": "discord:",
    },
    "notepad": {
        "label": "Bloco de Notas",
        "aliases": {"notepad", "bloco de notas", "notas"},
        "executables": ("notepad.exe",),
    },
    "explorer": {
        "label": "Explorador de Ficheiros",
        "aliases": {"explorador", "explorador de ficheiros", "file explorer", "ficheiros"},
        "executables": ("explorer.exe",),
    },
}


@dataclass(frozen=True)
class DesktopActionResult:
    ok: bool
    message: str


class DesktopActionRunner(Protocol):
    def open_application(self, app_key: str, executable: str | None, uri: str | None = None) -> DesktopActionResult:
        ...

    def open_path(self, path: Path) -> DesktopActionResult:
        ...

    def open_url(self, url: str) -> DesktopActionResult:
        ...

    def open_project(self, editor_executable: str | None, project_path: Path) -> DesktopActionResult:
        ...


class WindowsDesktopActionRunner:
    """Runs whitelisted desktop actions without shell=True or arbitrary commands."""

    def open_application(self, app_key: str, executable: str | None, uri: str | None = None) -> DesktopActionResult:
        label = app_label(app_key)
        try:
            if executable:
                subprocess.Popen([executable], close_fds=True)
            elif uri:
                os.startfile(uri)  # type: ignore[attr-defined]
            else:
                return DesktopActionResult(False, f"Nao encontrei o {label} neste computador.")
        except Exception as exc:
            return DesktopActionResult(False, f"Nao consegui abrir {label}: {exc}")
        return DesktopActionResult(True, f"Abri {label}.")

    def open_path(self, path: Path) -> DesktopActionResult:
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            return DesktopActionResult(False, f"Nao consegui abrir '{path.name}': {exc}")
        return DesktopActionResult(True, f"Abri '{path.name}'.")

    def open_url(self, url: str) -> DesktopActionResult:
        try:
            opened = webbrowser.open(url)
        except Exception as exc:
            return DesktopActionResult(False, f"Nao consegui abrir o URL: {exc}")
        if not opened:
            return DesktopActionResult(False, "Nao consegui abrir o URL no browser.")
        return DesktopActionResult(True, f"Abri o URL: {url}")

    def open_project(self, editor_executable: str | None, project_path: Path) -> DesktopActionResult:
        try:
            if editor_executable:
                subprocess.Popen([editor_executable, str(project_path)], close_fds=True)
            else:
                os.startfile(str(project_path))  # type: ignore[attr-defined]
        except Exception as exc:
            return DesktopActionResult(False, f"Nao consegui abrir o projeto '{project_path.name}': {exc}")
        return DesktopActionResult(True, f"Abri o projeto {project_path.name}.")


def normalize_app_name(value: str) -> str:
    text = _normalize(value)
    for key, info in ALLOWED_APPLICATIONS.items():
        aliases = {_normalize(alias) for alias in info["aliases"]}
        aliases.add(key)
        if text in aliases:
            return key
    return ""


def app_label(app_key: str) -> str:
    info = ALLOWED_APPLICATIONS.get(app_key)
    return str(info.get("label")) if info else app_key


def is_application_open(app_key: str, context_observer=None) -> bool:
    if context_observer is None:
        return False
    try:
        snapshot = context_observer.observe_once()
    except Exception:
        try:
            snapshot = context_observer.latest_snapshot()
        except Exception:
            snapshot = None
    if snapshot is None:
        return False
    markers = _app_markers(app_key)
    for window in getattr(snapshot, "open_windows", ()) or ():
        haystack = _normalize(" ".join((getattr(window, "title", ""), getattr(window, "process_name", ""))))
        if any(marker in haystack for marker in markers):
            return True
    active = _normalize(" ".join((getattr(snapshot, "active_app", ""), getattr(snapshot, "active_window", ""))))
    return any(marker in active for marker in markers)


def resolve_application(app_name: str) -> tuple[str, str | None, str | None]:
    key = normalize_app_name(app_name)
    if not key:
        return "", None, None
    info = ALLOWED_APPLICATIONS[key]
    executable = _first_executable(info.get("executables", ()))
    uri = str(info.get("uri", "")) or None
    return key, executable, uri


def resolve_safe_path(raw_path: str, allowed_roots: list[Path]) -> Path | None:
    value = raw_path.strip().strip("\"'")
    if not value:
        return None
    candidate = Path(value)
    paths = [candidate] if candidate.is_absolute() else [root / value for root in allowed_roots]
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not _is_inside_allowed_roots(resolved, allowed_roots):
            continue
        if resolved.exists():
            return resolved
    return None


def validate_url(url: str) -> str:
    value = url.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return value


def known_project_path(project_name: str, known_projects: dict[str, str], project_root: Path) -> Path | None:
    normalized = _normalize(project_name)
    candidates = dict(known_projects)
    candidates.setdefault("assistenteia", str(project_root))
    for name, raw_path in candidates.items():
        if _normalize(name) != normalized:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        try:
            resolved = path.resolve()
        except OSError:
            return None
        if resolved.exists() and resolved.is_dir():
            return resolved
    return None


def remember_desktop_action(long_term_memory, action: str, target: str) -> None:
    if long_term_memory is None:
        return
    remember = getattr(long_term_memory, "remember", None)
    if remember is None:
        return
    try:
        remember(f"O utilizador usa frequentemente {target} para {action}.", category="preferencias")
    except Exception:
        return


def _first_executable(names) -> str | None:
    for name in names:
        found = which(str(name))
        if found:
            return found
    return None


def _is_inside_allowed_roots(path: Path, allowed_roots: list[Path]) -> bool:
    for root in allowed_roots:
        resolved_root = root.resolve()
        if path == resolved_root or resolved_root in path.parents:
            return True
    return False


def _app_markers(app_key: str) -> set[str]:
    info = ALLOWED_APPLICATIONS.get(app_key, {})
    markers = {_normalize(app_key), _normalize(str(info.get("label", "")))}
    markers.update(_normalize(executable) for executable in info.get("executables", ()))
    markers.update(_normalize(alias) for alias in info.get("aliases", ()))
    return {marker for marker in markers if marker}


def _normalize(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char)).strip()
