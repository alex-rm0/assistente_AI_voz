from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_CONTEXT_DB = "context_observer.sqlite"
DEFAULT_SUMMARY_MIN_SECONDS = 60.0
NOISE_WINDOW_TITLES = {
    "program manager",
    "textinputhost",
    "applicationframehost",
    "nvidia overlay",
}
NOISE_WINDOW_PROCESSES = {
    "textinputhost.exe",
    "applicationframehost.exe",
    "nvidia overlay.exe",
}


@dataclass(frozen=True)
class WindowInfo:
    title: str
    process_name: str = ""
    process_id: int = 0
    is_active: bool = False


@dataclass(frozen=True)
class ProcessInfo:
    process_id: int
    name: str
    executable: str = ""


@dataclass(frozen=True)
class VscodeSession:
    window_title: str
    folder: str = ""
    process_id: int = 0


@dataclass(frozen=True)
class GitRepoContext:
    path: str
    branch: str = ""
    modified_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextSummary:
    summary: str
    project: str
    start_at: float
    end_at: float
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextSnapshot:
    active_app: str
    active_window: str
    recent_files: tuple[str, ...] = ()
    current_project: str = ""
    observed_at: float = 0.0
    open_windows: tuple[WindowInfo, ...] = ()
    active_processes: tuple[ProcessInfo, ...] = ()
    vscode_sessions: tuple[VscodeSession, ...] = ()
    git_repositories: tuple[GitRepoContext, ...] = ()
    recently_modified_files: tuple[str, ...] = ()


SnapshotProvider = Callable[[], ContextSnapshot]
SummaryCallback = Callable[[ContextSummary], None]


@dataclass(frozen=True)
class _ActivitySession:
    start_at: float
    last_seen_at: float
    active_app: str
    active_window: str
    project: str
    tools: tuple[str, ...]
    modified_files: tuple[str, ...]


class ContextObserver:
    """Passive observer that records local context without executing actions."""

    def __init__(
        self,
        data_path: Path,
        project_root: Path,
        db_file: str = DEFAULT_CONTEXT_DB,
        recent_files_limit: int = 10,
        snapshot_provider: SnapshotProvider | None = None,
        summary_callback: SummaryCallback | None = None,
        summary_min_seconds: float = DEFAULT_SUMMARY_MIN_SECONDS,
        debug_context: bool = False,
    ) -> None:
        self.data_path = data_path.resolve()
        self.db_path = (self.data_path / db_file).resolve()
        self.project_root = project_root.resolve()
        self.recent_files_limit = recent_files_limit
        self.snapshot_provider = snapshot_provider or self._default_snapshot
        self.summary_callback = summary_callback
        self.summary_min_seconds = max(1.0, summary_min_seconds)
        self.debug_context = debug_context
        self._last_snapshot: ContextSnapshot | None = None
        self._last_observed_at: float | None = None
        self._current_session: _ActivitySession | None = None

        self._ensure_inside_data()
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def set_summary_callback(self, callback: SummaryCallback | None) -> None:
        self.summary_callback = callback

    def observe_once(self) -> ContextSnapshot:
        self._debug_log("A iniciar observacao de contexto.")
        snapshot = self.snapshot_provider()
        observed_at = snapshot.observed_at or time.time()
        open_windows = snapshot.open_windows or tuple(_open_windows())
        active_processes = snapshot.active_processes or tuple(_active_processes())
        vscode_sessions = snapshot.vscode_sessions or tuple(_vscode_sessions(open_windows))
        git_repositories = snapshot.git_repositories or tuple(_git_repositories(self.project_root))
        recently_modified_files = snapshot.recently_modified_files or tuple(
            _recently_modified_files(self.project_root, self.recent_files_limit)
        )
        snapshot = ContextSnapshot(
            active_app=snapshot.active_app,
            active_window=snapshot.active_window,
            recent_files=snapshot.recent_files[: self.recent_files_limit],
            current_project=snapshot.current_project or self._infer_project(snapshot),
            observed_at=observed_at,
            open_windows=open_windows,
            active_processes=active_processes,
            vscode_sessions=vscode_sessions,
            git_repositories=git_repositories,
            recently_modified_files=recently_modified_files,
        )
        self._store_snapshot(snapshot)
        self._update_activity_time(snapshot)
        self._update_summary_session(snapshot)
        self._last_snapshot = snapshot
        self._last_observed_at = observed_at
        self._debug_log(
            "Snapshot atualizado: "
            f"active_app={snapshot.active_app!r}, active_window={snapshot.active_window!r}, "
            f"open_windows={len(snapshot.open_windows)}, processes={len(snapshot.active_processes)}"
        )
        return snapshot

    def flush_summary(self) -> ContextSummary | None:
        if self._current_session is None:
            return None
        summary = self._finalize_session(self._current_session)
        self._current_session = None
        return summary

    def latest_snapshot(self) -> ContextSnapshot | None:
        if self._last_snapshot is not None:
            self._debug_log("A usar snapshot vivo em memoria.")
            return self._last_snapshot

        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT active_app, active_window, recent_files, current_project, observed_at
                    , open_windows, active_processes, vscode_sessions, git_repositories, recently_modified_files
                FROM context_snapshots
                ORDER BY observed_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            self._debug_log("Nao existe snapshot guardado em SQLite.")
            return None
        self._debug_log("A usar snapshot persistido em SQLite.")
        return ContextSnapshot(
            active_app=row[0],
            active_window=row[1],
            recent_files=tuple(item for item in row[2].split("\n") if item),
            current_project=row[3],
            observed_at=float(row[4]),
            open_windows=_decode_windows(row[5]),
            active_processes=_decode_processes(row[6]),
            vscode_sessions=_decode_vscode_sessions(row[7]),
            git_repositories=_decode_git_repositories(row[8]),
            recently_modified_files=tuple(_loads_json_list(row[9])),
        )

    def latest_summary(self) -> ContextSummary | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT summary, project, start_at, end_at, tools
                FROM context_summaries
                ORDER BY end_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return ContextSummary(
            summary=row[0],
            project=row[1],
            start_at=float(row[2]),
            end_at=float(row[3]),
            tools=tuple(item for item in row[4].split("\n") if item),
        )

    def activity_summary(self, limit: int = 10) -> list[tuple[str, str, str, float]]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT active_app, active_window, current_project, total_seconds
                FROM activity_time
                ORDER BY total_seconds DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(row[0], row[1], row[2], float(row[3])) for row in rows]

    def _store_snapshot(self, snapshot: ContextSnapshot) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO context_snapshots (
                    observed_at, active_app, active_window, recent_files, current_project,
                    open_windows, active_processes, vscode_sessions, git_repositories, recently_modified_files
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.observed_at,
                    snapshot.active_app,
                    snapshot.active_window,
                    "\n".join(snapshot.recent_files),
                    snapshot.current_project,
                    _encode_windows(snapshot.open_windows),
                    _encode_processes(snapshot.active_processes),
                    _encode_vscode_sessions(snapshot.vscode_sessions),
                    _encode_git_repositories(snapshot.git_repositories),
                    json.dumps(list(snapshot.recently_modified_files), ensure_ascii=True),
                ),
            )
            connection.execute(
                """
                DELETE FROM context_snapshots
                WHERE id NOT IN (
                    SELECT id FROM context_snapshots
                    ORDER BY observed_at DESC
                    LIMIT 50
                )
                """
            )

    def _update_activity_time(self, snapshot: ContextSnapshot) -> None:
        if self._last_snapshot is None or self._last_observed_at is None:
            return

        elapsed = max(0.0, min(snapshot.observed_at - self._last_observed_at, 3600.0))
        if elapsed <= 0:
            return

        previous = self._last_snapshot
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO activity_time (
                    active_app, active_window, current_project, total_seconds, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(active_app, active_window, current_project)
                DO UPDATE SET
                    total_seconds = total_seconds + excluded.total_seconds,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    previous.active_app,
                    previous.active_window,
                    previous.current_project,
                    elapsed,
                    snapshot.observed_at,
                ),
            )

    def _default_snapshot(self) -> ContextSnapshot:
        active_app, active_window = _active_window_context()
        recent_files = tuple(_recent_file_names(self.recent_files_limit))
        open_windows = tuple(_open_windows())
        return ContextSnapshot(
            active_app=active_app,
            active_window=active_window,
            recent_files=recent_files,
            current_project="",
            observed_at=time.time(),
            open_windows=open_windows,
            active_processes=tuple(_active_processes()),
            vscode_sessions=tuple(_vscode_sessions(open_windows)),
            git_repositories=tuple(_git_repositories(self.project_root)),
            recently_modified_files=tuple(_recently_modified_files(self.project_root, self.recent_files_limit)),
        )

    def _infer_project(self, snapshot: ContextSnapshot) -> str:
        title = snapshot.active_window.lower()
        project_name = self.project_root.name
        if project_name.lower() in title:
            return project_name

        separators = (" - Visual Studio Code", " - Cursor", " - PyCharm")
        for separator in separators:
            if separator.lower() in title:
                return snapshot.active_window.split(separator, 1)[0].strip()

        return ""

    def _update_summary_session(self, snapshot: ContextSnapshot) -> None:
        tools = _tools_from_snapshot(snapshot)
        project = snapshot.current_project or _project_from_vscode(snapshot) or _project_from_git(snapshot)
        session_key = (snapshot.active_app, project, tools)

        if self._current_session is None:
            self._current_session = _ActivitySession(
                start_at=snapshot.observed_at,
                last_seen_at=snapshot.observed_at,
                active_app=snapshot.active_app,
                active_window=snapshot.active_window,
                project=project,
                tools=tools,
                modified_files=snapshot.recently_modified_files,
            )
            return

        current_key = (
            self._current_session.active_app,
            self._current_session.project,
            self._current_session.tools,
        )
        if session_key == current_key:
            self._current_session = _ActivitySession(
                start_at=self._current_session.start_at,
                last_seen_at=snapshot.observed_at,
                active_app=self._current_session.active_app,
                active_window=snapshot.active_window,
                project=self._current_session.project,
                tools=self._current_session.tools,
                modified_files=_merge_names(self._current_session.modified_files, snapshot.recently_modified_files),
            )
            return

        self._finalize_session(self._current_session)
        self._current_session = _ActivitySession(
            start_at=snapshot.observed_at,
            last_seen_at=snapshot.observed_at,
            active_app=snapshot.active_app,
            active_window=snapshot.active_window,
            project=project,
            tools=tools,
            modified_files=snapshot.recently_modified_files,
        )

    def _finalize_session(self, session: _ActivitySession) -> ContextSummary | None:
        duration = max(0.0, session.last_seen_at - session.start_at)
        if duration < self.summary_min_seconds:
            return None

        summary_text = _format_session_summary(session)
        summary = ContextSummary(
            summary=summary_text,
            project=session.project,
            start_at=session.start_at,
            end_at=session.last_seen_at,
            tools=session.tools,
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO context_summaries (summary, project, start_at, end_at, tools)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    summary.summary,
                    summary.project,
                    summary.start_at,
                    summary.end_at,
                    "\n".join(summary.tools),
                ),
            )
        if self.summary_callback is not None:
            self.summary_callback(summary)
        return summary

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS context_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at REAL NOT NULL,
                    active_app TEXT NOT NULL,
                    active_window TEXT NOT NULL,
                    recent_files TEXT NOT NULL,
                    current_project TEXT NOT NULL,
                    open_windows TEXT NOT NULL DEFAULT '[]',
                    active_processes TEXT NOT NULL DEFAULT '[]',
                    vscode_sessions TEXT NOT NULL DEFAULT '[]',
                    git_repositories TEXT NOT NULL DEFAULT '[]',
                    recently_modified_files TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            self._ensure_snapshot_columns(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_context_observed_at
                ON context_snapshots(observed_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_time (
                    active_app TEXT NOT NULL,
                    active_window TEXT NOT NULL,
                    current_project TEXT NOT NULL,
                    total_seconds REAL NOT NULL DEFAULT 0,
                    last_seen_at REAL NOT NULL,
                    PRIMARY KEY (active_app, active_window, current_project)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS context_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT '',
                    start_at REAL NOT NULL,
                    end_at REAL NOT NULL,
                    tools TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_context_summaries_end_at
                ON context_summaries(end_at)
                """
            )

    def _ensure_snapshot_columns(self, connection: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(context_snapshots)").fetchall()
        }
        additions = {
            "open_windows": "TEXT NOT NULL DEFAULT '[]'",
            "active_processes": "TEXT NOT NULL DEFAULT '[]'",
            "vscode_sessions": "TEXT NOT NULL DEFAULT '[]'",
            "git_repositories": "TEXT NOT NULL DEFAULT '[]'",
            "recently_modified_files": "TEXT NOT NULL DEFAULT '[]'",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE context_snapshots ADD COLUMN {name} {definition}")

    def _ensure_inside_data(self) -> None:
        if self.db_path != self.data_path and self.data_path not in self.db_path.parents:
            raise ValueError("Context observer database must stay inside the data folder.")

    def _debug_log(self, message: str) -> None:
        if self.debug_context:
            print(f"[ContextObserver DEBUG] {message}")


def _loads_json_list(value: str | None) -> list:
    if not value:
        return []
    try:
        data = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _encode_windows(windows: tuple[WindowInfo, ...]) -> str:
    return json.dumps(
        [
            {
                "title": window.title,
                "process_name": window.process_name,
                "process_id": window.process_id,
                "is_active": window.is_active,
            }
            for window in windows
        ],
        ensure_ascii=True,
    )


def _decode_windows(value: str | None) -> tuple[WindowInfo, ...]:
    windows = []
    for item in _loads_json_list(value):
        if not isinstance(item, dict):
            continue
        windows.append(
            WindowInfo(
                title=str(item.get("title", "")),
                process_name=str(item.get("process_name", "")),
                process_id=int(item.get("process_id", 0) or 0),
                is_active=bool(item.get("is_active", False)),
            )
        )
    return tuple(windows)


def _encode_processes(processes: tuple[ProcessInfo, ...]) -> str:
    return json.dumps(
        [
            {
                "process_id": process.process_id,
                "name": process.name,
                "executable": process.executable,
            }
            for process in processes
        ],
        ensure_ascii=True,
    )


def _decode_processes(value: str | None) -> tuple[ProcessInfo, ...]:
    processes = []
    for item in _loads_json_list(value):
        if not isinstance(item, dict):
            continue
        processes.append(
            ProcessInfo(
                process_id=int(item.get("process_id", 0) or 0),
                name=str(item.get("name", "")),
                executable=str(item.get("executable", "")),
            )
        )
    return tuple(processes)


def _encode_vscode_sessions(sessions: tuple[VscodeSession, ...]) -> str:
    return json.dumps(
        [
            {
                "window_title": session.window_title,
                "folder": session.folder,
                "process_id": session.process_id,
            }
            for session in sessions
        ],
        ensure_ascii=True,
    )


def _decode_vscode_sessions(value: str | None) -> tuple[VscodeSession, ...]:
    sessions = []
    for item in _loads_json_list(value):
        if not isinstance(item, dict):
            continue
        sessions.append(
            VscodeSession(
                window_title=str(item.get("window_title", "")),
                folder=str(item.get("folder", "")),
                process_id=int(item.get("process_id", 0) or 0),
            )
        )
    return tuple(sessions)


def _encode_git_repositories(repositories: tuple[GitRepoContext, ...]) -> str:
    return json.dumps(
        [
            {
                "path": repo.path,
                "branch": repo.branch,
                "modified_files": list(repo.modified_files),
            }
            for repo in repositories
        ],
        ensure_ascii=True,
    )


def _decode_git_repositories(value: str | None) -> tuple[GitRepoContext, ...]:
    repositories = []
    for item in _loads_json_list(value):
        if not isinstance(item, dict):
            continue
        modified_files = item.get("modified_files", [])
        if not isinstance(modified_files, list):
            modified_files = []
        repositories.append(
            GitRepoContext(
                path=str(item.get("path", "")),
                branch=str(item.get("branch", "")),
                modified_files=tuple(str(name) for name in modified_files),
            )
        )
    return tuple(repositories)


def _active_window_context() -> tuple[str, str]:
    if os.name != "nt":
        return "", ""

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]

    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return "", ""

    title_buffer = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, title_buffer, 512)
    window_title = title_buffer.value

    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    process_name = _process_name_from_pid(process_id.value, kernel32)
    return process_name, window_title


def _open_windows() -> list[WindowInfo]:
    if os.name != "nt":
        return []

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    enum_windows_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [enum_windows_proc, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    active_hwnd = user32.GetForegroundWindow()
    windows: list[WindowInfo] = []

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        title = title_buffer.value.strip()
        if not title:
            return True

        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        process_name = _process_name_from_pid(process_id.value, kernel32)
        if _is_noise_window(title, process_name):
            return True

        windows.append(
            WindowInfo(
                title=title,
                process_name=process_name,
                process_id=int(process_id.value),
                is_active=hwnd == active_hwnd,
            )
        )
        return True

    user32.EnumWindows(enum_windows_proc(callback), 0)
    return windows


def _is_noise_window(title: str, process_name: str) -> bool:
    normalized_title = title.strip().lower()
    normalized_process = process_name.strip().lower()
    if not normalized_title:
        return True
    if normalized_title in NOISE_WINDOW_TITLES:
        return True
    if normalized_process in NOISE_WINDOW_PROCESSES:
        return True
    return False


def debug_open_windows_report() -> str:
    active_app, active_title = _active_window_context()
    windows = _open_windows()
    lines = [
        f"Janela ativa: {active_title or '(desconhecida)'} [{active_app or 'processo desconhecido'}]",
        f"Numero de janelas abertas detetadas: {len(windows)}",
        "Titulos detetados:",
    ]
    lines.extend(
        f"- {window.title} [{window.process_name or 'processo desconhecido'}]"
        f"{' (ativa)' if window.is_active else ''}"
        for window in windows
    )
    return "\n".join(lines)


def _active_processes() -> list[ProcessInfo]:
    if os.name != "nt":
        return []

    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        return []

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    processes: list[ProcessInfo] = []
    try:
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            return []
        while True:
            pid = int(entry.th32ProcessID)
            name = str(entry.szExeFile)
            processes.append(ProcessInfo(process_id=pid, name=name, executable=name))
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)

    return processes


def _vscode_sessions(open_windows: tuple[WindowInfo, ...] | list[WindowInfo]) -> list[VscodeSession]:
    sessions: list[VscodeSession] = []
    for window in open_windows:
        process_name = window.process_name.lower()
        title = window.title
        if process_name not in {"code.exe", "code - insiders.exe"} and "visual studio code" not in title.lower():
            continue
        sessions.append(
            VscodeSession(
                window_title=title,
                folder=_folder_from_vscode_title(title),
                process_id=window.process_id,
            )
        )
    return sessions


def _folder_from_vscode_title(title: str) -> str:
    normalized = title.strip()
    separators = (" - Visual Studio Code", " - Code", " - Cursor")
    for separator in separators:
        if separator in normalized:
            left = normalized.split(separator, 1)[0].strip()
            parts = [part.strip() for part in left.split(" - ") if part.strip()]
            if parts:
                return parts[-1]
    return ""


def _git_repositories(root: Path) -> list[GitRepoContext]:
    repositories: list[GitRepoContext] = []
    if not root.exists():
        return repositories

    candidates: list[Path] = []
    if (root / ".git").exists():
        candidates.append(root)
    try:
        candidates.extend(path.parent for path in root.rglob(".git") if path.exists())
    except OSError:
        return repositories

    seen: set[Path] = set()
    for repo in candidates:
        resolved = repo.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        repositories.append(
            GitRepoContext(
                path=resolved.name,
                branch=_git_branch(resolved),
                modified_files=tuple(_git_modified_files(resolved)),
            )
        )
    return repositories


def _git_branch(repo: Path) -> str:
    head = repo / ".git" / "HEAD"
    try:
        content = head.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    prefix = "ref: refs/heads/"
    if content.startswith(prefix):
        return content[len(prefix) :]
    return content[:12]


def _git_modified_files(repo: Path, limit: int = 20) -> list[str]:
    files: list[str] = []
    try:
        for path in repo.rglob("*"):
            if len(files) >= limit:
                break
            if not path.is_file() or ".git" in path.parts:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if time.time() - stat.st_mtime <= 24 * 60 * 60:
                files.append(path.relative_to(repo).as_posix())
    except OSError:
        return []
    return files


def _recently_modified_files(root: Path, limit: int = 10) -> list[str]:
    if not root.exists():
        return []

    ignored = {".git", ".venv", "__pycache__", "data"}
    files: list[tuple[float, str]] = []
    try:
        for path in root.rglob("*"):
            if len(files) > limit * 10:
                break
            if not path.is_file():
                continue
            if any(part in ignored for part in path.parts):
                continue
            try:
                modified_at = path.stat().st_mtime
            except OSError:
                continue
            files.append((modified_at, path.relative_to(root).as_posix()))
    except OSError:
        return []

    files.sort(reverse=True)
    return [name for _modified_at, name in files[:limit]]


def _process_name_from_pid(pid: int, kernel32) -> str:
    if not pid:
        return ""

    process_query_limited_information = 0x1000
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return ""

    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buffer))
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return Path(buffer.value).name
        return ""
    finally:
        kernel32.CloseHandle(handle)


def _recent_file_names(limit: int) -> list[str]:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return []

    recent_path = Path(appdata) / "Microsoft" / "Windows" / "Recent"
    if not recent_path.exists():
        return []

    try:
        entries = sorted(
            (path for path in recent_path.iterdir() if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []

    return [path.stem for path in entries[:limit]]


def _tools_from_snapshot(snapshot: ContextSnapshot) -> tuple[str, ...]:
    tools: list[str] = []
    if snapshot.vscode_sessions or "code" in snapshot.active_app.lower():
        tools.append("VSCode")
    if snapshot.git_repositories:
        tools.append("Git")
    app = snapshot.active_app
    if app and app not in {"Code.exe", "code.exe"} and app not in tools:
        tools.append(app)
    return tuple(dict.fromkeys(tool for tool in tools if tool))


def _project_from_vscode(snapshot: ContextSnapshot) -> str:
    for session in snapshot.vscode_sessions:
        if session.folder:
            return session.folder
    return ""


def _project_from_git(snapshot: ContextSnapshot) -> str:
    if snapshot.git_repositories:
        return snapshot.git_repositories[0].path
    return ""


def _merge_names(left: tuple[str, ...], right: tuple[str, ...], limit: int = 20) -> tuple[str, ...]:
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
        if len(merged) >= limit:
            break
    return tuple(merged)


def _format_session_summary(session: _ActivitySession) -> str:
    start = _format_time(session.start_at)
    end = _format_time(session.last_seen_at)
    project = session.project or "um projeto nao identificado"
    tools = _human_join(session.tools) if session.tools else session.active_app or "ferramentas locais"
    return f"Entre as {start} e as {end} o Alexandre trabalhou no projeto {project} usando {tools}."


def _format_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%H:%M")


def _human_join(items: tuple[str, ...]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " e " + items[-1]
