from __future__ import annotations

from pathlib import Path

from assistant.context_observer import (
    ContextObserver,
    ContextSnapshot,
    GitRepoContext,
    ProcessInfo,
    VscodeSession,
    WindowInfo,
    _is_noise_window,
)


def test_context_observer_stores_latest_snapshot(tmp_path: Path) -> None:
    snapshots = iter(
        [
            ContextSnapshot(
                active_app="Code.exe",
                active_window="assistenteIA - Visual Studio Code",
                recent_files=("nota.txt", "relatorio.pdf"),
                observed_at=100.0,
            )
        ]
    )
    observer = ContextObserver(
        data_path=tmp_path,
        project_root=tmp_path / "assistenteIA",
        snapshot_provider=lambda: next(snapshots),
    )

    observer.observe_once()
    latest = observer.latest_snapshot()

    assert latest is not None
    assert latest.active_app == "Code.exe"
    assert latest.active_window == "assistenteIA - Visual Studio Code"
    assert latest.current_project == "assistenteIA"
    assert latest.recent_files == ("nota.txt", "relatorio.pdf")


def test_context_observer_accumulates_activity_time(tmp_path: Path) -> None:
    values = iter(
        [
            ContextSnapshot("Code.exe", "assistenteIA - Visual Studio Code", observed_at=100.0),
            ContextSnapshot("Code.exe", "assistenteIA - Visual Studio Code", observed_at=115.0),
            ContextSnapshot("Browser.exe", "Documentacao", observed_at=125.0),
        ]
    )
    observer = ContextObserver(
        data_path=tmp_path,
        project_root=tmp_path / "assistenteIA",
        snapshot_provider=lambda: next(values),
    )

    observer.observe_once()
    observer.observe_once()
    observer.observe_once()
    summary = observer.activity_summary()

    assert summary[0][0] == "Code.exe"
    assert summary[0][3] == 25.0


def test_context_observer_database_must_stay_inside_data(tmp_path: Path) -> None:
    try:
        ContextObserver(
            data_path=tmp_path / "data",
            project_root=tmp_path,
            db_file="../outside.sqlite",
            snapshot_provider=lambda: ContextSnapshot("", ""),
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for a database outside data_path")


def test_context_observer_builds_rich_snapshot(tmp_path: Path) -> None:
    observer = ContextObserver(
        data_path=tmp_path,
        project_root=tmp_path / "assistenteIA",
        snapshot_provider=lambda: ContextSnapshot(
            active_app="Code.exe",
            active_window="main.py - assistenteIA - Visual Studio Code",
            open_windows=(
                WindowInfo(
                    title="main.py - assistenteIA - Visual Studio Code",
                    process_name="Code.exe",
                    process_id=10,
                    is_active=True,
                ),
            ),
            active_processes=(ProcessInfo(process_id=10, name="Code.exe"),),
            vscode_sessions=(
                VscodeSession(
                    window_title="main.py - assistenteIA - Visual Studio Code",
                    folder="assistenteIA",
                    process_id=10,
                ),
            ),
            git_repositories=(
                GitRepoContext(
                    path="assistenteIA",
                    branch="main",
                    modified_files=("assistant/context_observer.py",),
                ),
            ),
            recently_modified_files=("assistant/context_observer.py",),
            observed_at=100.0,
        ),
    )

    snapshot = observer.observe_once()

    assert snapshot.open_windows[0].title.startswith("main.py")
    assert snapshot.active_processes[0].name == "Code.exe"
    assert snapshot.vscode_sessions[0].folder == "assistenteIA"
    assert snapshot.git_repositories[0].branch == "main"
    assert snapshot.recently_modified_files == ("assistant/context_observer.py",)


def test_context_observer_summarizes_sessions_instead_of_every_event(tmp_path: Path) -> None:
    summaries = []
    values = iter(
        [
            ContextSnapshot(
                active_app="Code.exe",
                active_window="assistenteIA - Visual Studio Code",
                current_project="AssistenteIA",
                vscode_sessions=(VscodeSession("assistenteIA - Visual Studio Code", "AssistenteIA", 10),),
                git_repositories=(GitRepoContext("AssistenteIA", "main"),),
                observed_at=100.0,
            ),
            ContextSnapshot(
                active_app="Code.exe",
                active_window="assistenteIA - Visual Studio Code",
                current_project="AssistenteIA",
                vscode_sessions=(VscodeSession("assistenteIA - Visual Studio Code", "AssistenteIA", 10),),
                git_repositories=(GitRepoContext("AssistenteIA", "main"),),
                observed_at=220.0,
            ),
            ContextSnapshot(
                active_app="Browser.exe",
                active_window="Documentacao",
                observed_at=230.0,
            ),
        ]
    )
    observer = ContextObserver(
        data_path=tmp_path,
        project_root=tmp_path / "assistenteIA",
        snapshot_provider=lambda: next(values),
        summary_callback=summaries.append,
        summary_min_seconds=60.0,
    )

    observer.observe_once()
    observer.observe_once()
    observer.observe_once()
    latest = observer.latest_summary()

    assert len(summaries) == 1
    assert latest is not None
    assert "projeto AssistenteIA" in latest.summary
    assert "VSCode e Git" in latest.summary


def test_context_observer_filters_technical_noise_windows() -> None:
    assert _is_noise_window("Program Manager", "explorer.exe")
    assert _is_noise_window("TextInputHost", "TextInputHost.exe")
    assert _is_noise_window("NVIDIA Overlay", "NVIDIA Overlay.exe")
    assert not _is_noise_window("assistenteIA - Visual Studio Code", "Code.exe")
