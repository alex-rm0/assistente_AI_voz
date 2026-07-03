from __future__ import annotations

from assistant.context_interpreter import interpret_snapshot
from assistant.context_observer import ContextSnapshot, GitRepoContext, VscodeSession, WindowInfo


def test_interpret_snapshot_summarizes_human_context() -> None:
    snapshot = ContextSnapshot(
        active_app="Code.exe",
        active_window="assistant/agent.py - assistenteIA - Visual Studio Code",
        current_project="AssistenteIA",
        open_windows=(
            WindowInfo("assistant/agent.py - assistenteIA - Visual Studio Code", "Code.exe", 1, True),
            WindowInfo("Codex - AssistenteIA", "Codex.exe", 2, False),
            WindowInfo("AssistenteIA conversa - Google Chrome", "chrome.exe", 3, False),
        ),
        vscode_sessions=(VscodeSession("assistant/agent.py - assistenteIA - Visual Studio Code", "assistenteIA", 1),),
        git_repositories=(GitRepoContext("assistenteIA", "main", ("assistant/agent.py", "docs/architecture.md")),),
        recently_modified_files=("README.md", "assistant/agent.py", "docs/architecture.md"),
    )

    result = interpret_snapshot(snapshot)

    assert "Estás a trabalhar no projeto AssistenteIA." in result
    assert "VS Code aberto no repositório assistenteIA" in result
    assert "Codex aberto" in result
    assert "browser aberto com contexto relacionado com o projeto" in result
    assert "README.md" in result
    assert "assistant/agent.py" in result


def test_interpret_snapshot_ignores_noise_windows() -> None:
    snapshot = ContextSnapshot(
        active_app="TextInputHost.exe",
        active_window="",
        open_windows=(
            WindowInfo("Program Manager", "Program Manager", 1, False),
            WindowInfo("TextInputHost", "TextInputHost.exe", 2, False),
        ),
    )

    result = interpret_snapshot(snapshot)

    assert "dados suficientes" in result


def test_interpret_snapshot_debug_context_includes_raw_data() -> None:
    snapshot = ContextSnapshot(
        active_app="Code.exe",
        active_window="assistenteIA - Visual Studio Code",
        current_project="AssistenteIA",
        open_windows=(WindowInfo("assistenteIA - Visual Studio Code", "Code.exe", 1, True),),
    )

    result = interpret_snapshot(snapshot, debug_context=True)

    assert "[DEBUG_CONTEXT]" in result
    assert "Aplicação ativa: Code.exe" in result
