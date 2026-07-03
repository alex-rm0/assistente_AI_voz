from __future__ import annotations

from assistant.context_observer import ContextSnapshot, GitRepoContext, VscodeSession, WindowInfo
from assistant.context_reasoning import reason_about_context


def test_reasoning_identifies_development_activity_from_evidence() -> None:
    snapshot = ContextSnapshot(
        active_app="Code.exe",
        active_window="assistant/agent.py - assistenteIA - Visual Studio Code",
        current_project="AssistenteIA",
        open_windows=(
            WindowInfo("assistant/agent.py - assistenteIA - Visual Studio Code", "Code.exe", 10, True),
            WindowInfo("Codex - AssistenteIA", "Codex.exe", 11, False),
            WindowInfo("AssistenteIA conversa - Google Chrome", "chrome.exe", 12, False),
        ),
        vscode_sessions=(VscodeSession("assistant/agent.py - assistenteIA - Visual Studio Code", "assistenteIA", 10),),
        git_repositories=(GitRepoContext("assistenteIA", "main", ("assistant/agent.py", "docs/architecture.md")),),
        recently_modified_files=("README.md", "assistant/agent.py", "docs/architecture.md"),
    )

    result = reason_about_context(
        snapshot,
        active_contexts=("TECH_CONTEXT", "WORK_CONTEXT"),
        relevant_memory="Projeto AssistenteIA recorrente.",
        pending_tasks="Tarefas pendentes:\n- Rever arquitetura",
    )

    assert result.main_activity == "Parece que estás a trabalhar no desenvolvimento do AssistenteIA."
    assert result.main_project == "AssistenteIA"
    assert "VS Code" in result.relevant_applications
    assert "Git" in result.relevant_applications

    formatted = result.format()
    assert "Contexto atual observado agora:" in formatted
    assert "Atividade recente:" in formatted
    assert "Ficheiros recentemente modificados:" in formatted
    assert "Inferências prováveis:" in formatted
    assert "Evidências observadas:" in formatted
    assert "Parece que estás a trabalhar" in formatted
    assert any("Ficheiros modificados recentemente" in evidence for evidence in result.evidence)


def test_reasoning_uses_pt_pt_vocabulary() -> None:
    snapshot = ContextSnapshot(
        active_app="Code.exe",
        active_window="assistenteIA - Visual Studio Code",
        current_project="AssistenteIA",
        open_windows=(WindowInfo("assistenteIA - Visual Studio Code", "Code.exe", 10, True),),
        recently_modified_files=("assistant/context_reasoning.py",),
    )

    formatted = reason_about_context(snapshot).format().lower()

    assert "aplicação" in formatted or "aplicações" in formatted
    assert "ficheiros" in formatted
    assert "aplicativos" not in formatted
    assert "arquivos" not in formatted
    assert "tela" not in formatted
    assert "assistindo" not in formatted
    assert "acessar" not in formatted


def test_reasoning_does_not_invent_without_snapshot() -> None:
    result = reason_about_context(None)

    assert result.main_activity == ""
    assert result.evidence == ()
    assert "não tenho dados suficientes" in result.format()
