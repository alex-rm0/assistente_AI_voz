from __future__ import annotations

from pathlib import Path

import pytest

from assistant.tools import (
    WorkspaceGuard,
    create_workspace_file,
    get_active_application,
    get_active_window,
    get_last_context_snapshot,
    get_open_windows,
    get_presence_state,
    get_recent_activity,
    list_workspace_files,
    read_workspace_file,
)
from assistant.presence_manager import PresenceManager, PresenceState


def test_list_workspace_files_shows_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nota.txt").write_text("ola", encoding="utf-8")
    (workspace / "docs").mkdir()
    (workspace / "docs" / "guia.md").write_text("guia", encoding="utf-8")

    result = list_workspace_files(workspace)

    assert "Ficheiros na pasta workspace:" in result
    assert "- nota.txt" in result
    assert "- docs/guia.md" in result


def test_list_workspace_files_empty_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = list_workspace_files(workspace)

    assert result == "A pasta workspace esta vazia. Nao existem ficheiros para listar."


def test_read_workspace_file_reads_txt(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nota.txt").write_text("conteudo de teste", encoding="utf-8")

    result = read_workspace_file("nota.txt", workspace)

    assert "Conteudo de nota.txt:" in result
    assert "conteudo de teste" in result


def test_read_workspace_file_missing_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = read_workspace_file("nao_existe.txt", workspace)

    assert "nao existe na pasta workspace" in result


def test_create_workspace_file_creates_txt(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = create_workspace_file("nova.txt", "texto novo", workspace)

    assert result == "Criei o ficheiro 'nova.txt' na pasta workspace."
    assert (workspace / "nova.txt").read_text(encoding="utf-8") == "texto novo"


def test_create_workspace_file_does_not_overwrite(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "nota.txt"
    target.write_text("original", encoding="utf-8")

    result = create_workspace_file("nota.txt", "novo", workspace)

    assert "ja existe" in result
    assert target.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize(
    ("filename", "operation"),
    [
        ("../fora.txt", "read"),
        ("subpasta/../../fora.txt", "read"),
        ("../fora.txt", "create"),
        ("subpasta/../../fora.txt", "create"),
    ],
)
def test_path_traversal_is_blocked(tmp_path: Path, filename: str, operation: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "fora.txt"
    outside.write_text("segredo", encoding="utf-8")

    if operation == "read":
        result = read_workspace_file(filename, workspace)
    else:
        result = create_workspace_file(filename, "novo", workspace)

    assert "fora da pasta workspace" in result
    assert outside.read_text(encoding="utf-8") == "segredo"


def test_absolute_path_outside_workspace_is_blocked(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "fora.txt"
    outside.write_text("segredo", encoding="utf-8")

    read_result = read_workspace_file(str(outside), workspace)
    create_result = create_workspace_file(str(outside), "novo", workspace)

    assert "fora da pasta workspace" in read_result
    assert "fora da pasta workspace" in create_result
    assert outside.read_text(encoding="utf-8") == "segredo"


def test_workspace_guard_blocks_outside_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    guard = WorkspaceGuard(workspace)

    with pytest.raises(ValueError):
        guard.resolve("../fora.txt")


def test_context_observer_tools_return_clear_message_without_observer() -> None:
    expected = (
        "Consigo tentar observar o computador, mas ainda não tenho dados suficientes. "
        "Experimenta mudar de janela ou aguardar alguns segundos."
    )

    assert get_active_window() == expected
    assert get_active_application() == expected
    assert get_open_windows() == expected
    assert get_recent_activity() == expected
    assert get_last_context_snapshot() == expected


def test_get_presence_state_uses_presence_manager() -> None:
    presence = PresenceManager(PresenceState.FOCUS_MODE)

    result = get_presence_state(presence)

    assert result.startswith("Estou em FOCUS_MODE.")
