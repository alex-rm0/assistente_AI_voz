from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assistant.tool_registry import tool_registry


DEFAULT_WORKSPACE_PATH = Path(__file__).resolve().parents[1] / "workspace"
INTERNAL_WORKSPACE_FILES = {"conversation.json", ".gitkeep"}
READABLE_EXTENSIONS = {".txt", ".md"}
WRITABLE_EXTENSIONS = {".txt"}


@dataclass(frozen=True)
class WorkspaceFileContent:
    filename: str
    content: str
    error: str | None = None


class WorkspaceGuard:
    """Small safety helper for future tools that may read or write files."""

    def __init__(self, workspace_path: Path, create: bool = True) -> None:
        self.workspace_path = workspace_path.resolve()
        if create:
            self.workspace_path.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str = "") -> Path:
        target = (self.workspace_path / relative_path).resolve()
        if target != self.workspace_path and self.workspace_path not in target.parents:
            raise ValueError("Access outside the workspace folder is not allowed.")
        return target


@tool_registry.register(
    name="list_workspace_files",
    description="Lista os ficheiros existentes dentro da pasta workspace.",
    permissions=("read:workspace",),
)
def list_workspace_files(workspace_path: Path | None = None) -> str:
    """Lists user-visible files inside the workspace folder only."""

    workspace = WorkspaceGuard(workspace_path or DEFAULT_WORKSPACE_PATH, create=False)
    root = workspace.resolve()

    if not root.exists():
        return "A pasta workspace nao existe."

    try:
        files = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.name not in INTERNAL_WORKSPACE_FILES
        ]
    except OSError as exc:
        return f"Nao consegui listar a pasta workspace: {exc}"

    if not files:
        return "A pasta workspace esta vazia. Nao existem ficheiros para listar."

    relative_files = sorted(path.relative_to(root).as_posix() for path in files)
    file_list = "\n".join(f"- {file_path}" for file_path in relative_files)
    return f"Ficheiros na pasta workspace:\n{file_list}"


@tool_registry.register(
    name="read_workspace_file",
    description="Le o conteudo de um ficheiro .txt ou .md dentro da pasta workspace. Argumentos: filename.",
    permissions=("read:workspace",),
    remember_result=False,
)
def read_workspace_file(filename: str, workspace_path: Path | None = None) -> str:
    """Reads a .txt or .md file inside the workspace folder only."""

    result = read_workspace_file_content(filename, workspace_path)
    if result.error is not None:
        return result.error

    if not result.content.strip():
        return f"O ficheiro '{result.filename}' esta vazio."

    return f"Conteudo de {result.filename}:\n\n{result.content}"


def read_workspace_file_content(filename: str, workspace_path: Path | None = None) -> WorkspaceFileContent:
    """Returns raw text content for a .txt or .md file inside workspace only."""

    if not filename or not filename.strip():
        return WorkspaceFileContent("", "", "Indica o nome do ficheiro que queres ler.")

    raw_filename = filename.strip().strip("\"'")
    candidate = Path(raw_filename)

    if candidate.is_absolute() or ".." in candidate.parts:
        return WorkspaceFileContent(raw_filename, "", "Nao posso ler caminhos fora da pasta workspace.")

    if candidate.suffix.lower() not in READABLE_EXTENSIONS:
        return WorkspaceFileContent(
            raw_filename,
            "",
            "So posso ler ficheiros de texto simples com extensao .txt ou .md.",
        )

    workspace = WorkspaceGuard(workspace_path or DEFAULT_WORKSPACE_PATH, create=False)
    root = workspace.resolve()

    if not root.exists():
        return WorkspaceFileContent(raw_filename, "", "A pasta workspace nao existe.")

    try:
        target = workspace.resolve(raw_filename)
    except ValueError:
        return WorkspaceFileContent(raw_filename, "", "Nao posso ler caminhos fora da pasta workspace.")

    if not target.exists():
        return WorkspaceFileContent(raw_filename, "", f"O ficheiro '{raw_filename}' nao existe na pasta workspace.")

    if not target.is_file():
        return WorkspaceFileContent(raw_filename, "", f"'{raw_filename}' nao e um ficheiro.")

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return WorkspaceFileContent(raw_filename, "", f"Nao consegui ler '{raw_filename}' como texto UTF-8.")
    except OSError as exc:
        return WorkspaceFileContent(raw_filename, "", f"Nao consegui ler '{raw_filename}': {exc}")

    return WorkspaceFileContent(target.relative_to(root).as_posix(), content)


@tool_registry.register(
    name="create_workspace_file",
    description=(
        "Cria um novo ficheiro .txt dentro da pasta workspace sem sobrescrever ficheiros existentes. "
        "Argumentos: filename, content."
    ),
    permissions=("write:workspace", "no-overwrite"),
)
def create_workspace_file(filename: str, content: str, workspace_path: Path | None = None) -> str:
    """Create a new .txt file inside workspace without overwriting existing files."""

    if not filename or not filename.strip():
        return "Indica o nome do ficheiro que queres criar."

    raw_filename = filename.strip().strip("\"'")
    candidate = Path(raw_filename)

    if candidate.is_absolute() or ".." in candidate.parts:
        return "Nao posso criar ficheiros fora da pasta workspace."

    if candidate.suffix.lower() not in WRITABLE_EXTENSIONS:
        return "So posso criar ficheiros com extensao .txt."

    if not content.strip():
        return "Indica o texto que queres guardar no ficheiro."

    workspace = WorkspaceGuard(workspace_path or DEFAULT_WORKSPACE_PATH)
    root = workspace.resolve()

    try:
        target = workspace.resolve(raw_filename)
    except ValueError:
        return "Nao posso criar ficheiros fora da pasta workspace."

    if target.exists():
        return (
            f"O ficheiro '{raw_filename}' ja existe. "
            "Nao vou sobrescrever ficheiros nesta versao; no futuro posso pedir confirmacao."
        )

    if not target.parent.exists():
        return "A pasta de destino dentro da workspace nao existe."

    try:
        with target.open("x", encoding="utf-8") as file:
            file.write(content)
    except FileExistsError:
        return (
            f"O ficheiro '{raw_filename}' ja existe. "
            "Nao vou sobrescrever ficheiros nesta versao; no futuro posso pedir confirmacao."
        )
    except OSError as exc:
        return f"Nao consegui criar '{raw_filename}': {exc}"

    return f"Criei o ficheiro '{target.relative_to(root).as_posix()}' na pasta workspace."
