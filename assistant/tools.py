from __future__ import annotations

import unicodedata
from pathlib import Path

from assistant.tool_registry import tool_registry
from assistant.workspace import WorkspaceFileContent, WorkspaceGuard


DEFAULT_WORKSPACE_PATH = Path(__file__).resolve().parents[1] / "workspace"
INTERNAL_WORKSPACE_FILES = {"conversation.json", ".gitkeep"}
READABLE_EXTENSIONS = {".txt", ".md"}
DOCUMENT_EXTENSIONS = {".docx", ".pdf"}
ALL_READABLE_EXTENSIONS = READABLE_EXTENSIONS | DOCUMENT_EXTENSIONS
WRITABLE_EXTENSIONS = {".txt"}


def _normalize_name(text: str) -> str:
    """Lowercase + strip accents for fuzzy matching."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _resolve_workspace_file(raw_filename: str, root: Path) -> Path | None:
    """
    Try to find a readable file in the workspace by name.
    Attempts, in order:
      1. Exact path match.
      2. Add each known extension when the name has none.
      3. Fuzzy match: normalize unicode/case and check if any file stem
         starts with or contains the given name fragment.
    Returns the resolved Path or None if nothing matches.
    """
    candidate = Path(raw_filename)

    # 1. Exact match
    exact = root / raw_filename
    if exact.exists() and exact.is_file() and exact.suffix.lower() in ALL_READABLE_EXTENSIONS:
        return exact

    # 2. Try appending extensions when no extension was given
    if not candidate.suffix:
        for ext in sorted(ALL_READABLE_EXTENSIONS):
            p = root / (raw_filename + ext)
            if p.exists() and p.is_file():
                return p

    # 3. Fuzzy: normalize and look for a partial stem match
    norm_input = _normalize_name(candidate.stem or raw_filename)
    best: Path | None = None
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in ALL_READABLE_EXTENSIONS:
            continue
        norm_stem = _normalize_name(f.stem)
        if norm_stem.startswith(norm_input) or norm_input in norm_stem:
            best = f
            break

    return best


@tool_registry.register(
    name="list_workspace_files",
    description=(
        "Lista os ficheiros existentes dentro da pasta workspace. "
        "Usa APENAS quando o utilizador pede explicitamente para listar, ver ou mostrar os ficheiros da pasta workspace."
    ),
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
    description=(
        "Le o conteudo de um ficheiro dentro da pasta workspace. "
        "Suporta .txt, .md, .docx e .pdf. "
        "Usa APENAS quando o utilizador menciona ou implica claramente um nome de ficheiro especifico para ler ou resumir. "
        "NAO usar para perguntas gerais, conversas, ou quando nenhum ficheiro concreto e mencionado. "
        "Argumentos: filename (nome do ficheiro, com ou sem extensao)."
    ),
    permissions=("read:workspace",),
    remember_result=False,
)
def read_workspace_file(filename: str, workspace_path: Path | None = None) -> str:
    """Reads any supported file inside the workspace folder."""

    result = read_workspace_file_content(filename, workspace_path)
    if result.error is not None:
        return result.error

    if not result.content.strip():
        return f"O ficheiro '{result.filename}' esta vazio ou nao tem texto extraivel."

    return f"Conteudo de {result.filename}:\n\n{result.content}"


def read_workspace_file_content(filename: str, workspace_path: Path | None = None) -> WorkspaceFileContent:
    """Returns raw text content for any supported file inside workspace only."""

    from assistant.document_reader import read_docx_content, read_pdf_content

    if not filename or not filename.strip():
        return WorkspaceFileContent("", "", "Indica o nome do ficheiro que queres ler.")

    raw_filename = filename.strip().strip("\"'")
    candidate = Path(raw_filename)

    if candidate.is_absolute() or ".." in candidate.parts:
        return WorkspaceFileContent(raw_filename, "", "Nao posso ler caminhos fora da pasta workspace.")

    workspace = WorkspaceGuard(workspace_path or DEFAULT_WORKSPACE_PATH, create=False)
    root = workspace.resolve()

    if not root.exists():
        return WorkspaceFileContent(raw_filename, "", "A pasta workspace nao existe.")

    # Resolve the actual file path (with fuzzy matching)
    try:
        workspace.resolve(raw_filename)  # security check on the raw input
    except ValueError:
        return WorkspaceFileContent(raw_filename, "", "Nao posso ler caminhos fora da pasta workspace.")

    resolved = _resolve_workspace_file(raw_filename, root)

    if resolved is None:
        return WorkspaceFileContent(
            raw_filename,
            "",
            f"Nao encontrei nenhum ficheiro com o nome '{raw_filename}' na pasta workspace.",
        )

    ext = resolved.suffix.lower()

    if ext == ".docx":
        return read_docx_content(resolved.relative_to(root).as_posix(), root)

    if ext == ".pdf":
        return read_pdf_content(resolved.relative_to(root).as_posix(), root)

    # .txt / .md
    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return WorkspaceFileContent(raw_filename, "", f"Nao consegui ler '{raw_filename}' como texto UTF-8.")
    except OSError as exc:
        return WorkspaceFileContent(raw_filename, "", f"Nao consegui ler '{raw_filename}': {exc}")

    return WorkspaceFileContent(resolved.relative_to(root).as_posix(), content)


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
