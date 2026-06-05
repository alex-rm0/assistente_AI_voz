from __future__ import annotations

from pathlib import Path

from assistant.tool_registry import tool_registry
from assistant.workspace import WorkspaceFileContent, WorkspaceGuard

DEFAULT_WORKSPACE_PATH = Path(__file__).resolve().parents[1] / "workspace"


def read_docx_content(filename: str, workspace_path: Path | None = None) -> WorkspaceFileContent:
    """Returns the text content of a .docx file inside the workspace only."""

    if not filename or not filename.strip():
        return WorkspaceFileContent("", "", "Indica o nome do ficheiro que queres ler.")

    raw_filename = filename.strip().strip("\"'")
    candidate = Path(raw_filename)

    if candidate.is_absolute() or ".." in candidate.parts:
        return WorkspaceFileContent(raw_filename, "", "Nao posso ler caminhos fora da pasta workspace.")

    if candidate.suffix.lower() != ".docx":
        return WorkspaceFileContent(raw_filename, "", "Esta funcao so le ficheiros com extensao .docx.")

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
        import docx  # type: ignore

        doc = docx.Document(str(target))
        content = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except ImportError:
        return WorkspaceFileContent(raw_filename, "", "A biblioteca python-docx nao esta instalada.")
    except Exception as exc:
        return WorkspaceFileContent(raw_filename, "", f"Nao consegui ler '{raw_filename}': {exc}")

    return WorkspaceFileContent(target.relative_to(root).as_posix(), content)


def read_pdf_content(filename: str, workspace_path: Path | None = None) -> WorkspaceFileContent:
    """Returns the text content of a .pdf file inside the workspace only."""

    if not filename or not filename.strip():
        return WorkspaceFileContent("", "", "Indica o nome do ficheiro que queres ler.")

    raw_filename = filename.strip().strip("\"'")
    candidate = Path(raw_filename)

    if candidate.is_absolute() or ".." in candidate.parts:
        return WorkspaceFileContent(raw_filename, "", "Nao posso ler caminhos fora da pasta workspace.")

    if candidate.suffix.lower() != ".pdf":
        return WorkspaceFileContent(raw_filename, "", "Esta funcao so le ficheiros com extensao .pdf.")

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
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(target))
        pages = [page.extract_text() or "" for page in reader.pages]
        content = "\n".join(pages)
    except ImportError:
        return WorkspaceFileContent(raw_filename, "", "A biblioteca pypdf nao esta instalada.")
    except Exception as exc:
        return WorkspaceFileContent(raw_filename, "", f"Nao consegui ler '{raw_filename}': {exc}")

    return WorkspaceFileContent(target.relative_to(root).as_posix(), content)


@tool_registry.register(
    name="read_workspace_docx",
    description="Le o conteudo de um ficheiro .docx dentro da pasta workspace. Argumentos: filename.",
    permissions=("read:workspace",),
    remember_result=False,
)
def read_workspace_docx(filename: str, workspace_path: Path | None = None) -> str:
    """Reads a .docx file inside the workspace folder only."""

    result = read_docx_content(filename, workspace_path)
    if result.error is not None:
        return result.error

    if not result.content.strip():
        return f"O ficheiro '{result.filename}' esta vazio ou nao tem texto extraivel."

    return f"Conteudo de {result.filename}:\n\n{result.content}"


@tool_registry.register(
    name="read_workspace_pdf",
    description="Le o conteudo de um ficheiro .pdf dentro da pasta workspace. Argumentos: filename.",
    permissions=("read:workspace",),
    remember_result=False,
)
def read_workspace_pdf(filename: str, workspace_path: Path | None = None) -> str:
    """Reads a .pdf file inside the workspace folder only."""

    result = read_pdf_content(filename, workspace_path)
    if result.error is not None:
        return result.error

    if not result.content.strip():
        return f"O ficheiro '{result.filename}' esta vazio ou nao tem texto extraivel."

    return f"Conteudo de {result.filename}:\n\n{result.content}"
