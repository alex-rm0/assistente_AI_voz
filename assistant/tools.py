from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

from assistant.context_interpreter import interpret_snapshot
from assistant.context_reasoning import reason_about_context
from assistant.desktop_actions import (
    WindowsDesktopActionRunner,
    app_label,
    is_application_open,
    known_project_path,
    normalize_app_name,
    remember_desktop_action,
    resolve_application,
    resolve_safe_path,
    validate_url,
)
from assistant.tool_registry import tool_registry
from assistant.workspace import WorkspaceFileContent, WorkspaceGuard


DEFAULT_WORKSPACE_PATH = Path(__file__).resolve().parents[1] / "workspace"
NO_SYSTEM_CONTEXT_MESSAGE = (
    "Consigo tentar observar o computador, mas ainda não tenho dados suficientes. "
    "Experimenta mudar de janela ou aguardar alguns segundos."
)
INTERNAL_WORKSPACE_FILES = {"conversation.json", ".gitkeep"}
READABLE_EXTENSIONS = {".txt", ".md"}
DOCUMENT_EXTENSIONS = {".docx", ".pdf"}
ALL_READABLE_EXTENSIONS = READABLE_EXTENSIONS | DOCUMENT_EXTENSIONS
WRITABLE_EXTENSIONS = {".txt"}
DESKTOP_ACTION_RUNNER = WindowsDesktopActionRunner()


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


def _latest_context_snapshot(context_observer: Any | None):
    if context_observer is None:
        return None

    try:
        snapshot = context_observer.latest_snapshot()
    except Exception:
        snapshot = None

    if snapshot is not None:
        return snapshot

    try:
        return context_observer.observe_once()
    except Exception:
        return None


def _fresh_context_snapshot(context_observer: Any | None):
    if context_observer is None:
        return None

    try:
        snapshot = context_observer.observe_once()
    except Exception:
        snapshot = None

    if snapshot is not None:
        return snapshot

    return _latest_context_snapshot(context_observer)


def _allowed_desktop_roots(workspace_path: Path | None, project_root: Path | None) -> list[Path]:
    roots = []
    if workspace_path is not None:
        roots.append(Path(workspace_path).resolve())
    if project_root is not None:
        roots.append(Path(project_root).resolve())
    elif workspace_path is not None:
        roots.append(Path(workspace_path).resolve().parent)
    return roots


@tool_registry.register(
    name="get_presence_state",
    description=(
        "Devolve o estado real atual do PresenceManager. "
        "Usa SEMPRE que o utilizador perguntar em que modo ou estado de presenca o assistente esta."
    ),
    permissions=("read:presence",),
    remember_result=False,
)
def get_presence_state(presence_manager: Any | None = None) -> str:
    if presence_manager is None:
        return "A ferramenta de presença ainda não está ligada ao agente."

    state_report = getattr(presence_manager, "state_report", None)
    if state_report is not None:
        return state_report()

    state = getattr(presence_manager, "state", None)
    description = getattr(presence_manager, "description", None)
    if state is None:
        return "A ferramenta de presença ainda não está ligada ao agente."
    state_value = getattr(state, "value", str(state))
    details = description() if description is not None else ""
    return f"Estou em {state_value}. {details}".strip()


@tool_registry.register(
    name="open_application",
    description=(
        "Abre uma aplicacao permitida no Windows: Chrome, VS Code, Outlook, Teams, Discord, "
        "Bloco de Notas ou Explorador de Ficheiros. Requer confirmacao previa."
    ),
    permissions=("desktop:open_application", "confirmation-required"),
    remember_result=False,
)
def open_application(
    app_name: str,
    context_observer: Any | None = None,
    long_term_memory: Any | None = None,
    desktop_action_runner: Any | None = None,
) -> str:
    app_key, executable, uri = resolve_application(app_name)
    if not app_key:
        return "Nao posso abrir essa aplicacao. So posso abrir apps permitidas."
    label = app_label(app_key)
    if is_application_open(app_key, context_observer):
        return f"O {label} ja esta aberto."
    runner = desktop_action_runner or DESKTOP_ACTION_RUNNER
    result = runner.open_application(app_key, executable, uri)
    if result.ok:
        remember_desktop_action(long_term_memory, "abrir aplicacoes", label)
    return result.message


@tool_registry.register(
    name="open_file",
    description="Abre um ficheiro dentro da workspace ou de um projeto conhecido. Requer confirmacao previa.",
    permissions=("desktop:open_file", "confirmation-required", "safe-path-only"),
    remember_result=False,
)
def open_file(
    path: str,
    workspace_path: Path | None = None,
    project_root: Path | None = None,
    desktop_action_runner: Any | None = None,
) -> str:
    roots = _allowed_desktop_roots(workspace_path, project_root)
    resolved = resolve_safe_path(path, roots)
    if resolved is None or not resolved.is_file():
        return "Nao posso abrir esse ficheiro. So abro ficheiros dentro da workspace ou de projetos conhecidos."
    runner = desktop_action_runner or DESKTOP_ACTION_RUNNER
    return runner.open_path(resolved).message


@tool_registry.register(
    name="open_folder",
    description="Abre uma pasta dentro da workspace ou de um projeto conhecido. Requer confirmacao previa.",
    permissions=("desktop:open_folder", "confirmation-required", "safe-path-only"),
    remember_result=False,
)
def open_folder(
    path: str,
    workspace_path: Path | None = None,
    project_root: Path | None = None,
    desktop_action_runner: Any | None = None,
) -> str:
    roots = _allowed_desktop_roots(workspace_path, project_root)
    resolved = resolve_safe_path(path, roots)
    if resolved is None or not resolved.is_dir():
        return "Nao posso abrir essa pasta. So abro pastas dentro da workspace ou de projetos conhecidos."
    runner = desktop_action_runner or DESKTOP_ACTION_RUNNER
    return runner.open_path(resolved).message


@tool_registry.register(
    name="open_url",
    description="Abre um URL http/https no browser predefinido. Requer confirmacao previa.",
    permissions=("desktop:open_url", "confirmation-required"),
    remember_result=False,
)
def open_url(
    url: str,
    long_term_memory: Any | None = None,
    desktop_action_runner: Any | None = None,
) -> str:
    safe_url = validate_url(url)
    if not safe_url:
        return "Nao posso abrir esse URL. So aceito URLs http ou https validos."
    runner = desktop_action_runner or DESKTOP_ACTION_RUNNER
    result = runner.open_url(safe_url)
    if result.ok:
        remember_desktop_action(long_term_memory, "abrir URLs", safe_url)
    return result.message


@tool_registry.register(
    name="open_project",
    description="Abre um projeto conhecido, preferencialmente no VS Code. Requer confirmacao previa.",
    permissions=("desktop:open_project", "confirmation-required", "known-project-only"),
    remember_result=False,
)
def open_project(
    project_name: str,
    project_root: Path | None = None,
    known_projects: dict[str, str] | None = None,
    context_observer: Any | None = None,
    long_term_memory: Any | None = None,
    desktop_action_runner: Any | None = None,
) -> str:
    root = Path(project_root).resolve() if project_root is not None else DEFAULT_WORKSPACE_PATH.parent.resolve()
    resolved = known_project_path(project_name, known_projects or {}, root)
    if resolved is None:
        return "Nao conheco esse projeto ou a pasta ja nao existe."
    if is_application_open("vscode", context_observer):
        return "O VS Code ja esta aberto."
    _, executable, _uri = resolve_application("vscode")
    runner = desktop_action_runner or DESKTOP_ACTION_RUNNER
    result = runner.open_project(executable, resolved)
    if result.ok:
        remember_desktop_action(long_term_memory, "abrir projetos", resolved.name)
    return result.message


def _missing_task_manager_message() -> str:
    return "Não consegui alterar a tarefa. Ela continua pendente."


def _task_action_query_is_ambiguous(query: str) -> bool:
    normalized = _normalize_name(query).strip(" .,!?:;")
    return normalized in {
        "",
        "essa tarefa",
        "esta tarefa",
        "a tarefa",
        "esse lembrete",
        "este lembrete",
        "o lembrete",
        "isso",
    }


def _pending_task_count(long_term_memory: Any | None) -> int:
    if long_term_memory is None:
        return 0
    counter = getattr(long_term_memory, "pending_task_count", None)
    if counter is None:
        return 0
    try:
        return int(counter())
    except Exception:
        return 0


def _record_task_timeline(long_term_memory: Any | None, summary: str) -> None:
    if long_term_memory is None:
        return
    remember_timeline_event = getattr(long_term_memory, "remember_timeline_event", None)
    if remember_timeline_event is None:
        return
    try:
        remember_timeline_event(summary)
    except Exception:
        return


@tool_registry.register(
    name="list_pending_tasks",
    description="Lista tarefas pendentes atuais a partir da base SQLite local.",
    permissions=("read:tasks",),
    remember_result=False,
)
def list_pending_tasks(long_term_memory: Any | None = None, show_details: bool = False) -> str:
    if long_term_memory is None:
        return "Não consegui consultar as tarefas pendentes."
    pending_tasks = getattr(long_term_memory, "pending_tasks", None)
    if pending_tasks is None:
        return "Não consegui consultar as tarefas pendentes."
    return pending_tasks(show_details=show_details)


@tool_registry.register(
    name="complete_task",
    description="Marca uma tarefa pendente como concluida na base SQLite local.",
    permissions=("write:tasks",),
    remember_result=False,
)
def complete_task(long_term_memory: Any | None = None, query: str = "") -> str:
    if long_term_memory is None:
        return _missing_task_manager_message()
    count = _pending_task_count(long_term_memory)
    if count > 1 and _task_action_query_is_ambiguous(query):
        return "Tenho várias tarefas pendentes. Diz-me qual queres marcar como concluída."

    updater = getattr(long_term_memory, "complete_task", None)
    if updater is None:
        return _missing_task_manager_message()
    result = updater(query)
    if "Nao encontrei" in result or "Não encontrei" in result:
        return _missing_task_manager_message()

    _record_task_timeline(long_term_memory, f"Tarefa concluida: {query or 'tarefa pendente unica'}.")
    return "Está feito. Marquei essa tarefa como concluída e já não aparece nas pendentes."


@tool_registry.register(
    name="cancel_task",
    description="Cancela uma tarefa pendente na base SQLite local.",
    permissions=("write:tasks",),
    remember_result=False,
)
def cancel_task(long_term_memory: Any | None = None, query: str = "") -> str:
    if long_term_memory is None:
        return _missing_task_manager_message()
    count = _pending_task_count(long_term_memory)
    if count > 1 and _task_action_query_is_ambiguous(query):
        return "Tenho várias tarefas pendentes. Diz-me qual queres cancelar."

    updater = getattr(long_term_memory, "cancel_task", None)
    if updater is None:
        return _missing_task_manager_message()
    result = updater(query)
    if "Nao encontrei" in result or "Não encontrei" in result:
        return _missing_task_manager_message()

    _record_task_timeline(long_term_memory, f"Tarefa cancelada: {query or 'tarefa pendente unica'}.")
    return "Está feito. Cancelei essa tarefa e já não aparece nas pendentes."


@tool_registry.register(
    name="postpone_task",
    description="Adia uma tarefa pendente na base SQLite local.",
    permissions=("write:tasks",),
    remember_result=False,
)
def postpone_task(long_term_memory: Any | None = None, query: str = "") -> str:
    if long_term_memory is None:
        return _missing_task_manager_message()
    count = _pending_task_count(long_term_memory)
    if count > 1 and _task_action_query_is_ambiguous(query):
        return "Tenho várias tarefas pendentes. Diz-me qual queres adiar."

    updater = getattr(long_term_memory, "postpone_task", None)
    if updater is None:
        return _missing_task_manager_message()
    result = updater(query)
    if "Nao encontrei" in result or "Não encontrei" in result:
        return _missing_task_manager_message()

    _record_task_timeline(long_term_memory, f"Tarefa adiada: {query or 'tarefa pendente unica'}.")
    return result


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
    name="get_active_window",
    description=(
        "Devolve a janela ativa detetada pelo Context Observer. "
        "Usa SEMPRE que o utilizador perguntar qual e a janela ativa ou que janela esta a ser usada."
    ),
    permissions=("read:context_observer",),
    remember_result=False,
)
def get_active_window(context_observer: Any | None = None) -> str:
    snapshot = _fresh_context_snapshot(context_observer)
    if snapshot is None or not snapshot.active_window:
        return NO_SYSTEM_CONTEXT_MESSAGE

    return f"Janela ativa detetada: {snapshot.active_window}"


@tool_registry.register(
    name="get_active_application",
    description=(
        "Devolve a aplicacao ativa detetada pelo Context Observer. "
        "Usa SEMPRE que o utilizador perguntar que aplicacao/programa esta ativo."
    ),
    permissions=("read:context_observer",),
    remember_result=False,
)
def get_active_application(context_observer: Any | None = None) -> str:
    snapshot = _fresh_context_snapshot(context_observer)
    if snapshot is None or not snapshot.active_app:
        return NO_SYSTEM_CONTEXT_MESSAGE

    return f"Aplicacao ativa detetada: {snapshot.active_app}"


@tool_registry.register(
    name="get_open_windows",
    description=(
        "Lista janelas abertas detetadas pelo Context Observer. "
        "Usa SEMPRE que o utilizador perguntar por janelas, aplicacoes abertas ou programas abertos."
    ),
    permissions=("read:context_observer",),
    remember_result=False,
)
def get_open_windows(context_observer: Any | None = None) -> str:
    snapshot = _fresh_context_snapshot(context_observer)
    if snapshot is None:
        return NO_SYSTEM_CONTEXT_MESSAGE

    if not snapshot.open_windows and snapshot.active_window:
        app_label = f" [{snapshot.active_app}]" if snapshot.active_app else ""
        return f"Janelas detetadas:\n- {snapshot.active_window}{app_label} (ativa)"

    if not snapshot.open_windows:
        return NO_SYSTEM_CONTEXT_MESSAGE

    lines = ["Janelas detetadas:"]
    for window in snapshot.open_windows:
        active_marker = " (ativa)" if getattr(window, "is_active", False) else ""
        process_name = getattr(window, "process_name", "")
        title = getattr(window, "title", "")
        if process_name:
            lines.append(f"- {title} [{process_name}]{active_marker}")
        else:
            lines.append(f"- {title}{active_marker}")
    return "\n".join(lines)


@tool_registry.register(
    name="get_recent_activity",
    description=(
        "Devolve atividade recente resumida pelo Context Observer. "
        "Usa SEMPRE que o utilizador perguntar por atividade, monitorizacao ou programas usados recentemente."
    ),
    permissions=("read:context_observer",),
    remember_result=False,
)
def get_recent_activity(context_observer: Any | None = None) -> str:
    if context_observer is None:
        return NO_SYSTEM_CONTEXT_MESSAGE

    latest_summary = context_observer.latest_summary()
    if latest_summary is not None and latest_summary.summary:
        return f"Atividade recente: {latest_summary.summary}"

    summary = context_observer.activity_summary(limit=5)
    if not summary:
        return NO_SYSTEM_CONTEXT_MESSAGE

    lines = ["Atividade recente detetada:"]
    for active_app, active_window, current_project, total_seconds in summary:
        minutes = max(1, round(total_seconds / 60))
        project = f" no projeto {current_project}" if current_project else ""
        window = f" ({active_window})" if active_window else ""
        lines.append(f"- {active_app}{window}{project}: cerca de {minutes} min")
    return "\n".join(lines)


@tool_registry.register(
    name="get_last_context_snapshot",
    description=(
        "Devolve o ultimo snapshot resumido do Context Observer: janela ativa, aplicacao ativa, "
        "janelas detetadas, processos, sessoes VSCode, repositorios Git e ficheiros modificados."
    ),
    permissions=("read:context_observer",),
    remember_result=False,
)
def get_last_context_snapshot(context_observer: Any | None = None) -> str:
    snapshot = _latest_context_snapshot(context_observer)
    if snapshot is None:
        return NO_SYSTEM_CONTEXT_MESSAGE

    debug_context = bool(getattr(context_observer, "debug_context", False))
    interpreted = interpret_snapshot(snapshot, debug_context=debug_context)
    raw_snapshot = _format_raw_context_snapshot(snapshot)
    return f"{interpreted}\n\nSnapshot bruto:\n{raw_snapshot}"


@tool_registry.register(
    name="get_current_activity_summary",
    description=(
        "Analisa o snapshot do Context Observer e devolve uma conclusao suportada por evidencias "
        "sobre a atividade principal atual, projeto principal, aplicacoes relevantes, objetivos possiveis "
        "e sugestoes opcionais."
    ),
    permissions=("read:context_observer",),
    remember_result=False,
)
def get_current_activity_summary(
    context_observer: Any | None = None,
    active_contexts: list[str] | None = None,
    relevant_memory: str = "",
    pending_tasks: str = "",
) -> str:
    snapshot = _latest_context_snapshot(context_observer)
    if snapshot is None:
        return NO_SYSTEM_CONTEXT_MESSAGE
    return reason_about_context(
        snapshot=snapshot,
        active_contexts=active_contexts or [],
        relevant_memory=relevant_memory,
        pending_tasks=pending_tasks,
    ).format()


@tool_registry.register(
    name="get_raw_context_snapshot",
    description=(
        "Devolve o snapshot bruto do Context Observer para debug tecnico. "
        "Usa apenas quando DEBUG_CONTEXT estiver ativo ou quando o utilizador pedir explicitamente dados brutos."
    ),
    permissions=("read:context_observer",),
    remember_result=False,
)
def get_raw_context_snapshot(context_observer: Any | None = None) -> str:
    snapshot = _latest_context_snapshot(context_observer)
    if snapshot is None:
        return NO_SYSTEM_CONTEXT_MESSAGE

    return "Ultimo snapshot do computador:\n" + _format_raw_context_snapshot(snapshot)


def _format_raw_context_snapshot(snapshot: Any) -> str:
    lines: list[str] = []
    if snapshot.active_app:
        lines.append(f"- Aplicacao ativa: {snapshot.active_app}")
    if snapshot.active_window:
        lines.append(f"- Janela ativa: {snapshot.active_window}")
    if snapshot.current_project:
        lines.append(f"- Projeto atual: {snapshot.current_project}")
    if snapshot.open_windows:
        lines.append("- Janelas detetadas:")
        for window in snapshot.open_windows[:10]:
            active_marker = " (ativa)" if getattr(window, "is_active", False) else ""
            process_name = getattr(window, "process_name", "")
            title = getattr(window, "title", "")
            label = f" [{process_name}]" if process_name else ""
            lines.append(f"  - {title}{label}{active_marker}")
    if snapshot.active_processes:
        lines.append(f"- Processos ativos detetados: {len(snapshot.active_processes)}")
    if snapshot.vscode_sessions:
        lines.append("- Sessoes VSCode:")
        for session in snapshot.vscode_sessions[:5]:
            folder = f" -> {session.folder}" if session.folder else ""
            lines.append(f"  - {session.window_title}{folder}")
    if snapshot.git_repositories:
        lines.append("- Repositorios Git:")
        for repo in snapshot.git_repositories[:5]:
            branch = f" ({repo.branch})" if repo.branch else ""
            lines.append(f"  - {repo.path}{branch}")
    if snapshot.recently_modified_files:
        lines.append("- Ficheiros modificados recentemente:")
        lines.extend(f"  - {name}" for name in snapshot.recently_modified_files[:10])

    if not lines:
        return NO_SYSTEM_CONTEXT_MESSAGE
    return "\n".join(lines)


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
            f"O ficheiro '{raw_filename}' nao existe na pasta workspace.",
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
