from __future__ import annotations

from pathlib import Path

from assistant.agent import Agent, AgentContext
from assistant.context_observer import ContextSnapshot, ContextSummary, GitRepoContext, VscodeSession, WindowInfo
from assistant.prompts import get_system_prompt
from assistant.tool_registry import ToolRegistry
from assistant.tools import (
    create_workspace_file,
    get_active_application,
    get_active_window,
    get_current_activity_summary,
    get_last_context_snapshot,
    get_open_windows,
    get_presence_state,
    get_recent_activity,
    list_workspace_files,
    read_workspace_file,
)
from assistant.presence_manager import PresenceManager, PresenceState


class FakeLLM:
    def __init__(self) -> None:
        self.chat_prompts: list[str] = []

    def choose_tool(self, user_message, tools_description, profile_name=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_prompts.append(user_message)
        if "Cria uma nota curta" in user_message:
            return "Ponto principal do ficheiro."
        if "Resume em portugues" in user_message:
            return "Resumo do primeiro ficheiro."
        if "sugere uma organizacao" in user_message:
            return "Sugestao de organizacao."
        if "parecem relevantes" in user_message:
            return "documento_rvcc.txt parece relevante."
        return "resposta direta"


class FakeContextObserver:
    def latest_snapshot(self):
        return ContextSnapshot(
            active_app="Code.exe",
            active_window="assistenteIA - Visual Studio Code",
            open_windows=(
                WindowInfo(
                    title="assistenteIA - Visual Studio Code",
                    process_name="Code.exe",
                    process_id=10,
                    is_active=True,
                ),
            ),
            observed_at=100.0,
        )

    def latest_summary(self):
        return ContextSummary(
            summary="Entre as 09:00 e as 11:00 o Alexandre trabalhou no projeto AssistenteIA usando VSCode e Git.",
            project="AssistenteIA",
            start_at=100.0,
            end_at=200.0,
            tools=("VSCode", "Git"),
        )

    def activity_summary(self, limit=5):
        return [("Code.exe", "assistenteIA - Visual Studio Code", "AssistenteIA", 120.0)]


class FakeStaleThenFreshContextObserver:
    def latest_snapshot(self):
        return ContextSnapshot(
            active_app="python3.11.exe",
            active_window="AssistenteIA",
            open_windows=(
                WindowInfo(
                    title="AssistenteIA",
                    process_name="python3.11.exe",
                    process_id=1,
                    is_active=True,
                ),
            ),
            observed_at=100.0,
        )

    def observe_once(self):
        return ContextSnapshot(
            active_app="Code.exe",
            active_window="assistant/agent.py - assistenteIA - Visual Studio Code",
            open_windows=(
                WindowInfo(
                    title="assistant/agent.py - assistenteIA - Visual Studio Code",
                    process_name="Code.exe",
                    process_id=10,
                    is_active=True,
                ),
                WindowInfo(
                    title="Codex - AssistenteIA",
                    process_name="Codex.exe",
                    process_id=11,
                    is_active=False,
                ),
                WindowInfo(
                    title="AssistenteIA - Google Chrome",
                    process_name="chrome.exe",
                    process_id=12,
                    is_active=False,
                ),
            ),
            observed_at=200.0,
        )

    def latest_summary(self):
        return None

    def activity_summary(self, limit=5):
        return []


class FakeReasoningContextObserver:
    def latest_snapshot(self):
        return ContextSnapshot(
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
            observed_at=200.0,
        )

    def observe_once(self):
        return self.latest_snapshot()

    def latest_summary(self):
        return None

    def activity_summary(self, limit=5):
        return []


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register("list_workspace_files", "Lista ficheiros.", ("read:workspace",))(
        list_workspace_files
    )
    registry.register("read_workspace_file", "Le ficheiros.", ("read:workspace",), remember_result=False)(
        read_workspace_file
    )
    registry.register(
        "create_workspace_file",
        "Cria ficheiros .txt.",
        ("write:workspace", "no-overwrite"),
    )(create_workspace_file)
    registry.register("get_active_window", "Mostra janela ativa.", ("read:context_observer",), remember_result=False)(
        get_active_window
    )
    registry.register(
        "get_active_application",
        "Mostra aplicacao ativa.",
        ("read:context_observer",),
        remember_result=False,
    )(get_active_application)
    registry.register("get_open_windows", "Mostra janelas abertas.", ("read:context_observer",), remember_result=False)(
        get_open_windows
    )
    registry.register(
        "get_recent_activity",
        "Mostra atividade recente.",
        ("read:context_observer",),
        remember_result=False,
    )(get_recent_activity)
    registry.register(
        "get_last_context_snapshot",
        "Mostra ultimo snapshot.",
        ("read:context_observer",),
        remember_result=False,
    )(get_last_context_snapshot)
    registry.register(
        "get_current_activity_summary",
        "Resume atividade atual.",
        ("read:context_observer",),
        remember_result=False,
    )(get_current_activity_summary)
    registry.register(
        "get_presence_state",
        "Mostra estado de presenca.",
        ("read:presence",),
        remember_result=False,
    )(get_presence_state)

    return registry


def make_registry_without_context_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("list_workspace_files", "Lista ficheiros.", ("read:workspace",))(
        list_workspace_files
    )
    return registry


def make_context() -> AgentContext:
    return AgentContext(
        profile_name="Geral",
        system_prompt=get_system_prompt("Geral"),
        history=[],
    )


def test_compound_list_and_summarize_first_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("conteudo do primeiro", encoding="utf-8")
    (workspace / "b.txt").write_text("outro conteudo", encoding="utf-8")

    agent = Agent(FakeLLM(), make_registry(), workspace, debug_agent=True)

    result = agent.run("Lista os ficheiros da workspace e resume o primeiro.", make_context())

    assert "Resumo do primeiro ficheiro." in result.response
    assert "[DEBUG_AGENT]" in result.response
    assert "list_workspace_files" in result.response
    assert "read_workspace_file" in result.response


def test_compound_read_and_create_note_requires_confirmation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "exemplo.txt").write_text("conteudo importante", encoding="utf-8")

    agent = Agent(FakeLLM(), make_registry(), workspace, debug_agent=False)

    result = agent.run(
        "Le o ficheiro exemplo.txt e cria uma nota com os pontos principais.",
        make_context(),
    )

    assert "preciso da tua confirmacao" in result.response
    assert not (workspace / "nota_exemplo.txt").exists()

    confirmed = agent.run("sim", make_context())

    assert "Criei o ficheiro 'nota_exemplo.txt'" in confirmed.response
    assert (workspace / "nota_exemplo.txt").read_text(encoding="utf-8") == "Ponto principal do ficheiro."


def test_find_relevant_documents_uses_multiple_reads(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "documento_rvcc.txt").write_text("RVCC CP CLC STC", encoding="utf-8")
    (workspace / "compras.txt").write_text("lista de compras", encoding="utf-8")

    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace, debug_agent=True)

    result = agent.run("Procura documentos sobre RVCC e diz-me quais parecem relevantes.", make_context())

    assert "documento_rvcc.txt parece relevante." in result.response
    assert "list_workspace_files" in result.response
    assert "read_workspace_file" in result.response


def test_analyze_existing_files_suggests_organization(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nota.txt").write_text("nota", encoding="utf-8")

    agent = Agent(FakeLLM(), make_registry(), workspace)

    result = agent.run("Analisa os ficheiros existentes e sugere uma organizacao.", make_context())

    assert result.response == "Sugestao de organizacao."


def test_system_state_question_uses_context_observer_tool_not_llm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace, context_observer=FakeContextObserver())

    result = agent.run("Qual é a janela ativa?", make_context())

    assert "Janela ativa detetada: assistenteIA - Visual Studio Code" in result.response
    assert llm.chat_prompts == []


def test_system_state_question_without_context_observer_never_uses_llm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace)

    result = agent.run("Que aplicações estão abertas?", make_context())

    assert "ainda não tenho dados suficientes" in result.response
    assert llm.chat_prompts == []


def test_system_state_debug_shows_tool_and_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = Agent(
        FakeLLM(),
        make_registry(),
        workspace,
        context_observer=FakeContextObserver(),
        debug_agent=True,
    )

    result = agent.run("Que janelas tens detetadas?", make_context())

    assert "Janelas detetadas:" in result.response
    assert "[DEBUG_AGENT]" in result.response
    assert "Ferramenta escolhida: get_open_windows" in result.response
    assert "Resultado resumido:" in result.response


def test_presence_state_question_uses_presence_tool_not_llm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(
        llm,
        make_registry(),
        workspace,
        presence_manager=PresenceManager(PresenceState.FOCUS_MODE),
    )

    result = agent.run("Em que modo estás?", make_context())

    assert result.response.startswith("Estou em FOCUS_MODE.")
    assert llm.chat_prompts == []


def test_system_state_question_with_missing_tool_never_uses_llm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry_without_context_tools(), workspace, context_observer=FakeContextObserver())

    result = agent.run("Que janelas tens detetadas?", make_context())

    assert result.response == "A ferramenta de monitorização ainda não está ligada ao agente."
    assert llm.chat_prompts == []


def test_last_context_snapshot_tool_is_used_for_computer_state_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace, context_observer=FakeContextObserver())

    result = agent.run("Mostra o snapshot do estado do computador.", make_context())

    assert "VS Code aberto" in result.response or "Tenho apenas um sinal parcial" in result.response
    assert "Snapshot bruto:" in result.response
    assert "Janela ativa: assistenteIA - Visual Studio Code" in result.response
    assert llm.chat_prompts == []


def test_current_context_question_uses_interpreter_tool_not_llm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace, context_observer=FakeContextObserver())

    result = agent.run("O que estou a fazer?", make_context())

    assert "Contexto atual observado agora:" in result.response
    assert "Evidências observadas:" in result.response
    assert llm.chat_prompts == []


def test_current_activity_question_uses_reasoning_tool_not_llm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace, context_observer=FakeReasoningContextObserver())

    result = agent.run("Qual parece ser a minha atividade principal?", make_context())

    assert "Inferências prováveis:" in result.response
    assert "desenvolvimento do AssistenteIA" in result.response
    assert "Evidências observadas:" in result.response
    assert "assistant/agent.py" in result.response
    assert llm.chat_prompts == []


def test_context_interpreter_does_not_reduce_open_windows_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace, context_observer=FakeContextObserver())

    result = agent.run("Que janelas tens detetadas?", make_context())

    assert "Janelas detetadas:" in result.response
    assert "assistenteIA - Visual Studio Code [Code.exe] (ativa)" in result.response
    assert "Snapshot bruto:" not in result.response
    assert llm.chat_prompts == []


def test_get_open_windows_observes_live_windows_instead_of_stale_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(
        llm,
        make_registry(),
        workspace,
        context_observer=FakeStaleThenFreshContextObserver(),
    )

    result = agent.run("Que janelas tens detetadas?", make_context())

    assert "assistant/agent.py - assistenteIA - Visual Studio Code [Code.exe] (ativa)" in result.response
    assert "Codex - AssistenteIA [Codex.exe]" in result.response
    assert "AssistenteIA - Google Chrome [chrome.exe]" in result.response
    assert "AssistenteIA [python3.11.exe]" not in result.response
    assert llm.chat_prompts == []
