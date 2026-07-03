from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from assistant.agent import Agent, AgentContext
from assistant.desktop_actions import DesktopActionResult, normalize_app_name, validate_url
from assistant.prompts import get_system_prompt
from assistant.tool_registry import ToolRegistry
from assistant.tools import open_application, open_file, open_folder, open_project, open_url


class FakeLLM:
    def choose_tool(self, user_message, tools_description, profile_name=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta direta"


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def open_application(self, app_key, executable, uri=None):
        self.calls.append(("application", app_key))
        return DesktopActionResult(True, f"Abri {app_key}.")

    def open_path(self, path: Path):
        self.calls.append(("path", str(path)))
        return DesktopActionResult(True, f"Abri '{path.name}'.")

    def open_url(self, url):
        self.calls.append(("url", url))
        return DesktopActionResult(True, f"Abri o URL: {url}")

    def open_project(self, editor_executable, project_path: Path):
        self.calls.append(("project", str(project_path)))
        return DesktopActionResult(True, f"Abri o projeto {project_path.name}.")


class FakeOpenAppObserver:
    def observe_once(self):
        return SimpleNamespace(
            active_app="Code.exe",
            active_window="assistenteIA - Visual Studio Code",
            open_windows=(
                SimpleNamespace(
                    title="assistenteIA - Visual Studio Code",
                    process_name="Code.exe",
                    is_active=True,
                ),
            ),
        )


def make_desktop_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "open_application",
        "Abre aplicacoes permitidas.",
        ("desktop:open_application", "confirmation-required"),
        remember_result=False,
    )(open_application)
    registry.register(
        "open_file",
        "Abre ficheiros permitidos.",
        ("desktop:open_file", "confirmation-required"),
        remember_result=False,
    )(open_file)
    registry.register(
        "open_folder",
        "Abre pastas permitidas.",
        ("desktop:open_folder", "confirmation-required"),
        remember_result=False,
    )(open_folder)
    registry.register(
        "open_url",
        "Abre URLs permitidos.",
        ("desktop:open_url", "confirmation-required"),
        remember_result=False,
    )(open_url)
    registry.register(
        "open_project",
        "Abre projetos conhecidos.",
        ("desktop:open_project", "confirmation-required"),
        remember_result=False,
    )(open_project)
    return registry


def make_context() -> AgentContext:
    return AgentContext(
        profile_name="Geral",
        system_prompt=get_system_prompt("Geral"),
        history=[],
    )


def test_application_aliases_are_normalized() -> None:
    assert normalize_app_name("mail") == "outlook"
    assert normalize_app_name("codigo") == "vscode"
    assert normalize_app_name("browser") == "chrome"


def test_validate_url_allows_only_http_and_https() -> None:
    assert validate_url("https://example.com") == "https://example.com"
    assert validate_url("http://example.com") == "http://example.com"
    assert validate_url("file:///C:/Windows/System32/cmd.exe") == ""


def test_open_file_blocks_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside = outside_root / "fora.txt"
    outside.write_text("fora", encoding="utf-8")

    project_root = tmp_path / "project"
    project_root.mkdir()
    result = open_file(str(outside), workspace_path=workspace, project_root=project_root, desktop_action_runner=FakeRunner())

    assert "Nao posso abrir esse ficheiro" in result


def test_open_file_inside_workspace_uses_runner(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "nota.txt"
    target.write_text("nota", encoding="utf-8")
    runner = FakeRunner()

    result = open_file("nota.txt", workspace_path=workspace, project_root=workspace.parent, desktop_action_runner=runner)

    assert result == "Abri 'nota.txt'."
    assert runner.calls == [("path", str(target))]


def test_open_folder_inside_project_uses_runner(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    docs = project / "docs"
    docs.mkdir()
    runner = FakeRunner()

    result = open_folder("docs", workspace_path=workspace, project_root=project, desktop_action_runner=runner)

    assert result == "Abri 'docs'."
    assert runner.calls == [("path", str(docs))]


def test_agent_asks_confirmation_before_opening_application(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = FakeRunner()
    agent = Agent(
        FakeLLM(),
        make_desktop_registry(),
        workspace,
        desktop_action_runner=runner,
    )

    first = agent.run("abre o mail", make_context())

    assert "Responde 'sim'" in first.response
    assert runner.calls == []

    second = agent.run("sim", make_context())

    assert "Abri outlook." in second.response
    assert runner.calls == [("application", "outlook")]


def test_agent_does_not_ask_confirmation_when_application_is_already_open(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = FakeRunner()
    agent = Agent(
        FakeLLM(),
        make_desktop_registry(),
        workspace,
        context_observer=FakeOpenAppObserver(),
        desktop_action_runner=runner,
    )

    result = agent.run("abre o codigo", make_context())

    assert result.response == "O VS Code ja esta aberto."
    assert runner.calls == []


def test_agent_can_open_known_project_after_confirmation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = tmp_path / "assistenteIA"
    project.mkdir()
    runner = FakeRunner()
    agent = Agent(
        FakeLLM(),
        make_desktop_registry(),
        workspace,
        known_projects={"AssistenteIA": str(project)},
        desktop_action_runner=runner,
    )

    first = agent.run("abre o projeto AssistenteIA", make_context())
    second = agent.run("sim", make_context())

    assert "Queres que abra o projeto" in first.response
    assert "Abri o projeto assistenteIA." in second.response
    assert runner.calls == [("project", str(project))]
