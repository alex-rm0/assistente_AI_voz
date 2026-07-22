from __future__ import annotations

from pathlib import Path

from assistant.agent import Agent, AgentContext
from assistant.desktop_actions import DesktopActionResult
from assistant.planner import plan_user_request
from assistant.prompts import get_system_prompt
from assistant.tool_registry import ToolRegistry
from assistant.tools import open_project


class FakeLLM:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.choose_tool_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None):
        self.choose_tool_calls += 1
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        return "resposta do llm"


class FakeRunner:
    def __init__(self) -> None:
        self.opened_projects: list[str] = []

    def open_application(self, app_key, executable, uri=None):
        raise AssertionError("Nao devia abrir aplicacoes neste teste.")

    def open_path(self, path: Path):
        raise AssertionError("Nao devia abrir caminhos neste teste.")

    def open_url(self, url):
        raise AssertionError("Nao devia abrir URLs neste teste.")

    def open_project(self, editor_executable, project_path: Path):
        self.opened_projects.append(str(project_path))
        return DesktopActionResult(True, f"Abri o projeto {project_path.name}.")


def make_context() -> AgentContext:
    return AgentContext(
        profile_name="Geral",
        system_prompt=get_system_prompt("Geral"),
        history=[],
        context_debug="VS Code, Codex e Chrome ja estao abertos.",
        pending_tasks="- rever Planner",
        recurring_context="Projeto AssistenteIA em desenvolvimento.",
    )


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "open_project",
        "Abre projetos conhecidos.",
        ("desktop:open_project", "confirmation-required"),
        remember_result=False,
    )(open_project)
    return registry


def test_planner_detects_resume_project() -> None:
    result = plan_user_request("Vamos continuar o projeto AssistenteIA")

    assert result.intent == "retomar projeto"
    assert result.related_project == "AssistenteIA"
    assert "rever onde ficamos" in result.suggested_plan[0]
    assert not result.needs_confirmation


def test_planner_detects_travel_planning() -> None:
    result = plan_user_request("Planeia comigo umas ferias")

    assert result.intent == "planeamento pessoal"
    assert result.delegate_to == "pesquisa web opcional"
    assert "datas" in " ".join(result.suggested_plan)


def test_planner_detects_workspace_environment_action() -> None:
    result = plan_user_request("Abre o ambiente de trabalho do AssistenteIA")

    assert result.intent == "preparar ambiente de trabalho"
    assert result.related_project == "AssistenteIA"
    assert result.needs_confirmation
    assert result.recommended_actions[0].tool_name == "open_project"


def test_planner_detects_task_management() -> None:
    result = plan_user_request("Que tarefas tenho pendentes?")

    assert result.intent == "gerir tarefas"
    assert "identificar" in result.suggested_plan[0]


def test_planner_detects_normal_conversation() -> None:
    result = plan_user_request("Ola, tudo bem?")

    assert result.intent == "conversa normal"
    assert result.is_normal_conversation


def test_agent_uses_planner_for_resume_project_without_llm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace)

    result = agent.run("Vamos continuar o projeto AssistenteIA", make_context())

    assert "retomar o projeto AssistenteIA" in result.response
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0


def test_agent_uses_planner_for_travel_without_llm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace)

    result = agent.run("Planeia comigo umas ferias", make_context())

    assert "destino, datas" in result.response
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0


def test_agent_asks_confirmation_for_workspace_environment(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = tmp_path / "assistenteIA"
    project.mkdir()
    runner = FakeRunner()
    agent = Agent(
        FakeLLM(),
        make_registry(),
        workspace,
        known_projects={"assistenteIA": str(project)},
        desktop_action_runner=runner,
    )

    result = agent.run("Abre o ambiente de trabalho do AssistenteIA", make_context())

    assert "preciso da tua confirmacao" in result.response or "Responde 'sim'" in result.response
    assert agent.has_pending_confirmation()
    assert runner.opened_projects == []


def test_agent_normal_conversation_still_reaches_llm(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    agent = Agent(llm, make_registry(), workspace)

    result = agent.run("Ola, tudo bem?", make_context())

    assert result.response == "resposta do llm"
    assert llm.choose_tool_calls == 1
    assert llm.chat_calls == 1
