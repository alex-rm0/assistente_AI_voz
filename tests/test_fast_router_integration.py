from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.desktop_actions import DesktopActionResult
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry
from assistant.tools import open_url


class SpyLLM:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.choose_tool_calls = 0
        self.embed_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        self.choose_tool_calls += 1
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        return "resposta do llm"

    def embed(self, text: str):
        self.embed_calls += 1
        return None


class SpyLongTermMemory:
    def __init__(self) -> None:
        self.preferences: dict[str, str] = {}
        self.context_for_calls = 0
        self.remember_calls: list[tuple[str, str | None]] = []

    def get_preference(self, key: str, default: str = "") -> str:
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: str) -> None:
        self.preferences[key] = value

    def context_for(self, query: str, limit: int = 5) -> str:
        self.context_for_calls += 1
        return ""

    def pending_tasks(self, *args, **kwargs) -> str:
        return ""

    def remember(self, content: str, category: str | None = None) -> str:
        self.remember_calls.append((content, category))
        return "guardado"


class SpyDesktopRunner:
    def __init__(self) -> None:
        self.opened_urls: list[str] = []

    def open_application(self, app_key, executable, uri=None):
        raise AssertionError("Nao devia abrir aplicacoes neste teste.")

    def open_path(self, path: Path):
        raise AssertionError("Nao devia abrir caminhos neste teste.")

    def open_project(self, editor_executable, project_path: Path):
        raise AssertionError("Nao devia abrir projetos neste teste.")

    def open_url(self, url: str):
        self.opened_urls.append(url)
        return DesktopActionResult(True, f"Abri o URL: {url}")


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "open_url",
        "Abre URLs permitidos.",
        ("desktop:open_url", "confirmation-required"),
        remember_result=False,
    )(open_url)
    return registry


def make_engine(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    data = tmp_path / "data"
    llm = SpyLLM()
    long_term_memory = SpyLongTermMemory()
    runner = SpyDesktopRunner()
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=long_term_memory,
        tools=make_registry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        desktop_action_runner=runner,
    )
    return engine, llm, long_term_memory, runner


def assert_no_model_or_semantic_memory_calls(llm: SpyLLM, memory: SpyLongTermMemory) -> None:
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0
    assert llm.embed_calls == 0
    assert memory.context_for_calls == 0


def test_fast_open_youtube_creates_pending_url_without_llm_or_embeddings(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    response = engine.respond("abre o youtube")

    assert "Queres que abra este URL? https://www.youtube.com" in response
    assert engine.agent.has_pending_confirmation()
    assert runner.opened_urls == []
    assert_no_model_or_semantic_memory_calls(llm, memory)


def test_fast_open_bare_domain_normalizes_to_https(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    response = engine.respond("abre www.youtube.com")

    assert "https://www.youtube.com" in response
    assert engine.agent.has_pending_confirmation()
    assert runner.opened_urls == []
    assert_no_model_or_semantic_memory_calls(llm, memory)


def test_fast_safe_url_examples_create_pending_actions_without_llm(tmp_path: Path) -> None:
    examples = (
        ("abre youtube", "https://www.youtube.com"),
        ("abre o google", "https://www.google.com"),
        ("abre gmail", "https://mail.google.com"),
        ("abre o gmail", "https://mail.google.com"),
        ("abre chatgpt", "https://chatgpt.com"),
        ("abre github", "https://github.com"),
        ("abre www.youtube.com", "https://www.youtube.com"),
        ("abre https://www.google.com", "https://www.google.com"),
    )

    for index, (message, url) in enumerate(examples):
        engine, llm, memory, runner = make_engine(tmp_path / f"safe_{index}")

        response = engine.respond(message)

        if url == "https://mail.google.com":
            assert "Queres que abra o Gmail?" in response
        else:
            assert f"Queres que abra este URL? {url}" in response
        assert engine.agent.has_pending_confirmation()
        assert runner.opened_urls == []
        assert_no_model_or_semantic_memory_calls(llm, memory)


def test_confirming_each_configured_site_executes_expected_url_without_llm(tmp_path: Path) -> None:
    examples = (
        ("abre youtube", "https://www.youtube.com"),
        ("abre google", "https://www.google.com"),
        ("abre gmail", "https://mail.google.com"),
        ("abre chatgpt", "https://chatgpt.com"),
        ("abre github", "https://github.com"),
    )

    for index, (message, url) in enumerate(examples):
        engine, llm, memory, runner = make_engine(tmp_path / f"confirm_{index}")

        engine.respond(message)
        llm.chat_calls = 0
        llm.choose_tool_calls = 0
        llm.embed_calls = 0
        memory.context_for_calls = 0
        response = engine.respond("sim")

        assert response == f"Abri o URL: {url}"
        assert runner.opened_urls == [url]
        assert not engine.agent.has_pending_confirmation()
        assert_no_model_or_semantic_memory_calls(llm, memory)


def test_confirming_pending_fast_url_executes_without_llm(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    engine.respond("abre o youtube")
    llm.chat_calls = 0
    llm.choose_tool_calls = 0
    llm.embed_calls = 0
    memory.context_for_calls = 0
    response = engine.respond("sim")

    assert response == "Abri o URL: https://www.youtube.com"
    assert runner.opened_urls == ["https://www.youtube.com"]
    assert not engine.agent.has_pending_confirmation()
    assert_no_model_or_semantic_memory_calls(llm, memory)


def test_canceling_pending_fast_url_does_not_execute_or_call_llm(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    engine.respond("abre o youtube")
    llm.chat_calls = 0
    llm.choose_tool_calls = 0
    llm.embed_calls = 0
    memory.context_for_calls = 0
    response = engine.respond("nao")

    assert "cancelada" in response.lower()
    assert runner.opened_urls == []
    assert not engine.agent.has_pending_confirmation()
    assert_no_model_or_semantic_memory_calls(llm, memory)


def test_fast_search_examples_create_pending_actions_without_llm_or_runner(tmp_path: Path) -> None:
    examples = (
        (
            "pesquisa no google por gatos",
            "Google",
            "gatos",
            "https://www.google.com/search?q=gatos",
        ),
        (
            "procura no google por temperaturas históricas em Coimbra",
            "Google",
            "temperaturas históricas em Coimbra",
            "https://www.google.com/search?q=temperaturas+hist%C3%B3ricas+em+Coimbra",
        ),
        (
            "pesquisar no google comandos powershell básicos",
            "Google",
            "comandos powershell básicos",
            "https://www.google.com/search?q=comandos+powershell+b%C3%A1sicos",
        ),
        (
            "pesquisa no youtube por tutorial python",
            "YouTube",
            "tutorial python",
            "https://www.youtube.com/results?search_query=tutorial+python",
        ),
        (
            "procura no youtube por música lo-fi",
            "YouTube",
            "música lo-fi",
            "https://www.youtube.com/results?search_query=m%C3%BAsica+lo-fi",
        ),
        (
            "pesquisa no google por file:///C:/Windows/System32/cmd.exe",
            "Google",
            "file:///C:/Windows/System32/cmd.exe",
            "https://www.google.com/search?q=file%3A%2F%2F%2FC%3A%2FWindows%2FSystem32%2Fcmd.exe",
        ),
    )

    for index, (message, engine_name, query, url) in enumerate(examples):
        engine, llm, memory, runner = make_engine(tmp_path / f"search_{index}")

        response = engine.respond(message)

        assert f"Queres que pesquise no {engine_name} por: {query}?" in response
        assert engine.agent.has_pending_confirmation()
        assert runner.opened_urls == []
        pending = engine.agent.pending_confirmation
        assert pending is not None
        assert pending["arguments"]["url"] == url
        assert_no_model_or_semantic_memory_calls(llm, memory)


def test_confirming_pending_fast_search_executes_without_llm(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    engine.respond("pesquisa no youtube por tutorial python tkinter")
    llm.chat_calls = 0
    llm.choose_tool_calls = 0
    llm.embed_calls = 0
    memory.context_for_calls = 0
    response = engine.respond("sim")

    expected_url = "https://www.youtube.com/results?search_query=tutorial+python+tkinter"
    assert response == f"Abri o URL: {expected_url}"
    assert runner.opened_urls == [expected_url]
    assert not engine.agent.has_pending_confirmation()
    assert_no_model_or_semantic_memory_calls(llm, memory)


def test_canceling_pending_fast_search_does_not_execute_or_call_llm(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    engine.respond("pesquisa no google por gatos")
    llm.chat_calls = 0
    llm.choose_tool_calls = 0
    llm.embed_calls = 0
    memory.context_for_calls = 0
    response = engine.respond("nao")

    assert "cancelada" in response.lower()
    assert runner.opened_urls == []
    assert not engine.agent.has_pending_confirmation()
    assert_no_model_or_semantic_memory_calls(llm, memory)


def test_dangerous_requests_are_blocked_without_execution_or_llm(tmp_path: Path) -> None:
    messages = (
        "executa dir",
        "corre powershell",
        "abre file:///C:/Windows/System32/cmd.exe",
        "abre javascript:alert(1)",
        "abre data:text/html,<script>alert(1)</script>",
        "abre C:\\Windows\\System32",
        "abre powershell",
        "abre cmd",
        "abre terminal",
        "abre programa.exe",
        "abre script.ps1",
        "abre ../ficheiro",
    )

    for index, message in enumerate(messages):
        engine, llm, memory, runner = make_engine(tmp_path / f"blocked_{index}")

        response = engine.respond(message)

        assert "Não posso" in response
        assert runner.opened_urls == []
        assert not engine.agent.has_pending_confirmation()
        assert_no_model_or_semantic_memory_calls(llm, memory)


def test_complex_request_still_reaches_llm_path(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    response = engine.respond("Ajuda-me a pensar numa arquitetura melhor para este assistente.")

    assert response == "resposta do llm"
    assert runner.opened_urls == []
    assert memory.context_for_calls == 1
    assert llm.choose_tool_calls == 1
    assert llm.chat_calls == 1
