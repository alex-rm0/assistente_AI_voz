from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.desktop_actions import DesktopActionResult
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry
from assistant.tools import open_url


class ToolChoosingLLM:
    def __init__(self, decision: dict | None = None) -> None:
        self.decision = decision or {"tool": None, "arguments": {}, "reason": "sem ferramenta"}
        self.chat_calls = 0
        self.choose_tool_calls = 0
        self.embed_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        self.choose_tool_calls += 1
        return dict(self.decision)

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        return "resposta do llm"

    def embed(self, text: str):
        self.embed_calls += 1
        return None


class MinimalMemory:
    def __init__(self) -> None:
        self.preferences: dict[str, str] = {}
        self.context_for_calls = 0

    def get_preference(self, key: str, default: str = "") -> str:
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: str) -> None:
        self.preferences[key] = value

    def context_for(self, query: str, limit: int = 5) -> str:
        self.context_for_calls += 1
        return ""

    def pending_tasks(self, *args, **kwargs) -> str:
        return ""


class SpyDesktopRunner:
    def __init__(self) -> None:
        self.opened_urls: list[str] = []

    def open_application(self, app_key, executable, uri=None):
        raise AssertionError("Não devia abrir aplicações neste teste.")

    def open_path(self, path: Path):
        raise AssertionError("Não devia abrir caminhos neste teste.")

    def open_project(self, editor_executable, project_path: Path):
        raise AssertionError("Não devia abrir projetos neste teste.")

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


def make_engine(tmp_path: Path, llm: ToolChoosingLLM | None = None):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    data = tmp_path / "data"
    runner = SpyDesktopRunner()
    memory = MinimalMemory()
    engine = AssistantEngine(
        llm=llm or ToolChoosingLLM(),
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory,
        tools=make_registry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        desktop_action_runner=runner,
    )
    return engine, engine.llm, memory, runner


def test_full_sentence_refusal_cancels_pending_action_without_llm(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    engine.respond("abre o google")
    llm.chat_calls = llm.choose_tool_calls = llm.embed_calls = 0
    memory.context_for_calls = 0
    response = engine.respond("Não preciso que abras nada no Google, quero a tua ajuda apenas.")

    assert response == "Claro. Não abro nada. Continuamos por aqui."
    assert not engine.agent.has_pending_confirmation()
    assert runner.opened_urls == []
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0
    assert llm.embed_calls == 0
    assert memory.context_for_calls == 0


def test_pending_action_accepts_cancel_variants(tmp_path: Path) -> None:
    variants = (
        "Não, quero só falar contigo.",
        "Não abras nada.",
        "Esquece isso.",
        "Prefiro que me ajudes aqui.",
        "Não é preciso pesquisar.",
        "Cancela e volta ao que estávamos a falar.",
        "Não quero que abras o browser.",
        "Não, isso não foi o que pedi.",
        "Não.",
    )

    for index, message in enumerate(variants):
        engine, llm, memory, runner = make_engine(tmp_path / f"cancel_{index}")
        engine.respond("abre o google")
        llm.chat_calls = llm.choose_tool_calls = llm.embed_calls = 0
        memory.context_for_calls = 0

        response = engine.respond(message)

        assert "ação pendente" not in response.lower()
        assert "responde 'sim'" not in response.lower()
        assert not engine.agent.has_pending_confirmation()
        assert runner.opened_urls == []
        assert llm.chat_calls == 0
        assert llm.choose_tool_calls == 0
        assert llm.embed_calls == 0
        assert memory.context_for_calls == 0


def test_sports_season_statement_does_not_create_url_action_from_llm_choice(tmp_path: Path) -> None:
    llm = ToolChoosingLLM(
        {
            "tool": "open_url",
            "arguments": {"url": "https://www.example.com/preparacao-proxima-epoca-desportiva"},
            "reason": "associação temática indevida",
        }
    )
    engine, llm, _memory, runner = make_engine(tmp_path, llm=llm)

    response = engine.respond("Acabou agora a época desportiva, mas devia começar a preparar a próxima.")

    assert response == "resposta do llm"
    assert not engine.agent.has_pending_confirmation()
    assert runner.opened_urls == []
    assert llm.choose_tool_calls == 0
    assert llm.chat_calls == 1
