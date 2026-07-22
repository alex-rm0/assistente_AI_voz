from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.desktop_actions import DesktopActionResult
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry
from assistant.tools import get_open_windows, open_url


class PipelineLLM:
    def __init__(self, reply: str = "Isso parece estar a pesar-te. O que se passa?") -> None:
        self.reply = reply
        self.chat_calls = 0
        self.choose_tool_calls = 0
        self.embed_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        self.choose_tool_calls += 1
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        return self.reply

    def embed(self, text: str):
        self.embed_calls += 1
        return None


class PipelineMemory:
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


class PipelineRunner:
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
    registry.register(
        "get_open_windows",
        "Lista janelas abertas.",
        ("context:windows",),
        remember_result=False,
    )(get_open_windows)
    return registry


def make_engine(tmp_path: Path, llm: PipelineLLM | None = None):
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    runner = PipelineRunner()
    memory = PipelineMemory()
    engine = AssistantEngine(
        llm=llm or PipelineLLM(),
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory,
        tools=make_registry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        desktop_action_runner=runner,
    )
    return engine, engine.llm, memory, runner


def assert_no_capability_leak(response: str) -> None:
    lowered = response.lower()
    forbidden = ("ajudar", "aplicação", "aplicacoes", "ficheiro", "workspace", "ferramenta", "capacidades")
    assert not any(word in lowered for word in forbidden)
    assert "você" not in lowered
    assert "seu" not in lowered
    assert "sua" not in lowered


def assert_social_response_is_contained(response: str) -> None:
    lowered = response.lower()
    forbidden = (
        "sabe?",
        "voce",
        "voc\u00ea",
        "seus",
        "sua",
        "quais sao as datas",
        "quais s\u00e3o as datas",
        "atividade de remar",
        "a praia e um lugar perfeito",
        "a praia \u00e9 um lugar perfeito",
        "e bom saber que estas bem",
        "\u00e9 bom saber que est\u00e1s bem",
    )
    assert not any(phrase in lowered for phrase in forbidden)
    assert response.count("?") <= 1
    assert sum(response.count(mark) for mark in ".!?") <= 2


def test_viva_como_estas_uses_social_fast_path_without_agent_or_context(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    response = engine.respond("Viva, como estás?")

    assert response == "Estou bem. E tu?"
    assert_no_capability_leak(response)
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0
    assert memory.context_for_calls == 0
    assert runner.opened_urls == []
    assert engine.last_cognitive_strategy is None
    assert not engine.agent.has_pending_confirmation()


def test_ola_is_short_social_fast_path(tmp_path: Path) -> None:
    engine, llm, memory, _runner = make_engine(tmp_path)

    response = engine.respond("Olá")

    assert response == "Olá! Como estás?"
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0
    assert memory.context_for_calls == 0


def test_viva_tudo_bem_contigo_uses_natural_social_reply(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    response = engine.respond("Viva, tudo bem contigo?")

    assert response == "Tudo bem. E contigo?"
    assert_social_response_is_contained(response)
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0
    assert memory.context_for_calls == 0
    assert runner.opened_urls == []


def test_tambem_estou_bem_does_not_start_interview(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    response = engine.respond("Tamb\u00e9m estou bem.")

    assert response == "Ainda bem."
    assert_social_response_is_contained(response)
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0
    assert memory.context_for_calls == 0
    assert runner.opened_urls == []


def test_casual_weekend_share_does_not_activate_planning(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    response = engine.respond("No fim de semana penso ir \u00e0 praia com uns amigos e remar.")

    assert response == "Parece um bom fim de semana. Praia, amigos e ir remar combinam bem."
    assert_social_response_is_contained(response)
    assert "plano" not in response.lower()
    assert "organizar" not in response.lower()
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0
    assert memory.context_for_calls == 0
    assert runner.opened_urls == []
    assert engine.last_cognitive_strategy is None


def test_real_casual_share_gets_short_specific_response(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    response = engine.respond(
        "Foi pac\u00edfica por acaso, correu tudo bem. "
        "Estive a trabalhar em ti, neste projeto, no Echo. "
        "No fim de semana penso que vou \u00e0 praia com uns amigos e remar."
    )

    assert response == (
        "Então tiveste uma semana tranquila, apesar de a passares parcialmente a corrigir-me. "
        "E esse fim de semana parece bem escolhido."
    )
    assert_social_response_is_contained(response)
    assert "quais" not in response.lower()
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0
    assert memory.context_for_calls == 0
    assert runner.opened_urls == []
    assert engine.last_cognitive_strategy is None


def test_greeting_with_emotional_content_is_not_consumed_as_pure_social(tmp_path: Path) -> None:
    engine, llm, _memory, runner = make_engine(tmp_path)

    response = engine.respond("Olá, estou preocupado com uma coisa.")

    assert response == "Isso parece estar a pesar-te. O que se passa?"
    assert llm.chat_calls >= 1
    assert llm.choose_tool_calls == 0
    assert runner.opened_urls == []
    assert not engine.agent.has_pending_confirmation()


def test_document_help_does_not_announce_tools_or_open_files(tmp_path: Path) -> None:
    llm = PipelineLLM("Claro. O que se passa com o documento?")
    engine, llm, _memory, runner = make_engine(tmp_path, llm)

    response = engine.respond("Preciso de ajuda com um documento.")

    assert response == "Claro. O que se passa com o documento?"
    assert "ficheiro" not in response.lower()
    assert "workspace" not in response.lower()
    assert llm.choose_tool_calls == 0
    assert runner.opened_urls == []


def test_open_google_still_uses_tool_confirmation(tmp_path: Path) -> None:
    engine, llm, memory, runner = make_engine(tmp_path)

    response = engine.respond("Abre o Google.")

    assert "Queres que abra este URL? https://www.google.com" in response
    assert engine.agent.has_pending_confirmation()
    assert llm.chat_calls == 0
    assert llm.choose_tool_calls == 0
    assert memory.context_for_calls == 0
    assert runner.opened_urls == []
