from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app import DEFAULT_OLLAMA_MODEL, resolve_ollama_model
from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.tool_registry import ToolRegistry


def test_cli_model_wins_over_everything() -> None:
    model, source = resolve_ollama_model(
        cli_model="cli-model",
        env={"ECHO_MODEL_NAME": "echo-model", "OLLAMA_MODEL": "legacy-model"},
        settings={"ollama": {"model": "settings-model"}},
    )

    assert model == "cli-model"
    assert source == "cli"


def test_echo_model_name_wins_over_legacy_and_settings() -> None:
    model, source = resolve_ollama_model(
        env={"ECHO_MODEL_NAME": "echo-model", "OLLAMA_MODEL": "legacy-model"},
        settings={"ollama": {"model": "settings-model"}},
    )

    assert model == "echo-model"
    assert source == "ECHO_MODEL_NAME"


def test_ollama_model_works_as_legacy_fallback() -> None:
    model, source = resolve_ollama_model(
        env={"OLLAMA_MODEL": "legacy-model"},
        settings={"ollama": {"model": "settings-model"}},
    )

    assert model == "legacy-model"
    assert source == "OLLAMA_MODEL"


def test_settings_model_is_used_without_env_or_cli() -> None:
    model, source = resolve_ollama_model(env={}, settings={"ollama": {"model": "settings-model"}})

    assert model == "settings-model"
    assert source == "settings.json"


def test_default_model_is_used_without_configuration() -> None:
    model, source = resolve_ollama_model(env={}, settings={})

    assert model == DEFAULT_OLLAMA_MODEL
    assert source == "default"


def test_empty_strings_are_ignored() -> None:
    model, source = resolve_ollama_model(
        cli_model=" ",
        env={"ECHO_MODEL_NAME": "", "OLLAMA_MODEL": "  "},
        settings={"ollama": {"model": "settings-model"}},
    )

    assert model == "settings-model"
    assert source == "settings.json"


class TelemetryLLM:
    system_prompt = ""

    def __init__(self) -> None:
        self.settings = SimpleNamespace(model="telemetry-model", model_source="settings.json")
        self.chat_call_count = 0
        self.chat_call_sources: list[str] = []

    def chat(self, *args, **kwargs):
        self.chat_call_count += 1
        return "resposta"

    def choose_tool(self, *args, **kwargs):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, *args, **kwargs):
        return None


def test_model_and_source_appear_in_turn_telemetry(tmp_path: Path) -> None:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = TelemetryLLM()
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt="",
        presence_manager=PresenceManager(),
        debug_ollama_payload=True,
    )

    engine._begin_turn_trace("teste")
    engine._complete_turn("teste", "resposta local", "FAST_ROUTE", technical=True, selected_path="FAST_ROUTE")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["model"] == "telemetry-model"
    assert telemetry["model_source"] == "settings.json"
