from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.memory import ConversationMemory
from assistant.model_provider import ModelResponse
from assistant.model_router import AutomaticRoutingConfig, ModelRouter, ModelRoutingConfig, ModelUsageBudget, RoutedLLM
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeProvider:
    def __init__(self, name: str, model: str, replies: list[str] | None = None) -> None:
        self._name = name
        self.model = model
        self.replies = replies or ["Resposta."]
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages, *, model=None, response_format=None, tools=None, temperature=None):
        self.calls.append({"messages": messages, "model": model, "response_format": response_format})
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return ModelResponse(
            text=self.replies[index],
            provider=self._name,
            model=model or self.model,
            input_tokens=120 + index,
            output_tokens=40 + index,
            latency_ms=1.0,
            estimated_cost_usd=0.002 if self._name == "anthropic" else 0.0,
        )


class MemoryStub:
    def __init__(self) -> None:
        self.preferences: dict[str, str] = {}

    def get_preference(self, key: str, default: str = "") -> str:
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: str) -> None:
        self.preferences[key] = value

    def context_for(self, query: str, limit: int = 5) -> str:
        return ""

    def pending_tasks(self, *args, **kwargs) -> str:
        return ""


def make_engine(tmp_path: Path, *, mode: str = "automatic", claude_enabled: bool = True, env: dict[str, str] | None = None, ollama_replies=None, anthropic_replies=None):
    ollama = FakeProvider("ollama", "llama3.1:8b", ollama_replies)
    anthropic = FakeProvider("anthropic", "claude-haiku-4-5-20251001", anthropic_replies)
    router = ModelRouter(
        ModelRoutingConfig(
            mode=mode,
            mode_source="test",
            automatic=AutomaticRoutingConfig(claude_enabled=claude_enabled, daily_budget_usd=0.25, max_single_call_estimated_usd=0.05),
        ),
        ollama_model=ollama.model,
        anthropic_model=anthropic.model,
        budget=ModelUsageBudget(tmp_path / "usage.json"),
        env=env or {},
    )
    llm = RoutedLLM(
        providers={"ollama": ollama, "anthropic": anthropic},
        router=router,
        system_prompt=get_base_system_prompt(),
        model_source="test",
    )
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=MemoryStub(),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        debug_ollama_payload=True,
    )
    return engine, llm, ollama, anthropic


EMAIL_REQUEST = (
    "Escreve um email profissional detalhado a explicar o estado atual do projeto Echo, "
    "os progressos realizados e os próximos passos."
)


def test_professional_email_request_is_not_memory_command_and_uses_anthropic_when_authorized(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        env={"ANTHROPIC_API_KEY": "secret", "ECHO_ALLOW_PAID_MODEL_CALLS": "true"},
        anthropic_replies=["Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação do projeto Echo."],
    )

    response = engine.respond(EMAIL_REQUEST)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "MEMORY_COMMAND"
    assert telemetry["response_source"] != "MEMORY_COMMAND"
    assert "ACTIVE_CONVERSATION" not in response
    assert "Assunto:" in response
    assert telemetry["model_routing_provider"] == "anthropic"
    assert telemetry["model_routing_reason_code"] == "professional_writing"
    assert len(anthropic.calls) == 1
    assert ollama.calls == []


def test_professional_email_request_uses_ollama_when_automatic_claude_disabled(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        claude_enabled=False,
        ollama_replies=["Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação."],
    )

    response = engine.respond(EMAIL_REQUEST)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Assunto:" in response
    assert telemetry["model_routing_provider"] == "ollama"
    assert telemetry["model_routing_reason_code"] == "automatic_claude_disabled"
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


def test_real_presence_state_question_still_works(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(tmp_path, claude_enabled=False)

    response = engine.respond("Em que modo estás?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "ACTIVE_CONVERSATION" in response
    assert telemetry["selected_path"] == "MEMORY_COMMAND"
    assert llm.chat_call_count == 0


def test_complete_writing_request_regenerates_instead_of_offering_help(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=[
            "Posso ajudar-te a redigir o email?",
            "Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação.",
        ],
    )

    response = engine.respond(EMAIL_REQUEST)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Posso ajudar" not in response
    assert "Queres que" not in response
    assert "Assunto:" in response
    assert telemetry["llm_calls"] == 2
    assert [item["component"] for item in telemetry["llm_call_details"]] == [
        "RESPONSE_COMPOSER",
        "RESPONSE_COMPOSER_REGENERATION",
    ]
    assert all(item["provider"] == "ollama" for item in telemetry["llm_call_details"])


def test_pending_intent_confirmation_executes_original_email_request(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação."],
    )
    engine._pending_user_intent = {"kind": "professional_writing", "message": EMAIL_REQUEST, "ttl": "3"}

    response = engine.respond("sim")

    assert "Assunto:" in response
    assert "Posso ajudar" not in response
    assert engine._pending_user_intent is None


def test_pending_intent_subject_followup_recovers_email_context(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação."],
    )
    engine._pending_user_intent = {"kind": "professional_writing", "message": EMAIL_REQUEST, "ttl": "3"}

    response = engine.respond("do email")

    assert "Assunto:" in response
    assert "Echo" in response
    assert engine._pending_user_intent is None
