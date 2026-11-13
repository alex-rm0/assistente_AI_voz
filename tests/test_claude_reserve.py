"""Decision of 12/11: general conversation goes to Claude; the local model is
the reserve. A slow, failing or unaffordable Claude never fails the turn: the
local model answers, the reason is recorded, health reports "degraded", and
the UI label names the model that actually answered. No network: fake providers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.model_provider import ModelResponse, ProviderConfigurationError, ProviderTimeoutError
from assistant.model_router import (
    CLAUDE_CONVERSATION_TIMEOUT_SECONDS,
    CLOUD_RETRY_AFTER_SECONDS,
    AutomaticRoutingConfig,
    ModelRouter,
    ModelRoutingConfig,
    ModelUsageBudget,
    RoutedLLM,
)
from assistant.model_runtime import ModelRuntimeBridge, UserSettingsStore
from assistant.presence_manager import PresenceManager
from assistant.privacy import PERSONAL, PRIVATE
from assistant.prompts import get_base_system_prompt
from assistant.system_health import DEGRADED, HEALTHY, UNKNOWN, SystemHealth, default_probes, wellbeing_reply
from assistant.tool_registry import ToolRegistry


class FakeProvider:
    def __init__(self, name: str, model: str) -> None:
        self._name = name
        self.model = model
        self.error: Exception | None = None
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages, *, model=None, timeout_seconds=None, **kwargs) -> ModelResponse:
        self.calls.append({"timeout_seconds": timeout_seconds})
        if self.error is not None:
            raise self.error
        cost = 0.001 if self._name == "anthropic" else 0.0
        return ModelResponse(text=f"Reply from {self._name}.", provider=self._name, model=model or self.model,
                             input_tokens=10, output_tokens=5, latency_ms=1.0, estimated_cost_usd=cost)

    def embed(self, text):
        return None


def make_llm(*, mode: str = "automatic", budget: ModelUsageBudget | None = None, daily_budget_usd: float = 0.25):
    ollama, claude = FakeProvider("ollama", "llama3.1:8b"), FakeProvider("anthropic", "claude-haiku-4-5-20251001")
    router = ModelRouter(
        ModelRoutingConfig(mode=mode, mode_source="test", automatic=AutomaticRoutingConfig(claude_enabled=True, daily_budget_usd=daily_budget_usd)),
        ollama_model=ollama.model, anthropic_model=claude.model, budget=budget, env={},
        anthropic_key_available=lambda: True, paid_calls_allowed=lambda: True,
    )
    llm = RoutedLLM(providers={"ollama": ollama, "anthropic": claude}, router=router, system_prompt="system")
    return llm, ollama, claude


def talk(llm: RoutedLLM, **kwargs) -> str:
    llm.mark_next_call_privacy(PERSONAL)
    return llm.chat("Hoje estou cansado.", source="RESPONSE_COMPOSER", **kwargs)


def test_conversation_goes_to_claude_with_a_short_deadline() -> None:
    llm, ollama, claude = make_llm()

    assert talk(llm) == "Reply from anthropic."
    assert ollama.calls == []
    assert claude.calls == [{"timeout_seconds": CLAUDE_CONVERSATION_TIMEOUT_SECONDS}]
    assert CLAUDE_CONVERSATION_TIMEOUT_SECONDS == 4.0
    assert llm.last_routing_decision.reason_code == "general_conversation"
    assert llm.cloud_state() == HEALTHY


@pytest.mark.parametrize("error,reason", [
    (ProviderTimeoutError("slow", provider="anthropic"), "claude_timeout"),
    (RuntimeError("Nao consegui ligar a Anthropic."), "claude_unreachable"),
    (RuntimeError("A Anthropic devolveu um erro HTTP 529."), "claude_error"),
    (ProviderConfigurationError("no key", provider="anthropic", provider_error_type="missing_api_key"), "claude_missing_api_key"),
])
def test_a_failing_claude_is_answered_by_the_local_model_with_the_reason(error, reason, capsys) -> None:
    llm, ollama, claude = make_llm()
    claude.error = error

    assert talk(llm) == "Reply from ollama."

    decision = llm.last_routing_decision
    assert (decision.provider, decision.model, decision.paid_call) == ("ollama", "llama3.1:8b", False)
    assert decision.reason_code == reason and decision.fallback_reason == reason
    assert f"[MODEL_ROUTING] reserve=local reason={reason}" in capsys.readouterr().out
    assert llm.cloud_state() == DEGRADED and llm.last_cloud_failure_reason == reason
    # What the UI label and the telemetry read: the model that answered.
    settings = llm.settings
    assert (settings.model_routing_provider, settings.model_routing_model) == ("ollama", "llama3.1:8b")
    assert settings.model_routing_paid_call is False and settings.model_routing_fallback_reason == reason


def test_after_a_failure_claude_is_left_alone_for_a_while_then_tried_again() -> None:
    llm, ollama, claude = make_llm()
    now = [1000.0]
    llm._clock = lambda: now[0]
    claude.error = ProviderTimeoutError("slow", provider="anthropic")

    talk(llm)
    talk(llm)
    assert len(claude.calls) == 1  # the second turn did not wait for another timeout
    assert llm.last_routing_decision.reason_code == "claude_cooldown"

    now[0] += CLOUD_RETRY_AFTER_SECONDS + 1
    claude.error = None
    assert talk(llm) == "Reply from anthropic."
    assert llm.cloud_state() == HEALTHY


def test_exhausted_budget_keeps_the_turn_local_without_calling_claude(tmp_path) -> None:
    budget = ModelUsageBudget(tmp_path / "usage.json")
    llm, ollama, claude = make_llm(budget=budget, daily_budget_usd=0.0001)

    assert talk(llm) == "Reply from ollama."
    assert claude.calls == []
    assert llm.last_routing_decision.reason_code == "daily_budget_exceeded"


def test_private_data_keeps_the_turn_local_without_calling_claude() -> None:
    llm, ollama, claude = make_llm()

    llm.mark_next_call_privacy(PRIVATE)
    assert llm.chat("Resume as minhas notas.", source="RESPONSE_COMPOSER") == "Reply from ollama."
    assert claude.calls == [] and llm.last_routing_decision.reason_code == "privacy_class_private"


def test_document_tasks_keep_their_own_deadline() -> None:
    llm, ollama, claude = make_llm()
    profile = {"task_type": "document_refinement", "document_chars": 4000, "preferred_provider": "auto"}

    llm.mark_next_call_privacy(PRIVATE, True)
    llm.chat("ainda consegues melhor", source="RESPONSE_COMPOSER", timeout_seconds=45.0, task_profile=profile)

    assert claude.calls == [{"timeout_seconds": 45.0}]


def test_claude_mode_has_no_reserve() -> None:
    llm, ollama, claude = make_llm(mode="claude")
    claude.error = RuntimeError("A Anthropic devolveu um erro HTTP 529.")

    with pytest.raises(RuntimeError):
        talk(llm)
    assert ollama.calls == []


# --- health -----------------------------------------------------------------------


def test_cloud_health_is_degraded_after_a_failure_and_said_plainly() -> None:
    llm, ollama, claude = make_llm()
    probes = default_probes(llm=llm)
    assert probes["cloud"]() == UNKNOWN

    claude.error = ProviderTimeoutError("slow", provider="anthropic")
    talk(llm)
    assert probes["cloud"]() == DEGRADED

    health = SystemHealth({"cloud": probes["cloud"]})
    assert "cloud model is not responding" in wellbeing_reply("how are you?", "en-GB", health, variant=0)
    assert "modelo na cloud não está a responder" in wellbeing_reply("como estás?", "pt-PT", health, variant=0)

    claude.error = None
    llm._cloud_failed_at = None
    talk(llm)
    assert probes["cloud"]() == HEALTHY


# --- through the engine: the label names the model that answered -------------------


def _engine(tmp_path: Path, llm) -> AssistantEngine:
    data = tmp_path / "data"
    (tmp_path / "workspace").mkdir(exist_ok=True)
    memory = LongTermMemory(data, embedder=None)
    memory.set_preference("response_language", "en-GB")
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory,
        tools=ToolRegistry(),
        workspace_path=tmp_path / "workspace",
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


def test_ui_label_follows_the_model_that_answered(tmp_path: Path) -> None:
    llm, ollama, claude = make_llm()
    engine = _engine(tmp_path, llm)
    runtime = ModelRuntimeBridge(router=llm.router, budget=None, store=UserSettingsStore(tmp_path / "user_settings.json"),
                                 ollama_model=ollama.model, anthropic_model=claude.model, env={})

    assert engine.respond("I passed my exam today.") == "Reply from anthropic."
    payload = runtime.telemetry_payload(engine.get_last_turn_telemetry())
    assert (payload["mode"], payload["execution_provider"], payload["execution_model"]) == ("automatic", "anthropic", "claude-haiku-4-5-20251001")
    assert payload["paid_call"] is True

    claude.error = ProviderTimeoutError("slow", provider="anthropic")
    assert engine.respond("And I slept badly.") == "Reply from ollama."
    payload = runtime.telemetry_payload(engine.get_last_turn_telemetry())
    assert (payload["mode"], payload["execution_provider"], payload["execution_model"]) == ("automatic", "ollama", "llama3.1:8b")
    assert payload["paid_call"] is False
    assert payload["fallback_reason"] == "claude_timeout" and payload["reason_code"] == "claude_timeout"
