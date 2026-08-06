from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant.anthropic_provider import PAID_CALL_CONFIRMATION_ENV
from assistant.model_provider import ModelResponse, ProviderConfigurationError
from assistant.model_router import (
    AutomaticRoutingConfig,
    ModelRouter,
    ModelRoutingConfig,
    ModelRoutingInput,
    ModelUsageBudget,
    RoutedLLM,
    resolve_model_routing_config,
)


class FakeProvider:
    def __init__(self, name: str, model: str, text: str = "resposta") -> None:
        self._name = name
        self.model = model
        self.text = text
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    def chat(
        self,
        messages,
        *,
        model=None,
        response_format=None,
        tools=None,
        temperature=None,
        num_predict=None,
        timeout_seconds=None,
    ):
        self.calls.append({
            "messages": messages,
            "model": model,
            "response_format": response_format,
            "temperature": temperature,
            "num_predict": num_predict,
            "timeout_seconds": timeout_seconds,
        })
        if self._name == "anthropic" and self.text == "missing-key":
            raise ProviderConfigurationError(
                "O provider Anthropic esta selecionado, mas falta configurar ANTHROPIC_API_KEY.",
                provider="anthropic",
                provider_error_type="missing_api_key",
            )
        return ModelResponse(
            text=self.text,
            provider=self._name,
            model=model or self.model,
            input_tokens=100,
            output_tokens=20,
            latency_ms=3.0,
            estimated_cost_usd=0.001 if self._name == "anthropic" else 0.0,
        )


def _router(mode: str = "local", *, claude_enabled: bool = False, env: dict[str, str] | None = None, budget=None) -> ModelRouter:
    return ModelRouter(
        ModelRoutingConfig(
            mode=mode,
            mode_source="test",
            automatic=AutomaticRoutingConfig(claude_enabled=claude_enabled, daily_budget_usd=0.25, max_single_call_estimated_usd=0.05),
        ),
        ollama_model="llama3.1:8b",
        anthropic_model="claude-haiku-4-5-20251001",
        env=env or {},
        budget=budget,
    )


def test_resolve_model_routing_cli_wins_over_env_and_settings() -> None:
    config = resolve_model_routing_config(
        cli_mode="claude",
        env={"ECHO_MODEL_MODE": "automatic"},
        settings={"model_routing": {"mode": "local"}},
    )

    assert config.mode == "claude"
    assert config.mode_source == "cli"


def test_resolve_model_routing_uses_env_before_settings() -> None:
    config = resolve_model_routing_config(env={"ECHO_MODEL_MODE": "automatic"}, settings={"model_routing": {"mode": "local"}})

    assert config.mode == "automatic"
    assert config.mode_source == "ECHO_MODEL_MODE"


def test_resolve_model_routing_defaults_to_local() -> None:
    assert resolve_model_routing_config(env={}, settings={}).mode == "local"


def test_resolve_model_routing_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Modo de modelo desconhecido"):
        resolve_model_routing_config(cli_mode="magic", env={}, settings={})


def test_local_mode_always_chooses_ollama() -> None:
    decision = _router("local").decide(ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="explica em detalhe"))

    assert decision.provider == "ollama"
    assert decision.model == "llama3.1:8b"
    assert decision.paid_call is False


def test_claude_mode_chooses_anthropic_without_fallback() -> None:
    decision = _router("claude").decide(ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="olá"))

    assert decision.provider == "anthropic"
    assert decision.paid_call is True
    assert decision.reason_code == "claude_mode"


def test_automatic_simple_request_stays_local_even_when_paid_allowed(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")).decide(
        ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="Olá")
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "low_complexity"


def test_automatic_complex_request_uses_claude_only_when_authorized(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")).decide(
        ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="Faz um plano detalhado para estudar.", prompt_chars=1200)
    )

    assert decision.provider == "anthropic"
    assert decision.paid_call is True
    assert decision.reason_code in {"complex_request", "professional_writing", "structured_summary", "complex_planning"}


def test_automatic_professional_email_uses_claude_when_authorized(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="Escreve um email profissional detalhado a explicar o estado atual do projeto Echo, os progressos realizados e os próximos passos.",
            prompt_chars=1500,
        )
    )

    assert decision.provider == "anthropic"
    assert decision.reason_code == "professional_writing"


def test_automatic_structured_summary_uses_claude_when_authorized(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="Resume este texto em quatro pontos claros: A biblioteca alargou o horário durante exames e vai avaliar a medida.",
            prompt_chars=1200,
        )
    )

    assert decision.provider == "anthropic"
    assert decision.reason_code == "structured_summary"


def test_structured_summary_reason_wins_over_long_prompt(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="Resume este texto em quatro pontos claros: " + ("conteúdo " * 600),
            prompt_chars=5000,
            user_message_chars=120,
            context_chars=4880,
            constraint_count=2,
        )
    )

    assert decision.provider == "anthropic"
    assert decision.reason_code == "structured_summary"


def test_short_message_with_large_internal_prompt_does_not_become_long_prompt(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="Há quanto tempo estamos a trabalhar no projeto Echo?",
            prompt_chars=58,
            user_message_chars=58,
            context_chars=0,
            constraint_count=0,
        )
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "low_complexity"
    assert decision.routing_user_message_chars == 58
    assert decision.routing_context_chars == 0


def test_automatic_short_summary_without_structure_stays_local(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")).decide(
        ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="Resume isto.", prompt_chars=600)
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "low_complexity"


def test_automatic_without_api_key_falls_back_to_local_without_paid_call(tmp_path: Path) -> None:
    decision = _router("automatic", claude_enabled=True, env={}, budget=ModelUsageBudget(tmp_path / "usage.json")).decide(
        ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="Faz um plano detalhado.", prompt_chars=1200)
    )

    assert decision.provider == "ollama"
    assert decision.fallback_reason == "missing_api_key"
    assert decision.paid_call is False


def test_automatic_without_paid_confirmation_falls_back_to_local(tmp_path: Path) -> None:
    decision = _router(
        "automatic",
        claude_enabled=True,
        env={"ANTHROPIC_API_KEY": "secret"},
        budget=ModelUsageBudget(tmp_path / "usage.json"),
    ).decide(ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="Faz um plano detalhado.", prompt_chars=1200))

    assert decision.provider == "ollama"
    assert decision.fallback_reason == "paid_calls_not_confirmed"


def test_automatic_budget_exceeded_falls_back_to_local(tmp_path: Path) -> None:
    budget_path = tmp_path / "usage.json"
    budget_path.write_text(json.dumps({"date": "2099-01-01"}), encoding="utf-8")
    # Corrupt/out-of-date state is reset, so use a real budget with tiny limit.
    router = ModelRouter(
        ModelRoutingConfig(
            mode="automatic",
            automatic=AutomaticRoutingConfig(claude_enabled=True, daily_budget_usd=0.00001, max_single_call_estimated_usd=0.00001),
        ),
        env={"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"},
        budget=ModelUsageBudget(tmp_path / "usage.json"),
    )

    decision = router.decide(ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="Faz um plano detalhado.", prompt_chars=1200))

    assert decision.provider == "ollama"
    assert decision.fallback_reason in {"single_call_budget_exceeded", "daily_budget_exceeded"}


def test_automatic_corrupt_budget_state_falls_back_to_local(tmp_path: Path) -> None:
    budget_path = tmp_path / "usage.json"
    budget_path.write_text("{not-json", encoding="utf-8")

    decision = _router(
        "automatic",
        claude_enabled=True,
        env={"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"},
        budget=ModelUsageBudget(budget_path),
    ).decide(ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="Faz um plano detalhado.", prompt_chars=1200))

    assert decision.provider == "ollama"
    assert decision.fallback_reason == "budget_state_unavailable"


def test_tool_selector_source_never_uses_paid_model(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")).decide(
        ModelRoutingInput(source="TOOL_SELECTOR", user_message="escolhe ferramenta", prompt_chars=5000)
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "source_kept_local"


def test_explicit_provider_override_wins_over_mode() -> None:
    decision = _router("local").decide(
        ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="olá", explicit_provider="anthropic", explicit_model="claude-x")
    )

    assert decision.provider == "anthropic"
    assert decision.model == "claude-x"
    assert decision.override_source == "cli_provider"


def test_routed_llm_local_calls_only_ollama() -> None:
    ollama = FakeProvider("ollama", "llama3.1:8b")
    anthropic = FakeProvider("anthropic", "claude-haiku-4-5-20251001")
    llm = RoutedLLM(providers={"ollama": ollama, "anthropic": anthropic}, router=_router("local"), system_prompt="base")

    assert llm.chat("Olá", source="RESPONSE_COMPOSER") == "resposta"

    assert len(ollama.calls) == 1
    assert anthropic.calls == []
    assert llm.chat_call_tokens[0]["provider"] == "ollama"
    assert llm.settings.model_routing_mode == "local"


def test_routed_llm_automatic_complex_calls_anthropic_when_authorized(tmp_path: Path) -> None:
    ollama = FakeProvider("ollama", "llama3.1:8b")
    anthropic = FakeProvider("anthropic", "claude-haiku-4-5-20251001")
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    llm = RoutedLLM(
        providers={"ollama": ollama, "anthropic": anthropic},
        router=_router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")),
    )

    llm.chat("Faz um plano detalhado para estudar.", source="RESPONSE_COMPOSER")

    assert ollama.calls == []
    assert len(anthropic.calls) == 1
    assert llm.chat_call_tokens[0]["provider"] == "anthropic"
    assert llm.chat_call_tokens[0]["estimated_cost_usd"] > 0


def test_routed_llm_routes_on_original_composer_message_not_system_prompt(tmp_path: Path) -> None:
    ollama = FakeProvider("ollama", "llama3.1:8b")
    anthropic = FakeProvider("anthropic", "claude-haiku-4-5-20251001")
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    llm = RoutedLLM(
        providers={"ollama": ollama, "anthropic": anthropic},
        router=_router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "usage.json")),
        system_prompt="sistema " * 1000,
    )

    prompt = "Mensagem do Alexandre:\nHá quanto tempo estamos a trabalhar no projeto Echo?\n\nIntenção:\nconversa"
    llm.chat(prompt, history=[], source="RESPONSE_COMPOSER")

    assert anthropic.calls == []
    assert len(ollama.calls) == 1
    assert llm.settings.model_routing_reason_code == "low_complexity"
    assert llm.settings.routing_user_message_chars == len("Há quanto tempo estamos a trabalhar no projeto Echo?")


def test_routed_llm_claude_mode_surfaces_missing_key_without_ollama_fallback() -> None:
    ollama = FakeProvider("ollama", "llama3.1:8b")
    anthropic = FakeProvider("anthropic", "claude-haiku-4-5-20251001", text="missing-key")
    llm = RoutedLLM(providers={"ollama": ollama, "anthropic": anthropic}, router=_router("claude"))

    with pytest.raises(ProviderConfigurationError, match="ANTHROPIC_API_KEY"):
        llm.chat("Olá", source="RESPONSE_COMPOSER")

    assert ollama.calls == []
    assert len(anthropic.calls) == 1


def test_budget_register_stores_only_totals(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    budget = ModelUsageBudget(path)

    budget.register(
        ModelResponse(
            text="resposta secreta",
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            estimated_cost_usd=0.001,
        )
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["calls"] == 1
    assert data["providers"]["anthropic"]["input_tokens"] == 10
    assert "resposta secreta" not in path.read_text(encoding="utf-8")
