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


def _document_profile(**overrides) -> dict:
    """A document task profile shaped like a small, structurally simple
    first-time rewrite -- override fields per test to model refinement,
    prior local failures, larger documents, etc. Mirrors the structure
    assistant/conversation.py's rewrite branch builds (see
    _build_task_profile), never containing document content."""
    profile = {
        "task_type": "document_rewrite",
        "document_chars": 200,
        "document_has_draft": False,
        "document_revision_number": 0,
        "document_named_entity_count": 2,
        "document_list_item_count": 4,
        "document_requires_fidelity": True,
        "document_requires_full_output": True,
        "document_previous_local_failure": False,
        "document_validation_failure_reason": "",
        "document_regeneration_attempt": 1,
        "preferred_provider": "auto",
    }
    profile.update(overrides)
    return profile


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


# --- Document task_profile-driven escalation policy: complexity is scored
# from the RECONSTRUCTED task (revision number, prior local failure,
# structural density...), never from raw message length alone -- a short
# follow-up like "ainda consegues melhor" carries none of that on its own.

_PAID_ENV = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}


def test_document_task_simple_first_rewrite_stays_local_first(tmp_path: Path) -> None:
    """C: a short, simple first-time rewrite is Ollama-preferred (Via 2:
    local first), not sent straight to Claude."""
    decision = _router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="torna-o mais formal", task_profile=_document_profile())
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "document_local_first"
    assert decision.task_complexity_band == "medium"
    assert decision.escalation_considered is False


def test_document_task_trivial_document_is_low_complexity(tmp_path: Path) -> None:
    decision = _router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="torna-o mais formal",
            task_profile=_document_profile(document_chars=10, document_named_entity_count=0, document_list_item_count=0),
        )
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "document_simple_local"
    assert decision.task_complexity_band == "low"


def test_document_task_iterative_refinement_escalates_when_authorized(tmp_path: Path) -> None:
    """D: an iterative refinement with an active draft is high complexity
    and escalates to Claude once paid calls are authorized."""
    decision = _router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile=_document_profile(task_type="document_refinement", document_has_draft=True, document_revision_number=1),
        )
    )

    assert decision.provider == "anthropic"
    assert decision.paid_call is True
    assert decision.reason_code == "iterative_refinement_high_complexity"
    assert decision.task_complexity_band == "high"


def test_document_task_refinement_stays_local_without_paid_calls_confirmed(tmp_path: Path) -> None:
    """E: same high-complexity refinement, but paid calls are not
    confirmed -- stays on Ollama with a document-specific blocked reason."""
    env = {"ANTHROPIC_API_KEY": "secret"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile=_document_profile(task_type="document_refinement", document_has_draft=True, document_revision_number=1),
        )
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "document_escalation_blocked_paid_disabled"
    assert decision.paid_call is False


def test_document_task_first_attempt_placeholder_escalates_second_attempt(tmp_path: Path) -> None:
    """F: a first local attempt rejected for a placeholder makes the second
    attempt's profile escalation-eligible."""
    decision = _router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="torna-o mais formal",
            task_profile=_document_profile(
                document_previous_local_failure=True,
                document_validation_failure_reason="placeholder_detected",
                document_regeneration_attempt=2,
            ),
        )
    )

    assert decision.provider == "anthropic"
    assert decision.reason_code == "document_escalated_after_local_failure"
    assert decision.task_complexity_band == "high"


def test_document_task_first_attempt_valid_does_not_escalate(tmp_path: Path) -> None:
    """G: a first attempt with no failure signal never escalates."""
    decision = _router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="torna-o mais formal", task_profile=_document_profile())
    )

    assert decision.provider == "ollama"
    assert decision.escalation_considered is False


def test_document_task_noop_refinement_can_escalate_second_attempt(tmp_path: Path) -> None:
    """H: a no-op refinement result on the first attempt makes the second
    attempt escalation-eligible too."""
    decision = _router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile=_document_profile(
                task_type="document_refinement",
                document_has_draft=True,
                document_revision_number=1,
                document_previous_local_failure=True,
                document_validation_failure_reason="no_op_detected",
                document_regeneration_attempt=2,
            ),
        )
    )

    assert decision.provider == "anthropic"
    assert decision.reason_code == "document_escalated_after_local_failure"


def test_document_task_insufficient_budget_blocks_escalation(tmp_path: Path) -> None:
    """I: high-complexity task, everything authorized, but budget is too
    small -- stays on Ollama with a document-specific budget reason."""
    router = ModelRouter(
        ModelRoutingConfig(
            mode="automatic",
            automatic=AutomaticRoutingConfig(claude_enabled=True, daily_budget_usd=0.00001, max_single_call_estimated_usd=0.00001),
        ),
        env=_PAID_ENV,
        budget=ModelUsageBudget(tmp_path / "u.json"),
    )
    decision = router.decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile=_document_profile(task_type="document_refinement", document_has_draft=True, document_revision_number=1),
        )
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "document_escalation_blocked_budget"


def test_document_task_missing_api_key_blocks_escalation(tmp_path: Path) -> None:
    """J: high-complexity task, paid calls confirmed, but no API key --
    stays on Ollama with a document-specific api-key reason."""
    env = {PAID_CALL_CONFIRMATION_ENV: "true"}
    decision = _router("automatic", claude_enabled=True, env=env, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile=_document_profile(task_type="document_refinement", document_has_draft=True, document_revision_number=1),
        )
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "document_escalation_blocked_api_key"


def test_document_task_claude_disabled_blocks_escalation(tmp_path: Path) -> None:
    """K: automatic_claude_enabled=false blocks escalation regardless of
    complexity, same as any other automatic-mode call."""
    decision = _router("automatic", claude_enabled=False, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile=_document_profile(task_type="document_refinement", document_has_draft=True, document_revision_number=1),
        )
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "automatic_claude_disabled"


def test_document_task_local_regeneration_reason_when_still_medium(tmp_path: Path) -> None:
    """A prior local failure on a trivial-enough document does not
    automatically clear the high-complexity bar -- stays local with a
    distinct reason instead of silently reusing document_local_first."""
    decision = _router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="torna-o mais formal",
            task_profile=_document_profile(
                document_named_entity_count=0,
                document_list_item_count=0,
                document_chars=0,
                document_previous_provider="ollama",
                document_previous_local_failure=True,
                document_previous_validation_failed=True,
                document_validation_failure_reason="termina_em_pergunta",
                document_regeneration_attempt=2,
            ),
        )
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "document_local_regeneration"
    assert decision.task_complexity_band == "medium"


def test_document_task_preferred_provider_ollama_forces_local(tmp_path: Path) -> None:
    """Section 7: the document flow can force attempt 1 to stay local even
    when the profile would otherwise score as high complexity."""
    decision = _router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile=_document_profile(
                task_type="document_refinement", document_has_draft=True, document_revision_number=1, preferred_provider="ollama"
            ),
        )
    )

    assert decision.provider == "ollama"
    assert decision.reason_code == "document_local_first"


def test_document_task_complexity_fields_recorded_on_decision(tmp_path: Path) -> None:
    """N: task_complexity_score/band and escalation_considered are exposed
    on the decision object for telemetry."""
    decision = _router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")).decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile=_document_profile(task_type="document_refinement", document_has_draft=True, document_revision_number=1),
        )
    )

    assert decision.task_complexity_score > 40.0
    assert decision.task_complexity_band == "high"
    assert decision.escalation_considered is True


def test_routed_llm_forwards_task_profile_and_escalates(tmp_path: Path) -> None:
    """O: RoutedLLM.chat() actually threads task_profile through to the
    router -- no real Anthropic call, FakeProvider stands in."""
    ollama = FakeProvider("ollama", "llama3.1:8b")
    anthropic = FakeProvider("anthropic", "claude-haiku-4-5-20251001")
    llm = RoutedLLM(
        providers={"ollama": ollama, "anthropic": anthropic},
        router=_router("automatic", claude_enabled=True, env=_PAID_ENV, budget=ModelUsageBudget(tmp_path / "u.json")),
    )

    llm.chat(
        "ainda consegues melhor",
        source="RESPONSE_COMPOSER",
        task_profile=_document_profile(task_type="document_refinement", document_has_draft=True, document_revision_number=1),
    )

    assert ollama.calls == []
    assert len(anthropic.calls) == 1
    assert llm.last_routing_decision.reason_code == "iterative_refinement_high_complexity"


def test_routed_llm_without_task_profile_uses_generic_heuristic() -> None:
    """Every non-document call site never sets task_profile -- confirms the
    generic text-based heuristic is unchanged for them (RoutedLLM defaults
    task_profile to None)."""
    ollama = FakeProvider("ollama", "llama3.1:8b")
    anthropic = FakeProvider("anthropic", "claude-haiku-4-5-20251001")
    llm = RoutedLLM(providers={"ollama": ollama, "anthropic": anthropic}, router=_router("local"))

    llm.chat("Olá", source="RESPONSE_COMPOSER")

    assert llm.last_routing_decision.task_complexity_band == ""
    assert ollama.calls[0].get("timeout_seconds") is None
