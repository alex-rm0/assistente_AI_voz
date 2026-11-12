"""Controlled model routing for Echo.

This module decides which configured provider may answer an LLM call. It never
executes tools, never reads prompts from disk and never stores prompt text in
the budget file.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from assistant.anthropic_provider import PAID_CALL_CONFIRMATION_ENV
from assistant.model_provider import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_MODEL,
    ModelProvider,
    ModelResponse,
    ProviderBackedLLM,
    ProviderConfigurationError,
    ProviderTimeoutError,
)
from assistant.privacy import (
    DEFAULT_PRIVACY_CLASS,
    content_privacy_class,
    normalize_privacy_class,
    provider_allowed,
    stricter,
)


SUPPORTED_MODEL_MODES = ("local", "claude", "automatic")
NO_PAID_CALL_SOURCES = {
    "TOOL_SELECTOR",
    "MEMORY_SUMMARY",
    "SESSION_SUMMARY",
    "VOICE_CRITIC",
    "PLANNER",
}
# Decision of 12/11 (North Star §11): general conversation goes to Claude when
# the data may leave the machine; the local model is the reserve.
CONVERSATION_SOURCES = {"RESPONSE_COMPOSER", "RESPONSE_COMPOSER_REGENERATION"}
# A conversational turn does not wait longer than this for Claude before the
# local model answers instead. Document tasks keep their caller's own deadline.
CLAUDE_CONVERSATION_TIMEOUT_SECONDS = 4.0
# After a failed Claude call, stay local for this long instead of paying the
# timeout again on every turn.
CLOUD_RETRY_AFTER_SECONDS = 60.0


def paid_calls_allowed(env: dict[str, str] | None, setting: object) -> bool:
    """Are paid model calls switched on?

    The switch is `allow_paid_calls` in data/user_settings.json (off unless
    Alexandre turns it on). ECHO_ALLOW_PAID_MODEL_CALLS is only an override:
    "true" forces them on, "false" forces them off, unset leaves the setting.
    """
    override = str((env if env is not None else os.environ).get(PAID_CALL_CONFIRMATION_ENV) or "").strip().lower()
    if override in ("true", "false"):
        return override == "true"
    return setting is True


@dataclass(frozen=True)
class AutomaticRoutingConfig:
    claude_enabled: bool = False
    daily_budget_usd: float = 0.25
    max_single_call_estimated_usd: float = 0.05


@dataclass(frozen=True)
class ModelRoutingConfig:
    mode: str = "local"
    mode_source: str = "default"
    automatic: AutomaticRoutingConfig = AutomaticRoutingConfig()


@dataclass(frozen=True)
class ModelRoutingInput:
    source: str
    user_message: str
    prompt_chars: int = 0
    user_message_chars: int = 0
    context_chars: int = 0
    constraint_count: int = 0
    has_tools: bool = False
    has_context: bool = False
    explicit_provider: str = ""
    explicit_model: str = ""
    # Structured task description built by the document-rewrite/refinement
    # flow (see assistant/conversation.py) -- when present, complexity is
    # computed from these fields instead of the raw message-length/keyword
    # heuristic below, since a short follow-up ("ainda consegues melhor")
    # can hide a genuinely complex reconstructed task. Never contains the
    # document's actual content. See _document_task_complexity_score.
    task_profile: dict[str, object] | None = None
    # North Star §6: the class of the data in this request. Nothing declared
    # means PRIVATE. explicit_document_request is set when the user's own turn
    # asks about that document, the one case in which PRIVATE data may go out.
    privacy_class: str = DEFAULT_PRIVACY_CLASS
    explicit_document_request: bool = False


@dataclass(frozen=True)
class ModelRoutingDecision:
    mode: str
    provider: str
    model: str
    reason_code: str
    reason: str
    paid_call: bool = False
    budget_before_usd: float = 0.0
    budget_after_usd: float = 0.0
    fallback_reason: str = ""
    override_source: str = ""
    routing_user_message_chars: int = 0
    routing_context_chars: int = 0
    routing_constraint_count: int = 0
    task_complexity_score: float = 0.0
    task_complexity_band: str = ""
    escalation_considered: bool = False
    privacy_class: str = ""


class ModelUsageBudget:
    """Small local budget ledger for paid model calls.

    Stores only usage totals and the date. It deliberately does not store
    prompts, responses, user messages or memory/context snippets.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def snapshot(self) -> dict[str, Any] | None:
        today = date.today().isoformat()
        if not self.path.exists():
            return self._empty(today)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("date") != today:
            return self._empty(today)
        return data

    def can_spend(self, estimated_cost_usd: float, daily_budget_usd: float, max_single_call_usd: float) -> tuple[bool, str, float, float]:
        data = self.snapshot()
        if data is None:
            return False, "budget_state_unavailable", 0.0, 0.0
        before = float(data.get("accumulated_estimated_cost_usd") or 0.0)
        after = before + max(0.0, float(estimated_cost_usd or 0.0))
        if estimated_cost_usd > max_single_call_usd:
            return False, "single_call_budget_exceeded", before, after
        if after > daily_budget_usd:
            return False, "daily_budget_exceeded", before, after
        return True, "", before, after

    def register(self, response: ModelResponse) -> None:
        if response.provider != "anthropic":
            return
        data = self.snapshot()
        if data is None:
            return
        provider_totals = data.setdefault("providers", {}).setdefault(
            response.provider,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0},
        )
        provider_totals["calls"] = int(provider_totals.get("calls") or 0) + 1
        provider_totals["input_tokens"] = int(provider_totals.get("input_tokens") or 0) + int(response.input_tokens or 0)
        provider_totals["output_tokens"] = int(provider_totals.get("output_tokens") or 0) + int(response.output_tokens or 0)
        provider_totals["estimated_cost_usd"] = float(provider_totals.get("estimated_cost_usd") or 0.0) + float(
            response.estimated_cost_usd or 0.0
        )
        data["calls"] = int(data.get("calls") or 0) + 1
        data["input_tokens"] = int(data.get("input_tokens") or 0) + int(response.input_tokens or 0)
        data["output_tokens"] = int(data.get("output_tokens") or 0) + int(response.output_tokens or 0)
        data["accumulated_estimated_cost_usd"] = float(data.get("accumulated_estimated_cost_usd") or 0.0) + float(
            response.estimated_cost_usd or 0.0
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            return

    @staticmethod
    def _empty(today: str) -> dict[str, Any]:
        return {
            "date": today,
            "accumulated_estimated_cost_usd": 0.0,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "providers": {},
        }


class ModelRouter:
    def __init__(
        self,
        config: ModelRoutingConfig,
        *,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
        budget: ModelUsageBudget | None = None,
        env: dict[str, str] | None = None,
        anthropic_key_available: Callable[[], bool] | None = None,
        paid_calls_allowed: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.ollama_model = ollama_model
        self.anthropic_model = anthropic_model
        self.budget = budget
        self.env = env if env is not None else os.environ
        self.anthropic_key_available = anthropic_key_available
        self.paid_calls_allowed = paid_calls_allowed

    def decide(self, routing_input: ModelRoutingInput) -> ModelRoutingDecision:
        """Choose the provider, then check the privacy class at this boundary:
        whatever the mode, data that may not leave the machine stays local."""
        routing_input = _with_routing_metrics(routing_input)
        privacy_class = normalize_privacy_class(routing_input.privacy_class)
        decision = self._decide(routing_input)
        if not provider_allowed(decision.provider, privacy_class, explicit_document_request=routing_input.explicit_document_request):
            decision = self._privacy_blocked(decision.mode, privacy_class, routing_input)
        return replace(decision, privacy_class=privacy_class)

    def _privacy_blocked(self, mode: str, privacy_class: str, routing_input: ModelRoutingInput) -> ModelRoutingDecision:
        return self._ollama(
            mode,
            f"privacy_class_{privacy_class.lower()}",
            f"Dados de classe {privacy_class}; ficam no modelo local.",
            "privacy_class_not_allowed",
            routing_input=routing_input,
        )

    def _paid_calls_allowed(self) -> bool:
        if self.paid_calls_allowed is not None:
            return bool(self.paid_calls_allowed())
        return paid_calls_allowed(self.env, False)

    def _decide(self, routing_input: ModelRoutingInput) -> ModelRoutingDecision:
        mode = self.config.mode
        if routing_input.explicit_provider:
            provider = routing_input.explicit_provider
            model = routing_input.explicit_model or (self.anthropic_model if provider == "anthropic" else self.ollama_model)
            return ModelRoutingDecision(
                mode=mode,
                provider=provider,
                model=model,
                reason_code="explicit_provider",
                reason="Provider escolhido explicitamente.",
                paid_call=provider == "anthropic",
                override_source="cli_provider",
                routing_user_message_chars=routing_input.user_message_chars,
                routing_context_chars=routing_input.context_chars,
                routing_constraint_count=routing_input.constraint_count,
            )

        if mode == "local":
            return ModelRoutingDecision(
                mode=mode,
                provider="ollama",
                model=self.ollama_model,
                reason_code="local_mode",
                reason="Modo local.",
                routing_user_message_chars=routing_input.user_message_chars,
                routing_context_chars=routing_input.context_chars,
                routing_constraint_count=routing_input.constraint_count,
            )
        if mode == "claude":
            return ModelRoutingDecision(
                mode=mode,
                provider="anthropic",
                model=self.anthropic_model,
                reason_code="claude_mode",
                reason="Modo Claude escolhido explicitamente.",
                paid_call=True,
                override_source=self.config.mode_source,
                routing_user_message_chars=routing_input.user_message_chars,
                routing_context_chars=routing_input.context_chars,
                routing_constraint_count=routing_input.constraint_count,
            )
        return self._automatic_decision(routing_input)

    def _automatic_decision(self, routing_input: ModelRoutingInput) -> ModelRoutingDecision:
        source = str(routing_input.source or "").upper()
        if source in NO_PAID_CALL_SOURCES:
            return self._ollama("automatic", "source_kept_local", f"A origem {source} fica no modelo local.", routing_input=routing_input)

        profile = routing_input.task_profile
        task_type = str(profile.get("task_type") or "") if profile else ""
        is_document_task = bool(profile) and task_type.startswith("document_")

        if not self.config.automatic.claude_enabled:
            return self._ollama("automatic", "automatic_claude_disabled", "Claude automatico esta desligado.", routing_input=routing_input)
        if not self._has_anthropic_key():
            return self._ollama(
                "automatic",
                "document_escalation_blocked_api_key" if is_document_task else "missing_api_key",
                "Sem ANTHROPIC_API_KEY; fica no modelo local.",
                "missing_api_key",
                routing_input=routing_input,
            )
        if not self._paid_calls_allowed():
            return self._ollama(
                "automatic",
                "document_escalation_blocked_paid_disabled" if is_document_task else "paid_calls_not_confirmed",
                "Chamadas pagas desligadas (allow_paid_calls); fica no modelo local.",
                "paid_calls_not_confirmed",
                routing_input=routing_input,
            )
        privacy_class = normalize_privacy_class(routing_input.privacy_class)
        if not provider_allowed("anthropic", privacy_class, explicit_document_request=routing_input.explicit_document_request):
            return self._privacy_blocked("automatic", privacy_class, routing_input)

        if is_document_task:
            return self._document_task_decision(routing_input, profile)

        # A specific reason (professional writing, structured summary...) is kept; plain conversation has its own.
        complexity_reason = _complexity_reason_for_claude(routing_input) or ("general_conversation" if source in CONVERSATION_SOURCES else "")
        if not complexity_reason:
            return self._ollama("automatic", "low_complexity", "Pedido simples; o modelo local e suficiente.", routing_input=routing_input)

        estimated_cost = _estimate_prompt_cost_usd(routing_input.prompt_chars)
        before = 0.0
        after = estimated_cost
        if self.budget is not None:
            allowed, reason, before, after = self.budget.can_spend(
                estimated_cost,
                self.config.automatic.daily_budget_usd,
                self.config.automatic.max_single_call_estimated_usd,
            )
            if not allowed:
                return self._ollama(
                    "automatic",
                    reason,
                    "Orcamento automatico impede chamada paga.",
                    reason,
                    before,
                    after,
                    routing_input=routing_input,
                )
        return ModelRoutingDecision(
            mode="automatic",
            provider="anthropic",
            model=self.anthropic_model,
            reason_code=complexity_reason,
            reason="Pedido elegivel para Claude e chamada paga explicitamente autorizada dentro do orcamento.",
            paid_call=True,
            budget_before_usd=before,
            budget_after_usd=after,
            routing_user_message_chars=routing_input.user_message_chars,
            routing_context_chars=routing_input.context_chars,
            routing_constraint_count=routing_input.constraint_count,
        )

    def _document_task_decision(self, routing_input: ModelRoutingInput, profile: dict[str, object]) -> ModelRoutingDecision:
        """Complexity is computed from the structured task profile the
        document-rewrite/refinement flow reconstructs (revision number,
        prior local validation failure, structural density...), not from
        the literal length of a possibly-short follow-up message like
        "ainda consegues melhor" -- see _document_task_complexity_score."""
        preferred = str(profile.get("preferred_provider") or "auto").strip().lower()
        if preferred == "ollama":
            return self._ollama(
                "automatic", "document_local_first",
                "Fluxo documental pediu explicitamente o modelo local.",
                routing_input=routing_input,
            )

        score = _document_task_complexity_score(profile)
        band = _document_task_complexity_band(score)
        task_type = str(profile.get("task_type") or "")
        attempt_number = int(profile.get("document_regeneration_attempt") or 1)
        # A failed previous attempt is not automatically a *local* failure --
        # attempt 1 can itself already have been escalated to Claude on a
        # high-complexity profile. previous_local_failure/previous_claude_failure
        # distinguish Ollama-then-Claude from Claude-then-Claude so the reason
        # code never claims a local attempt happened when it didn't.
        previous_local_failure = bool(profile.get("document_previous_local_failure"))
        previous_validation_failed = bool(profile.get("document_previous_validation_failed"))
        previous_provider = str(profile.get("document_previous_provider") or "").strip().lower()
        previous_claude_failure = (
            previous_validation_failed and not previous_local_failure and previous_provider == "anthropic"
        )
        escalation_considered = band == "high" or previous_local_failure or previous_claude_failure

        def _decision(
            *, provider: str, reason_code: str, reason: str, paid_call: bool = False,
            before: float = 0.0, after: float = 0.0, fallback_reason: str = "",
        ) -> ModelRoutingDecision:
            model = self.anthropic_model if provider == "anthropic" else self.ollama_model
            return ModelRoutingDecision(
                mode="automatic", provider=provider, model=model,
                reason_code=reason_code, reason=reason, paid_call=paid_call,
                budget_before_usd=before, budget_after_usd=after, fallback_reason=fallback_reason,
                routing_user_message_chars=routing_input.user_message_chars,
                routing_context_chars=routing_input.context_chars,
                routing_constraint_count=routing_input.constraint_count,
                task_complexity_score=float(score), task_complexity_band=band,
                escalation_considered=escalation_considered,
            )

        if band == "low":
            return _decision(
                provider="ollama", reason_code="document_simple_local",
                reason="Complexidade documental baixa; o modelo local e suficiente.",
            )
        if band == "medium":
            if attempt_number <= 1:
                return _decision(
                    provider="ollama", reason_code="document_local_first",
                    reason="Primeira tentativa documental fica no modelo local.",
                )
            return _decision(
                provider="ollama", reason_code="document_local_regeneration",
                reason="Tentativa local falhou a validacao; nova tentativa local, complexidade nao justifica Claude.",
            )

        estimated_cost = _estimate_prompt_cost_usd(routing_input.prompt_chars)
        allowed, budget_reason, before, after = True, "", 0.0, estimated_cost
        if self.budget is not None:
            allowed, budget_reason, before, after = self.budget.can_spend(
                estimated_cost,
                self.config.automatic.daily_budget_usd,
                self.config.automatic.max_single_call_estimated_usd,
            )
        if not allowed:
            return _decision(
                provider="ollama", reason_code="document_escalation_blocked_budget",
                reason="Orcamento automatico impede a escalada documental para Claude.",
                before=before, after=after, fallback_reason=budget_reason,
            )

        if previous_local_failure:
            reason_code = "document_escalated_after_local_failure"
            reason = "Tentativa local falhou a validacao; a escalar para Claude dentro do orcamento."
        elif previous_claude_failure:
            reason_code = "document_regenerated_with_claude"
            reason = "Tentativa anterior em Claude falhou a validacao; a repetir com Claude dentro do orcamento."
        elif task_type == "document_refinement":
            reason_code = "iterative_refinement_high_complexity"
            reason = "Refinamento iterativo de alta complexidade; Claude escolhido dentro do orcamento."
        elif band == "high":
            reason_code = "document_high_complexity"
            reason = "Tarefa documental de alta complexidade; Claude escolhido dentro do orcamento."
        else:
            reason_code = "document_claude_selected"
            reason = "Tarefa documental elegivel para Claude dentro do orcamento."
        return _decision(provider="anthropic", reason_code=reason_code, reason=reason, paid_call=True, before=before, after=after)

    def _has_anthropic_key(self) -> bool:
        if self.anthropic_key_available is not None:
            return bool(self.anthropic_key_available())
        return bool(str(self.env.get("ANTHROPIC_API_KEY") or "").strip())

    def _ollama(
        self,
        mode: str,
        reason_code: str,
        reason: str,
        fallback_reason: str = "",
        before: float = 0.0,
        after: float = 0.0,
        routing_input: ModelRoutingInput | None = None,
    ) -> ModelRoutingDecision:
        return ModelRoutingDecision(
            mode=mode,
            provider="ollama",
            model=self.ollama_model,
            reason_code=reason_code,
            reason=reason,
            budget_before_usd=before,
            budget_after_usd=after,
            fallback_reason=fallback_reason,
            routing_user_message_chars=routing_input.user_message_chars if routing_input else 0,
            routing_context_chars=routing_input.context_chars if routing_input else 0,
            routing_constraint_count=routing_input.constraint_count if routing_input else 0,
        )


class RoutedLLM(ProviderBackedLLM):
    """ProviderBackedLLM compatible adapter with per-call routing."""

    def __init__(
        self,
        *,
        providers: dict[str, ModelProvider],
        router: ModelRouter,
        system_prompt: str = "",
        model_source: str = "model_router",
        explicit_provider: str = "",
    ) -> None:
        first_provider = providers.get(explicit_provider) or providers.get("ollama") or next(iter(providers.values()))
        super().__init__(first_provider, system_prompt=system_prompt, model_source=model_source)
        self.providers = providers
        self.router = router
        self.explicit_provider = explicit_provider
        self.last_routing_decision: ModelRoutingDecision | None = None
        self.routing_decisions: list[ModelRoutingDecision] = []
        self._next_call_source = ""
        self._next_call_privacy: tuple[str, bool] = (DEFAULT_PRIVACY_CLASS, False)
        self._clock = time.monotonic
        self._cloud_state = "unknown"
        self._cloud_failed_at: float | None = None
        self.last_cloud_failure_reason = ""

    def mark_next_call_source(self, source: str) -> None:
        self._next_call_source = str(source or "").strip()

    def mark_next_call_privacy(self, privacy_class: str, explicit_document_request: bool = False) -> None:
        """Declare the privacy class of the data in the next chat() call (North
        Star §6). It applies to that call only; an unmarked call is PRIVATE."""
        self._next_call_privacy = (normalize_privacy_class(privacy_class), bool(explicit_document_request))

    def cloud_state(self) -> str:
        """healthy / degraded / unknown, from the last Claude call. A probe for
        assistant/system_health.py: degraded means the local model is answering."""
        return self._cloud_state

    def _cloud_cooling_down(self) -> bool:
        return self._cloud_failed_at is not None and self._clock() - self._cloud_failed_at < CLOUD_RETRY_AFTER_SECONDS

    def _local_reserve(self, decision: ModelRoutingDecision, reason: str) -> ModelRoutingDecision:
        print(f"[MODEL_ROUTING] reserve=local reason={reason} source_decision={decision.reason_code}", flush=True)
        return replace(
            decision,
            provider="ollama",
            model=self.router.ollama_model,
            paid_call=False,
            reason_code=reason,
            reason="Claude indisponivel; o modelo local respondeu.",
            fallback_reason=reason,
        )

    @property
    def supports_local_semantic_extraction(self) -> bool:
        return False

    def chat(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        response_format: str | None = None,
        source: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        timeout_seconds: float | None = None,
        task_profile: dict[str, object] | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt or self.system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_message})
        source_name = str(source or self._next_call_source or "OTHER").strip() or "OTHER"
        self._next_call_source = ""
        declared_class, explicit_document_request = self._next_call_privacy
        self._next_call_privacy = (DEFAULT_PRIVACY_CLASS, False)
        # Known document or email markers anywhere in the payload outrank what the caller declared.
        found_class = content_privacy_class([str(message.get("content") or "") for message in messages])
        privacy_class = stricter(declared_class, found_class) if found_class else declared_class
        routing_message, routing_context_chars = _extract_routing_user_message(user_message, history or [])
        decision = self.router.decide(
            ModelRoutingInput(
                source=source_name,
                user_message=routing_message,
                prompt_chars=len(routing_message) + routing_context_chars,
                user_message_chars=len(routing_message),
                context_chars=routing_context_chars,
                constraint_count=_count_constraints(routing_message),
                explicit_provider=self.explicit_provider,
                task_profile=task_profile,
                privacy_class=privacy_class,
                explicit_document_request=explicit_document_request,
            )
        )
        call = dict(response_format=response_format, temperature=temperature, num_predict=num_predict)
        result = None
        # In automatic mode the local model is the reserve for Claude: a failed
        # or slow cloud call never fails the turn.
        has_reserve = (
            decision.provider == "anthropic" and decision.mode == "automatic"
            and not self.explicit_provider and "ollama" in self.providers and "anthropic" in self.providers
        )
        if has_reserve and self._cloud_cooling_down():
            decision = self._local_reserve(decision, "claude_cooldown")
        elif has_reserve:
            cloud_timeout = timeout_seconds
            if task_profile is None:
                cloud_timeout = min(value for value in (timeout_seconds, CLAUDE_CONVERSATION_TIMEOUT_SECONDS) if value is not None)
            try:
                result = self.providers["anthropic"].chat(messages, model=decision.model, timeout_seconds=cloud_timeout, **call)
                self._cloud_state, self._cloud_failed_at = "healthy", None
            except ProviderTimeoutError:
                failure = "claude_timeout"
            except ProviderConfigurationError as exc:
                failure = f"claude_{exc.provider_error_type or 'configuration_error'}"
            except Exception as exc:  # noqa: BLE001 - any cloud failure falls back to the local model
                failure = "claude_unreachable" if "ligar" in str(exc).lower() else "claude_error"
            if result is None:
                self._cloud_state, self._cloud_failed_at, self.last_cloud_failure_reason = "degraded", self._clock(), failure
                decision = self._local_reserve(decision, failure)
        if result is None:
            provider = self.providers.get(decision.provider)
            if provider is None:
                raise ProviderConfigurationError(
                    f"O provider {decision.provider} foi escolhido, mas nao esta configurado.",
                    provider=decision.provider,
                    provider_error_type="provider_not_configured",
                )
            result = provider.chat(messages, model=decision.model, timeout_seconds=timeout_seconds, **call)
        self.provider = self.providers[decision.provider]
        self.router.budget.register(result) if self.router.budget is not None else None
        self.last_response = result
        self.responses.append(result)
        self._call_sources.append(source_name)
        self.last_routing_decision = decision
        self.routing_decisions.append(decision)
        return result.text

    @property
    def settings(self) -> "_RoutedSettings":
        provider = self.last_response.provider if self.last_response else getattr(self.provider, "name", "")
        model = self.last_response.model if self.last_response else getattr(self.provider, "model", "")
        decision = self.last_routing_decision
        return _RoutedSettings(
            model=model,
            model_source=self.model_source,
            provider=provider,
            model_routing_mode=decision.mode if decision else self.router.config.mode,
            # Always the router's own configured mode_source ("user_settings",
            # "cli", "default", ...) — decision.override_source is a narrower,
            # per-call concept (only set for an explicit CLI/claude-mode
            # override) and is empty for the common "automatic"/"local" mode
            # decisions, which was making configured_model_mode_source go
            # blank again as soon as an actual LLM call happened.
            model_routing_mode_source=self.router.config.mode_source,
            # The provider and model that actually answered (the UI label reads
            # these), not the ones first chosen: after a reserve they differ.
            model_routing_provider=provider if self.last_response else (decision.provider if decision else provider),
            model_routing_model=model if self.last_response else (decision.model if decision else model),
            model_routing_reason_code=decision.reason_code if decision else "",
            model_routing_reason=decision.reason if decision else "",
            model_routing_paid_call=bool(decision.paid_call) if decision else False,
            model_routing_budget_before_usd=decision.budget_before_usd if decision else 0.0,
            model_routing_budget_after_usd=decision.budget_after_usd if decision else 0.0,
            model_routing_fallback_reason=decision.fallback_reason if decision else "",
            model_routing_override_source=decision.override_source if decision else "",
            routing_user_message_chars=decision.routing_user_message_chars if decision else 0,
            routing_context_chars=decision.routing_context_chars if decision else 0,
            routing_constraint_count=decision.routing_constraint_count if decision else 0,
        )


@dataclass
class _RoutedSettings:
    model: str
    model_source: str
    provider: str
    model_routing_mode: str = ""
    model_routing_mode_source: str = ""
    model_routing_provider: str = ""
    model_routing_model: str = ""
    model_routing_reason_code: str = ""
    model_routing_reason: str = ""
    model_routing_paid_call: bool = False
    model_routing_budget_before_usd: float = 0.0
    model_routing_budget_after_usd: float = 0.0
    model_routing_fallback_reason: str = ""
    model_routing_override_source: str = ""
    routing_user_message_chars: int = 0
    routing_context_chars: int = 0
    routing_constraint_count: int = 0


def resolve_model_routing_config(
    *,
    cli_mode: str | None = None,
    env: dict[str, str] | None = None,
    settings: dict[str, Any] | None = None,
) -> ModelRoutingConfig:
    environment = env if env is not None else os.environ
    config = settings or {}
    routing_config = config.get("model_routing", {}) if isinstance(config.get("model_routing", {}), dict) else {}
    mode, source = _first_non_empty(
        (cli_mode, "cli"),
        (environment.get("ECHO_MODEL_MODE"), "ECHO_MODEL_MODE"),
        (routing_config.get("mode"), "settings.json"),
        ("local", "default"),
    )
    mode = mode.lower()
    if mode not in SUPPORTED_MODEL_MODES:
        supported = ", ".join(SUPPORTED_MODEL_MODES)
        raise ValueError(f"Modo de modelo desconhecido: '{mode}'. Suportados: {supported}.")
    automatic_data = routing_config.get("automatic", {}) if isinstance(routing_config.get("automatic", {}), dict) else {}
    automatic = AutomaticRoutingConfig(
        claude_enabled=bool(automatic_data.get("claude_enabled", False)),
        daily_budget_usd=float(automatic_data.get("daily_budget_usd", 0.25)),
        max_single_call_estimated_usd=float(automatic_data.get("max_single_call_estimated_usd", 0.05)),
    )
    return ModelRoutingConfig(mode=mode, mode_source=source, automatic=automatic)


def _first_non_empty(*candidates: tuple[Any, str]) -> tuple[str, str]:
    for value, source in candidates:
        text = str(value or "").strip()
        if text:
            return text, source
    return "", "default"


def _complexity_reason_for_claude(routing_input: ModelRoutingInput) -> str:
    text = str(routing_input.user_message or "").lower()
    if _looks_like_professional_long_writing(text):
        return "professional_writing"
    if _looks_like_structured_summary(text):
        return "structured_summary"
    if "acao documental: review" in text or "ação documental: review" in text:
        return "document_review"
    if "acao documental: interpret" in text or "ação documental: interpret" in text:
        return "document_interpret"
    if "acao documental: rewrite" in text or "ação documental: rewrite" in text:
        return "document_rewrite"
    if _looks_like_document_synthesis(text):
        return "document_synthesis"
    if _looks_like_complex_planning(text):
        return "complex_planning"
    if _looks_like_technical_explanation(text):
        return "technical_explanation"
    if routing_input.prompt_chars >= 3500:
        return "long_prompt"
    complex_markers = (
        "texto longo",
        "reescreve",
        "reformula",
        "resume este documento",
        "resumo longo",
        "explica em detalhe",
        "explicacao tecnica",
        "explicação técnica",
        "compara",
        "comparacao",
        "comparação",
        "plano detalhado",
        "planeia",
        "passo a passo",
        "continuidade",
        # The same requests in English (en-GB is the default response language).
        "long text",
        "rewrite",
        "rephrase",
        "summarise this document",
        "summarize this document",
        "long summary",
        "explain in detail",
        "in detail",
        "technical explanation",
        "compare",
        "comparison",
        "detailed plan",
        "step by step",
        "step-by-step",
    )
    if any(marker in text for marker in complex_markers):
        return "complex_request"
    return ""


def _looks_like_document_synthesis(text: str) -> bool:
    markers = (
        "ficheiro lido da workspace",
        "conteudo extraido",
        "conteúdo extraído",
        "com base no ficheiro",
        "escreve um email",
        "cria um email",
    )
    return any(marker in text for marker in markers) and any(word in text for word in ("email", "mail", "resume", "resumo", "sintese", "síntese"))


def _with_routing_metrics(routing_input: ModelRoutingInput) -> ModelRoutingInput:
    user_chars = routing_input.user_message_chars or len(str(routing_input.user_message or ""))
    context_chars = max(0, int(routing_input.context_chars or 0))
    constraints = routing_input.constraint_count or _count_constraints(routing_input.user_message)
    prompt_chars = routing_input.prompt_chars or (user_chars + context_chars)
    return ModelRoutingInput(
        source=routing_input.source,
        user_message=routing_input.user_message,
        prompt_chars=prompt_chars,
        user_message_chars=user_chars,
        context_chars=context_chars,
        constraint_count=constraints,
        has_tools=routing_input.has_tools,
        has_context=routing_input.has_context,
        explicit_provider=routing_input.explicit_provider,
        explicit_model=routing_input.explicit_model,
        task_profile=routing_input.task_profile,
        privacy_class=routing_input.privacy_class,
        explicit_document_request=routing_input.explicit_document_request,
    )


def _extract_routing_user_message(user_message: str, history: list[dict[str, str]]) -> tuple[str, int]:
    text = str(user_message or "")
    match = re.search(r"(?:Mensagem do Alexandre|Alexandre's message):\s*(.*?)\s*\n\nInten", text, flags=re.IGNORECASE | re.DOTALL)
    routing_message = match.group(1).strip() if match else text.strip()
    context_chars = sum(len(str(item.get("content") or "")) for item in history)
    for heading in ("Contexto relevante:", "Factos relevantes:", "Próximo objetivo da conversa:", "Proximo objetivo da conversa:"):
        index = text.find(heading)
        if index >= 0:
            context_chars += max(0, len(text) - index)
            break
    return routing_message, context_chars


def _count_constraints(text: str) -> int:
    normalized = str(text or "").lower()
    markers = (
        "curta",
        "curto",
        "detalhado",
        "detalhada",
        "quatro pontos",
        "cinco pontos",
        "tres pontos",
        "três pontos",
        "portugues de portugal",
        "português de portugal",
        "sem ",
        "com ",
        "em pontos",
        "short",
        "detailed",
        "three points",
        "four points",
        "five points",
        "british english",
        "without ",
        "bullet points",
    )
    return sum(1 for marker in markers if marker in normalized)


def _looks_like_professional_long_writing(text: str) -> bool:
    writing = any(marker in text for marker in ("escreve", "redige", "cria", "prepara", "write", "draft", "compose", "prepare"))
    professional = any(marker in text for marker in ("email", "e-mail", "profissional", "relatorio", "relatório", "professional", "report"))
    detailed = any(marker in text for marker in (
        "detalhado", "estado atual", "progressos", "proximos passos", "próximos passos",
        "detailed", "current state", "progress", "next steps",
    ))
    return writing and professional and detailed


def _looks_like_structured_summary(text: str) -> bool:
    if not any(marker in text for marker in ("resume", "resumo", "sintetiza", "summarise", "summarize", "summary")):
        return False
    has_structure = any(marker in text for marker in ("pontos", "topicos", "tópicos", "ideias principais", "bullet", "points", "main ideas", "key ideas"))
    has_count = any(marker in text for marker in (" tres ", " três ", " quatro ", " cinco ", " 3 ", " 4 ", " 5 ", " three ", " four ", " five "))
    has_source_text = ":" in text or len(text.split()) >= 35
    return has_structure and (has_count or has_source_text)


def _looks_like_complex_planning(text: str) -> bool:
    planning = any(marker in text for marker in ("planeia", "planear", "plano", "organiza", "plan", "organise", "organize"))
    detailed = any(marker in text for marker in (
        "detalhado", "passo a passo", "semanal", "dias", "orcamento", "orçamento",
        "detailed", "in detail", "step by step", "step-by-step", "weekly", "days", "budget",
    ))
    return planning and detailed


def _looks_like_technical_explanation(text: str) -> bool:
    explanation = any(marker in text for marker in ("explica", "explica-me", "explicacao", "explicação", "explain", "explanation"))
    technical = any(marker in text for marker in (
        "python", "erro", "codigo", "código", "api", "git", "assíncrono", "assincrono", "error", "code", "async",
    ))
    return explanation and technical


def _estimate_prompt_cost_usd(prompt_chars: int) -> float:
    tokens = max(1, int(prompt_chars / 4))
    # Conservative Haiku estimate for pre-call budgeting. Real cost is recorded
    # afterwards from provider usage when available.
    return (tokens / 1_000_000) * 1.0 + (512 / 1_000_000) * 5.0


# Bands for _document_task_complexity_score. Deliberately simple additive
# weights (no ML/embeddings) -- tuned so a small first-time rewrite lands
# "medium" (local-first, see _document_task_decision), an iterative
# refinement lands "high" essentially always (its base weight alone clears
# the threshold), and a prior local validation failure pushes an otherwise-
# medium task toward "high" without automatically escalating every document
# task.
_DOCUMENT_COMPLEXITY_LOW_MAX = 15.0
_DOCUMENT_COMPLEXITY_MEDIUM_MAX = 40.0


def _document_task_complexity_score(profile: dict[str, object]) -> float:
    score = 0.0
    task_type = str(profile.get("task_type") or "")
    # Iterative refinement is inherently higher-stakes than a first pass:
    # it must preserve everything the first rewrite already got right while
    # visibly improving it, with no fresh original to fall back on.
    score += 45.0 if task_type == "document_refinement" else 10.0
    entity_count = int(profile.get("document_named_entity_count") or 0)
    list_count = int(profile.get("document_list_item_count") or 0)
    score += min(20.0, (entity_count + list_count) * 3.0)
    document_chars = int(profile.get("document_chars") or 0)
    score += min(15.0, document_chars / 300.0)
    if profile.get("document_previous_local_failure"):
        score += 20.0
    if str(profile.get("document_validation_failure_reason") or "") == "placeholder_detected":
        score += 10.0
    if int(profile.get("document_regeneration_attempt") or 1) >= 2:
        score += 10.0
    return score


def _document_task_complexity_band(score: float) -> str:
    if score <= _DOCUMENT_COMPLEXITY_LOW_MAX:
        return "low"
    if score <= _DOCUMENT_COMPLEXITY_MEDIUM_MAX:
        return "medium"
    return "high"
