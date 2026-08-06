"""Controlled model routing for Echo.

This module decides which configured provider may answer an LLM call. It never
executes tools, never reads prompts from disk and never stores prompt text in
the budget file.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
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
)


SUPPORTED_MODEL_MODES = ("local", "claude", "automatic")
NO_PAID_CALL_SOURCES = {
    "TOOL_SELECTOR",
    "MEMORY_SUMMARY",
    "SESSION_SUMMARY",
    "VOICE_CRITIC",
    "PLANNER",
}


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
    ) -> None:
        self.config = config
        self.ollama_model = ollama_model
        self.anthropic_model = anthropic_model
        self.budget = budget
        self.env = env if env is not None else os.environ
        self.anthropic_key_available = anthropic_key_available

    def decide(self, routing_input: ModelRoutingInput) -> ModelRoutingDecision:
        mode = self.config.mode
        routing_input = _with_routing_metrics(routing_input)
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
        if not self.config.automatic.claude_enabled:
            return self._ollama("automatic", "automatic_claude_disabled", "Claude automatico esta desligado.", routing_input=routing_input)
        if not self._has_anthropic_key():
            return self._ollama(
                "automatic",
                "missing_api_key",
                "Sem ANTHROPIC_API_KEY; fica no modelo local.",
                "missing_api_key",
                routing_input=routing_input,
            )
        if self.env.get(PAID_CALL_CONFIRMATION_ENV, "").strip().lower() != "true":
            return self._ollama(
                "automatic",
                "paid_calls_not_confirmed",
                f"Sem {PAID_CALL_CONFIRMATION_ENV}=true; fica no modelo local.",
                "paid_calls_not_confirmed",
                routing_input=routing_input,
            )
        complexity_reason = _complexity_reason_for_claude(routing_input)
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

    def mark_next_call_source(self, source: str) -> None:
        self._next_call_source = str(source or "").strip()

    def chat(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        response_format: str | None = None,
        source: str | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt or self.system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_message})
        source_name = str(source or self._next_call_source or "OTHER").strip() or "OTHER"
        self._next_call_source = ""
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
            )
        )
        provider = self.providers.get(decision.provider)
        if provider is None:
            raise ProviderConfigurationError(
                f"O provider {decision.provider} foi escolhido, mas nao esta configurado.",
                provider=decision.provider,
                provider_error_type="provider_not_configured",
            )
        self.provider = provider
        result = provider.chat(messages, model=decision.model, response_format=response_format)
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
            model_routing_provider=decision.provider if decision else provider,
            model_routing_model=decision.model if decision else model,
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
    )


def _extract_routing_user_message(user_message: str, history: list[dict[str, str]]) -> tuple[str, int]:
    text = str(user_message or "")
    match = re.search(r"Mensagem do Alexandre:\s*(.*?)\s*\n\nInten", text, flags=re.IGNORECASE | re.DOTALL)
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
    )
    return sum(1 for marker in markers if marker in normalized)


def _looks_like_professional_long_writing(text: str) -> bool:
    writing = any(marker in text for marker in ("escreve", "redige", "cria", "prepara"))
    professional = any(marker in text for marker in ("email", "e-mail", "profissional", "relatorio", "relatório"))
    detailed = any(marker in text for marker in ("detalhado", "estado atual", "progressos", "proximos passos", "próximos passos"))
    return writing and professional and detailed


def _looks_like_structured_summary(text: str) -> bool:
    if not any(marker in text for marker in ("resume", "resumo", "sintetiza")):
        return False
    has_structure = any(marker in text for marker in ("pontos", "topicos", "tópicos", "ideias principais", "bullet"))
    has_count = any(marker in text for marker in (" tres ", " três ", " quatro ", " cinco ", " 3 ", " 4 ", " 5 "))
    has_source_text = ":" in text or len(text.split()) >= 35
    return has_structure and (has_count or has_source_text)


def _looks_like_complex_planning(text: str) -> bool:
    planning = any(marker in text for marker in ("planeia", "planear", "plano", "organiza"))
    detailed = any(marker in text for marker in ("detalhado", "passo a passo", "semanal", "dias", "orcamento", "orçamento"))
    return planning and detailed


def _looks_like_technical_explanation(text: str) -> bool:
    explanation = any(marker in text for marker in ("explica", "explica-me", "explicacao", "explicação"))
    technical = any(marker in text for marker in ("python", "erro", "codigo", "código", "api", "git", "assíncrono", "assincrono"))
    return explanation and technical


def _estimate_prompt_cost_usd(prompt_chars: int) -> float:
    tokens = max(1, int(prompt_chars / 4))
    # Conservative Haiku estimate for pre-call budgeting. Real cost is recorded
    # afterwards from provider usage when available.
    return (tokens / 1_000_000) * 1.0 + (512 / 1_000_000) * 5.0
