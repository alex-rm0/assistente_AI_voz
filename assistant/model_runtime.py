from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assistant.anthropic_provider import PAID_CALL_CONFIRMATION_ENV
from assistant.model_provider import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OLLAMA_MODEL
from assistant.model_router import AutomaticRoutingConfig, ModelRouter, ModelRoutingConfig, ModelUsageBudget
from assistant.secret_storage import (
    ANTHROPIC_SECRET_NAME,
    NullSecretStorage,
    SecretStorage,
    resolve_anthropic_api_key,
    sanitize_secret_error,
)
from assistant.system_commands import SystemCommand, parse_system_command


SAFE_USER_SETTING_KEYS = {
    "model_routing_mode",
    "automatic_claude_enabled",
    "daily_budget_usd",
    "max_single_call_estimated_usd",
    "visual_preferences",
}


REASON_LABELS = {
    "professional_writing": "Professional writing",
    "structured_summary": "Structured summary",
    "technical_explanation": "Technical explanation",
    "project_memory_recall": "Project memory",
    "memory_recall": "Memory recall",
    "local_mode": "Local mode",
    "claude_mode": "Claude mode",
    "low_complexity": "Local response",
    "automatic_claude_disabled": "Cloud routing disabled",
    "paid_calls_disabled": "Cloud disabled",
    "paid_calls_not_confirmed": "Cloud disabled",
    "missing_api_key": "API key required",
    "secret_storage_unavailable": "Secure storage unavailable",
    "environment_key": "Environment key",
    "daily_budget_exceeded": "Daily budget reached",
    "single_call_budget_exceeded": "Call limit reached",
    "budget_state_unavailable": "Budget unavailable",
    "source_kept_local": "Local source",
    "local_tool": "Local tool",
    "social_fast_path": "Social fast path",
    "system_datetime": "System datetime",
    "system_status": "System status",
    "fast_route": "Fast route",
    "local_deterministic": "Local deterministic response",
    "document_task": "Document task",
    "complex_planning": "Complex planning",
    "complex_request": "Complex request",
    "document_synthesis": "Document synthesis",
    "long_prompt": "Long prompt",
}


@dataclass(frozen=True)
class UserModelPreferences:
    model_routing_mode: str = ""
    automatic_claude_enabled: bool | None = None
    daily_budget_usd: float | None = None
    max_single_call_estimated_usd: float | None = None
    model_routing_mode_source: str = ""
    automatic_claude_enabled_source: str = ""


class UserSettingsStore:
    """Local, ignored-by-Git settings for safe UI preferences only."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {key: data[key] for key in SAFE_USER_SETTING_KEYS if key in data}

    def save(self, data: dict[str, Any]) -> None:
        safe = {key: data[key] for key in SAFE_USER_SETTING_KEYS if key in data}
        safe.pop("api_key", None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")

    def preferences(self) -> UserModelPreferences:
        data = self.load()
        return UserModelPreferences(
            model_routing_mode=_safe_mode(data.get("model_routing_mode")),
            automatic_claude_enabled=_optional_bool(data.get("automatic_claude_enabled")),
            daily_budget_usd=_optional_positive_float(data.get("daily_budget_usd")),
            max_single_call_estimated_usd=_optional_positive_float(data.get("max_single_call_estimated_usd")),
            model_routing_mode_source="user_settings" if "model_routing_mode" in data else "",
            automatic_claude_enabled_source="user_settings" if "automatic_claude_enabled" in data else "",
        )

    def update(self, **changes: Any) -> dict[str, Any]:
        data = self.load()
        data.update({key: value for key, value in changes.items() if key in SAFE_USER_SETTING_KEYS})
        self.save(data)
        return self.load()


class ModelRuntimeBridge:
    """Control plane shared by the Qt controller and the model router."""

    def __init__(
        self,
        *,
        router: ModelRouter | None,
        budget: ModelUsageBudget | None,
        store: UserSettingsStore,
        cli_locked: bool = False,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
        env: dict[str, str] | None = None,
        secret_storage: SecretStorage | None = None,
        settings_source: str = "settings.json",
        model_routing_mode_source: str = "",
        automatic_claude_enabled_source: str = "",
        context_observer_state: str = "unknown",
    ) -> None:
        self.router = router
        self.budget = budget
        self.store = store
        self.cli_locked = cli_locked
        self.ollama_model = ollama_model
        self.anthropic_model = anthropic_model
        self.env = env if env is not None else os.environ
        self.secret_storage = secret_storage or NullSecretStorage()
        self.settings_source = settings_source
        self.model_routing_mode_source = model_routing_mode_source or (router.config.mode_source if router else "default")
        self.automatic_claude_enabled_source = automatic_claude_enabled_source or "settings.json"
        self.context_observer_state = context_observer_state

    def current_payload(self, *, state: str = "idle_local", note: str = "") -> dict[str, Any]:
        mode = self._mode()
        budget = self._budget_snapshot()
        reason_code = (
            "local_mode"
            if mode == "local"
            else "claude_mode"
            if mode == "claude"
            else "automatic_claude_disabled"
            if not self._automatic().claude_enabled
            else "low_complexity"
        )
        provider = "ollama" if mode in {"local", "automatic"} else "anthropic"
        model = self.anthropic_model if provider == "anthropic" else self.ollama_model
        return self._with_runtime_metadata(
            {
                "mode": mode,
                "configured_model_mode": mode,
                "configured_model_mode_source": self.model_routing_mode_source,
                "execution_path": "idle",
                "execution_provider": provider,
                "execution_model": "NONE",
                "mode_locked": self.cli_locked,
                "provider": provider,
                "model": "NONE",
                "base_provider": provider,
                "base_model": model,
                "reason_code": reason_code,
                "reason_label": human_reason_label(reason_code),
                "paid_call": provider == "anthropic",
                "latency_ms": 0,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_usd": 0.0,
                "daily_used_usd": budget["daily_used_usd"],
                "daily_budget_usd": self._automatic().daily_budget_usd,
                "max_single_call_estimated_usd": self._automatic().max_single_call_estimated_usd,
                "automatic_claude_enabled": self._automatic().claude_enabled,
                "llm_calls": 0,
                "source": "",
                "fallback_reason": "",
                "state": state,
                "note": note,
                "settings_source": self.settings_source,
                "context_observer_state": self.context_observer_state,
            },
            state=state,
        )

    def request_started_payload(self) -> dict[str, Any]:
        mode = self._mode()
        if mode == "claude":
            state = "thinking_cloud"
        elif mode == "automatic":
            state = "routing_automatic"
        else:
            state = "thinking_local"
        return self.current_payload(state=state)

    def telemetry_payload(self, telemetry: dict[str, Any] | None, *, state: str = "response_ready") -> dict[str, Any]:
        if not telemetry:
            return self.current_payload(state=state)
        budget = self._budget_snapshot()
        reason_code = str(
            telemetry.get("model_routing_reason_code")
            or _reason_from_selected_path(str(telemetry.get("selected_path") or ""))
            or ""
        )
        llm_calls = int(telemetry.get("llm_calls") or 0)
        configured_mode = str(telemetry.get("configured_model_mode") or telemetry.get("model_routing_mode") or self._mode())
        configured_source = str(telemetry.get("configured_model_mode_source") or telemetry.get("model_routing_mode_source") or self.model_routing_mode_source)
        provider = str(telemetry.get("execution_provider") or telemetry.get("model_routing_provider") or telemetry.get("provider") or "")
        model = str(telemetry.get("execution_model") or telemetry.get("model_routing_model") or telemetry.get("model") or "")
        execution_path = str(telemetry.get("execution_path") or _execution_path_from_telemetry(telemetry, llm_calls))
        paid_call = bool(telemetry.get("model_routing_paid_call"))
        if llm_calls <= 0:
            reason_code = _reason_from_selected_path(str(telemetry.get("selected_path") or ""))
            provider = str(telemetry.get("execution_provider") or _provider_for_local_path(str(telemetry.get("selected_path") or ""), provider))
            model = "NONE"
            execution_path = str(telemetry.get("execution_path") or _execution_path_from_telemetry(telemetry, llm_calls))
            paid_call = False
        return self._with_runtime_metadata(
            {
                "mode": configured_mode,
                "configured_model_mode": configured_mode,
                "configured_model_mode_source": configured_source,
                "execution_path": execution_path,
                "execution_provider": provider or "local",
                "execution_model": model or "NONE",
                "mode_locked": self.cli_locked,
                "provider": provider or "ollama",
                "model": model or self.ollama_model,
                "reason_code": reason_code,
                "reason_label": human_reason_label(reason_code),
                "paid_call": paid_call,
                "latency_ms": _latency_from_telemetry(telemetry),
                "input_tokens": telemetry.get("input_tokens"),
                "output_tokens": telemetry.get("output_tokens"),
                "estimated_cost_usd": float(telemetry.get("estimated_cost_usd") or 0.0),
                "daily_used_usd": budget["daily_used_usd"],
                "daily_budget_usd": self._automatic().daily_budget_usd,
                "max_single_call_estimated_usd": self._automatic().max_single_call_estimated_usd,
                "automatic_claude_enabled": self._automatic().claude_enabled,
                "llm_calls": llm_calls,
                "source": _first_call_source(telemetry),
                "fallback_reason": str(telemetry.get("model_routing_fallback_reason") or ""),
                "state": state,
                "selected_path": str(telemetry.get("selected_path") or ""),
                "response_source": str(telemetry.get("response_source") or ""),
                "provider_error_type": str(telemetry.get("provider_error_type") or ""),
                "configuration_error": str(telemetry.get("configuration_error") or ""),
                "settings_source": self.settings_source,
                "context_observer_state": self.context_observer_state,
            },
            state=state,
        )

    def error_payload(self, message: str, *, provider_error_type: str = "") -> dict[str, Any]:
        payload = self.current_payload(state="error", note=message)
        payload["provider_error_type"] = provider_error_type
        payload["reason_code"] = provider_error_type or payload["reason_code"]
        payload["reason_label"] = human_reason_label(payload["reason_code"])
        payload["configuration_error"] = message
        payload["active_runtime_state"] = "error"
        return payload

    def execute_command(self, payload: dict[str, Any] | str | SystemCommand, *, source: str = "ui") -> dict[str, Any]:
        command = payload if isinstance(payload, SystemCommand) else parse_system_command(payload, source=source)
        intent = command.intent
        params = command.parameters
        if intent == "set_model_mode":
            return self.set_mode(str(params.get("mode") or ""))
        if intent == "set_automatic_claude_enabled":
            return self.set_automatic_claude_enabled(bool(params.get("enabled")))
        if intent == "set_model_budget":
            return self.set_budget(float(params.get("daily_budget_usd") or 0.0), float(params.get("max_single_call_estimated_usd") or 0.0))
        if intent == "get_budget_status" or intent == "get_model_status":
            return self.current_payload(state="system_status")
        if intent == "save_anthropic_key":
            return self.save_anthropic_key(str(params.get("api_key") or ""))
        if intent == "remove_anthropic_key":
            return self.remove_anthropic_key()
        if intent == "test_anthropic_connection":
            return self.test_anthropic_connection()
        return self.error_payload("Comando de sistema desconhecido.", provider_error_type="unknown_system_command")

    def system_status_answer(self, user_message: str) -> str | None:
        text = _normalize(user_message)
        if not _looks_like_system_status_question(text):
            return None
        payload = self.current_payload(state="system_status")
        if "configuro" in text or "configurar" in text or "claude" in text and "como" in text:
            return (
                "Para configurares o Claude, abre o painel lateral, vai a Models e usa a área Anthropic API Key. "
                "A chave não é guardada no ficheiro de configuração nem enviada para a conversa."
            )
        if "api key" in text or "chave" in text:
            source = str(payload["api_key_source"])
            if payload["api_key_configured"] and source == "environment":
                return "Sim, a chave da Anthropic está configurada pelo ambiente."
            if payload["api_key_configured"]:
                return "Sim, a chave da Anthropic está configurada em armazenamento seguro."
            return "Ainda não tens uma chave da Anthropic configurada."
        if "gastei" in text or "custo" in text:
            return f"Hoje tens {payload['daily_used_usd']:.3f} dólares registados em chamadas pagas."
        if "orcamento" in text or "orçamento" in text:
            return f"O orçamento diário está em {payload['daily_budget_usd']:.2f} dólares."
        if "bloqueado" in text or "linha de comandos" in text or "cli" in text:
            return "Sim, o modo está bloqueado por CLI." if self.cli_locked else "Não, o modo não está bloqueado pela linha de comandos."
        if "automatic" in text or "automatico" in text or "automático" in text:
            state_label = "ligado" if payload["automatic_claude_enabled"] else "desligado"
            return f"O encaminhamento automático para Claude está {state_label}."
        if "modo" in text:
            return f"Estou em modo {payload['mode']}."
        if "modelo" in text:
            return f"Estou a usar {payload.get('base_model') or payload['model']} através de {payload.get('base_provider') or payload['provider']}."
        return None

    def save_anthropic_key(self, value: str) -> dict[str, Any]:
        clean = str(value or "").strip()
        if not clean:
            return self.error_payload("API KEY NOT CONFIGURED", provider_error_type="missing_api_key")
        if not self._secret_storage_available():
            return self.error_payload("SECURE STORAGE UNAVAILABLE", provider_error_type="secret_storage_unavailable")
        try:
            self.secret_storage.set_secret(ANTHROPIC_SECRET_NAME, clean)
        except Exception as exc:
            return self.error_payload(sanitize_secret_error(exc), provider_error_type="secret_storage_unavailable")
        return self.current_payload(state="configuration_changed", note="Chave guardada em armazenamento seguro.")

    def remove_anthropic_key(self) -> dict[str, Any]:
        if self._anthropic_key_source() == "environment":
            return self.error_payload("A chave vem do ambiente e não pode ser removida pela interface.", provider_error_type="environment_key")
        if not self._secret_storage_available():
            return self.error_payload("SECURE STORAGE UNAVAILABLE", provider_error_type="secret_storage_unavailable")
        self.secret_storage.delete_secret(ANTHROPIC_SECRET_NAME)
        return self.current_payload(state="configuration_changed", note="Chave removida.")

    def test_anthropic_connection(self) -> dict[str, Any]:
        if not self._anthropic_key_configured():
            return self.error_payload("API KEY NOT CONFIGURED", provider_error_type="missing_api_key")
        if not self._paid_calls_enabled():
            return self.error_payload("PAID CALLS DISABLED", provider_error_type="paid_calls_not_confirmed")
        return self.current_payload(
            state="configuration_status",
            note="Configuração pronta. O teste real de ligação não foi executado nesta validação.",
        )

    def set_mode(self, mode: str) -> dict[str, Any]:
        clean = _safe_mode(mode)
        if not clean:
            return self.error_payload("Modo inválido.", provider_error_type="invalid_mode")
        if self.cli_locked:
            return self.error_payload("O modo foi definido por CLI e está bloqueado nesta sessão.", provider_error_type="cli_locked")
        if self.router is None:
            return self.error_payload("O seletor de modelos não está disponível nesta sessão.", provider_error_type="router_unavailable")
        if clean == "claude":
            error = self._claude_configuration_error()
            if error:
                return self.error_payload(error[0], provider_error_type=error[1])
        self._apply_config(mode=clean)
        self.model_routing_mode_source = "ui"
        self.store.update(model_routing_mode=clean)
        return self.current_payload(state="mode_changed", note=f"Modo {clean} ativo.")

    def set_automatic_claude_enabled(self, enabled: bool) -> dict[str, Any]:
        if self.cli_locked:
            return self.error_payload("O modo foi definido por CLI e está bloqueado nesta sessão.", provider_error_type="cli_locked")
        self._apply_config(automatic_claude_enabled=bool(enabled))
        self.automatic_claude_enabled_source = "ui"
        self.store.update(automatic_claude_enabled=bool(enabled))
        note = "Claude automático ativo." if enabled else "Claude automático desligado."
        return self.current_payload(state="mode_changed", note=note)

    def set_budget(self, daily_budget_usd: float, max_single_call_estimated_usd: float) -> dict[str, Any]:
        daily = _bounded_money(daily_budget_usd, 0.0, 25.0)
        single = _bounded_money(max_single_call_estimated_usd, 0.0, 5.0)
        if single > daily and daily > 0:
            single = daily
        self._apply_config(daily_budget_usd=daily, max_single_call_estimated_usd=single)
        self.store.update(daily_budget_usd=daily, max_single_call_estimated_usd=single)
        return self.current_payload(state="mode_changed", note="Orçamento atualizado.")

    def apply_user_preferences(self) -> None:
        if self.router is None or self.cli_locked:
            return
        preferences = self.store.preferences()
        self._apply_config(
            mode=preferences.model_routing_mode or None,
            automatic_claude_enabled=preferences.automatic_claude_enabled,
            daily_budget_usd=preferences.daily_budget_usd,
            max_single_call_estimated_usd=preferences.max_single_call_estimated_usd,
        )
        if preferences.model_routing_mode_source:
            self.model_routing_mode_source = preferences.model_routing_mode_source
        if preferences.automatic_claude_enabled_source:
            self.automatic_claude_enabled_source = preferences.automatic_claude_enabled_source

    def get_anthropic_api_key(self) -> str | None:
        value, _source = resolve_anthropic_api_key(self.env, self.secret_storage)
        return value or None

    def _mode(self) -> str:
        if self.router is None:
            return "local"
        return self.router.config.mode

    def _automatic(self) -> AutomaticRoutingConfig:
        if self.router is None:
            return AutomaticRoutingConfig()
        return self.router.config.automatic

    def _apply_config(
        self,
        *,
        mode: str | None = None,
        automatic_claude_enabled: bool | None = None,
        daily_budget_usd: float | None = None,
        max_single_call_estimated_usd: float | None = None,
    ) -> None:
        if self.router is None:
            return
        current = self.router.config
        automatic = current.automatic
        self.router.config = ModelRoutingConfig(
            mode=_safe_mode(mode) or current.mode,
            mode_source="ui" if mode else current.mode_source,
            automatic=AutomaticRoutingConfig(
                claude_enabled=automatic.claude_enabled if automatic_claude_enabled is None else bool(automatic_claude_enabled),
                daily_budget_usd=automatic.daily_budget_usd if daily_budget_usd is None else float(daily_budget_usd),
                max_single_call_estimated_usd=(
                    automatic.max_single_call_estimated_usd
                    if max_single_call_estimated_usd is None
                    else float(max_single_call_estimated_usd)
                ),
            ),
        )

    def _with_runtime_metadata(self, payload: dict[str, Any], *, state: str) -> dict[str, Any]:
        payload.setdefault("configured_model_mode", payload.get("mode") or self._mode())
        payload.setdefault("configured_model_mode_source", self.model_routing_mode_source)
        payload.setdefault("execution_path", "none")
        payload.setdefault("execution_provider", payload.get("provider") or "none")
        payload.setdefault("execution_model", payload.get("model") or "NONE")
        payload["model_routing_mode_source"] = self.model_routing_mode_source
        payload["automatic_claude_enabled_source"] = self.automatic_claude_enabled_source
        payload["api_key_configured"] = self._anthropic_key_configured()
        payload["api_key_source"] = self._anthropic_key_source()
        payload["secret_storage_available"] = self._secret_storage_available()
        payload["paid_calls_enabled"] = self._paid_calls_enabled()
        payload.setdefault("configuration_error", "")
        payload["active_runtime_state"] = state
        return payload

    def _claude_configuration_error(self) -> tuple[str, str] | None:
        if not self._anthropic_key_configured():
            return "CLAUDE UNAVAILABLE: API KEY NOT CONFIGURED", "missing_api_key"
        if not self._paid_calls_enabled():
            return "PAID CALLS DISABLED", "paid_calls_not_confirmed"
        return None

    def _anthropic_key_configured(self) -> bool:
        return self._anthropic_key_source() != "none"

    def _anthropic_key_source(self) -> str:
        _value, source = resolve_anthropic_api_key(self.env, self.secret_storage)
        return source

    def _secret_storage_available(self) -> bool:
        try:
            return bool(self.secret_storage.is_available())
        except Exception:
            return False

    def _paid_calls_enabled(self) -> bool:
        return str(self.env.get(PAID_CALL_CONFIRMATION_ENV) or "").strip().lower() == "true"

    def _budget_snapshot(self) -> dict[str, float]:
        if self.budget is None:
            return {"daily_used_usd": 0.0}
        snapshot = self.budget.snapshot()
        if not snapshot:
            return {"daily_used_usd": 0.0}
        return {"daily_used_usd": float(snapshot.get("accumulated_estimated_cost_usd") or 0.0)}


def human_reason_label(reason_code: str) -> str:
    code = str(reason_code or "").strip()
    if not code:
        return "Local response"
    return REASON_LABELS.get(code, code.replace("_", " ").strip().title())


def user_preferences_for_routing(settings_path: Path) -> UserModelPreferences:
    return UserSettingsStore(settings_path).preferences()


def _safe_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"local", "claude", "automatic"} else ""


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _bounded_money(value: Any, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lower
    return min(max(number, lower), upper)


def _first_call_source(telemetry: dict[str, Any]) -> str:
    details = telemetry.get("llm_call_details")
    if isinstance(details, list) and details:
        first = details[0]
        if isinstance(first, dict):
            return str(first.get("component") or "")
    sources = telemetry.get("llm_call_sources")
    if isinstance(sources, list) and sources:
        return str(sources[0] or "")
    return ""


def _latency_from_telemetry(telemetry: dict[str, Any]) -> float:
    records = telemetry.get("llm_call_tokens")
    if isinstance(records, list) and records:
        return sum(float(record.get("latency_ms") or 0.0) for record in records if isinstance(record, dict))
    return 0.0


def _reason_from_selected_path(selected_path: str) -> str:
    path = selected_path.upper()
    if "SOCIAL" in path:
        return "social_fast_path"
    if "SYSTEM_DATETIME" in path:
        return "system_datetime"
    if "MEMORY" in path:
        return "memory_recall"
    if "TOOL" in path or "FAST_ROUTE" in path:
        return "local_tool"
    if "PROJECT" in path:
        return "project_memory_recall"
    return "low_complexity"


def _execution_path_from_telemetry(telemetry: dict[str, Any], llm_calls: int) -> str:
    if llm_calls > 0:
        source = str(telemetry.get("response_source") or "").upper()
        selected = str(telemetry.get("selected_path") or "").upper()
        if "AGENT" in source or selected == "AGENT":
            return "agent"
        if "RESPONSE_COMPOSER" in source or "COMPOSER" in source:
            return "response_composer"
        return "llm"
    selected = str(telemetry.get("selected_path") or "").upper()
    if selected == "SOCIAL_PATH":
        return "social_fast_path"
    if selected == "SYSTEM_DATETIME":
        return "system_datetime"
    if "MEMORY" in selected:
        return "memory_recall"
    if selected == "DOCUMENT_TASK":
        return "document_task"
    if "TOOL" in selected or "FAST" in selected:
        return "local_tool"
    if "ERROR" in selected:
        return "error"
    return _reason_from_selected_path(selected)


def _provider_for_local_path(selected_path: str, provider: str) -> str:
    path = selected_path.upper()
    if "SOCIAL" in path or "SYSTEM_DATETIME" in path:
        return "local"
    if "MEMORY" in path:
        return "memory"
    if "TOOL" in path or "FAST" in path or "DOCUMENT" in path:
        return "local_tool"
    return provider or "local"


def _normalize(text: str) -> str:
    return str(text or "").strip().lower()


def _looks_like_system_status_question(text: str) -> bool:
    markers = (
        "que modelo estas a usar",
        "que modelo estás a usar",
        "em que modo estas",
        "em que modo estás",
        "como configuro o claude",
        "tens uma api key configurada",
        "api key configurada",
        "quanto ja gastei hoje",
        "quanto já gastei hoje",
        "qual e o orcamento diario",
        "qual é o orçamento diário",
        "orcamento diario",
        "orçamento diário",
        "modo esta bloqueado",
        "modo está bloqueado",
        "linha de comandos",
        "claude automatico",
        "claude automático",
        "routing automatico",
        "routing automático",
    )
    return any(marker in text for marker in markers)
