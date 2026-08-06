"""Anthropic ModelProvider implementation.

Network calls are deliberately gated by ECHO_ALLOW_PAID_MODEL_CALLS=true so a
configured API key is not enough to spend money by accident.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import requests

from assistant.model_provider import ModelResponse, ProviderConfigurationError, estimate_cost


ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"
PAID_CALL_CONFIRMATION_ENV = "ECHO_ALLOW_PAID_MODEL_CALLS"


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_getter: Callable[[], str | None] | None = None,
        base_url: str = "https://api.anthropic.com",
        timeout_seconds: int = 120,
        max_tokens: int = 1024,
        allow_paid_calls: bool | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self.api_key_getter = api_key_getter
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.allow_paid_calls = (
            allow_paid_calls
            if allow_paid_calls is not None
            else os.environ.get(PAID_CALL_CONFIRMATION_ENV, "").strip().lower() == "true"
        )

    @property
    def name(self) -> str:
        return "anthropic"

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        response_format: dict | str | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> ModelResponse:
        # num_predict is Ollama-specific (options.num_predict); Anthropic has
        # no equivalent wired up yet, so it's accepted-and-ignored here only
        # to keep the ModelProvider interface uniform across providers.
        resolved_model = model or self.model
        api_key = str(self.api_key_getter() if self.api_key_getter is not None else self.api_key or "").strip()
        if not api_key:
            raise ProviderConfigurationError(
                "O provider Anthropic esta selecionado, mas falta configurar ANTHROPIC_API_KEY.",
                provider=self.name,
                provider_error_type="missing_api_key",
            )
        if not self.allow_paid_calls:
            raise ProviderConfigurationError(
                "O provider Anthropic esta selecionado, mas as chamadas pagas estao bloqueadas. "
                f"Define {PAID_CALL_CONFIRMATION_ENV}=true apenas quando quiseres executar um teste pago.",
                provider=self.name,
                provider_error_type="paid_calls_not_confirmed",
            )

        system_prompt, anthropic_messages = _split_system_prompt(messages)
        payload: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        started_at = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}{ANTHROPIC_MESSAGES_PATH}",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            raise RuntimeError("A chamada Anthropic excedeu o tempo limite configurado.") from exc
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 429:
                raise RuntimeError("A Anthropic devolveu rate limit. Tenta novamente mais tarde.") from exc
            raise RuntimeError(f"A Anthropic devolveu um erro HTTP{f' {status}' if status else ''}.") from exc
        except requests.RequestException as exc:
            raise RuntimeError("Nao consegui ligar a Anthropic.") from exc
        except ValueError as exc:
            raise RuntimeError("A Anthropic devolveu uma resposta invalida.") from exc

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        text = _extract_text(data)
        if not text:
            raise RuntimeError("A Anthropic nao devolveu texto para esta mensagem.")
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        input_tokens = _int_or_none(usage.get("input_tokens"))
        output_tokens = _int_or_none(usage.get("output_tokens"))
        return ModelResponse(
            text=text,
            provider=self.name,
            model=str(data.get("model") or resolved_model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            estimated_cost_usd=estimate_cost(self.name, resolved_model, input_tokens, output_tokens),
            raw=data,
        )


def _split_system_prompt(messages: list[dict]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    output: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "system":
            if content.strip():
                system_parts.append(content)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        output.append({"role": role, "content": content})
    return "\n\n".join(system_parts).strip(), output


def _extract_text(data: dict[str, Any]) -> str:
    content = data.get("content", [])
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None
