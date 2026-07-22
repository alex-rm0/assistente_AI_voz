"""Runtime model-provider abstraction for Echo.

The runtime talks to this seam instead of binding itself directly to one model
provider. Memory, routing, tools and UI keep seeing the existing duck-typed LLM
shape through ProviderBackedLLM.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests


DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
SUPPORTED_MODEL_PROVIDERS = ("ollama", "anthropic")


class ProviderConfigurationError(RuntimeError):
    """Raised when the selected provider cannot be used safely as configured."""

    def __init__(self, message: str, *, provider: str, provider_error_type: str) -> None:
        super().__init__(message)
        self.provider = provider
        self.provider_error_type = provider_error_type


@dataclass
class ModelResponse:
    text: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    estimated_cost_usd: float = 0.0
    raw: object | None = None


class ModelProvider(Protocol):
    @property
    def name(self) -> str: ...

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        response_format: dict | str | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse: ...


PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "anthropic:claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "anthropic:claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "anthropic:claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "anthropic:claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "anthropic:claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "anthropic:claude-opus-4-8": {"input": 5.0, "output": 25.0},
}


@dataclass(frozen=True)
class ResolvedModelProvider:
    provider: str
    provider_source: str
    model: str
    model_source: str
    base_url: str
    timeout_seconds: int


def resolve_model_provider(
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    env: dict[str, str] | None = None,
    settings: dict[str, Any] | None = None,
) -> ResolvedModelProvider:
    """Resolve provider/model with explicit priority and legacy compatibility."""
    environment = env if env is not None else os.environ
    config = settings or {}
    model_config = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    ollama_config = config.get("ollama", {}) if isinstance(config.get("ollama", {}), dict) else {}
    anthropic_config = config.get("anthropic", {}) if isinstance(config.get("anthropic", {}), dict) else {}

    provider, provider_source = _first_non_empty(
        (cli_provider, "cli"),
        (environment.get("ECHO_MODEL_PROVIDER"), "ECHO_MODEL_PROVIDER"),
        (model_config.get("provider"), "settings.json"),
        (DEFAULT_MODEL_PROVIDER, "default"),
    )
    provider = provider.lower()
    if provider not in SUPPORTED_MODEL_PROVIDERS:
        supported = ", ".join(SUPPORTED_MODEL_PROVIDERS)
        raise ValueError(f"Provider de modelo desconhecido: '{provider}'. Suportados: {supported}.")

    if provider == "ollama":
        model, model_source = _first_non_empty(
            (cli_model, "cli"),
            (environment.get("ECHO_MODEL_NAME"), "ECHO_MODEL_NAME"),
            (environment.get("OLLAMA_MODEL"), "OLLAMA_MODEL"),
            (model_config.get("name"), "settings.json"),
            (ollama_config.get("model"), "settings.json:ollama"),
            (DEFAULT_OLLAMA_MODEL, "default"),
        )
        base_url = str(ollama_config.get("base_url") or "http://127.0.0.1:11434")
        timeout_seconds = int(ollama_config.get("timeout_seconds", 120))
    else:
        model, model_source = _first_non_empty(
            (cli_model, "cli"),
            (environment.get("ECHO_MODEL_NAME"), "ECHO_MODEL_NAME"),
            (model_config.get("name"), "settings.json"),
            (anthropic_config.get("model"), "settings.json:anthropic"),
            (DEFAULT_ANTHROPIC_MODEL, "default"),
        )
        base_url = str(anthropic_config.get("base_url") or "https://api.anthropic.com")
        timeout_seconds = int(anthropic_config.get("timeout_seconds", 120))

    return ResolvedModelProvider(
        provider=provider,
        provider_source=provider_source,
        model=model,
        model_source=model_source,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )


def _first_non_empty(*candidates: tuple[Any, str]) -> tuple[str, str]:
    for value, source in candidates:
        text = str(value or "").strip()
        if text:
            return text, source
    return "", "default"


def estimate_cost(provider: str, model: str, input_tokens: int | None, output_tokens: int | None) -> float:
    if provider == "ollama":
        return 0.0
    prices = PRICING_PER_MILLION_TOKENS.get(f"{provider}:{model}")
    if not prices or input_tokens is None or output_tokens is None:
        return 0.0
    return (input_tokens / 1_000_000) * prices.get("input", 0.0) + (output_tokens / 1_000_000) * prices.get(
        "output", 0.0
    )


class OllamaProvider:
    """ModelProvider implementation over Ollama's /api/chat and /api/embeddings."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout_seconds: int = 120) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "ollama"

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        response_format: dict | str | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        resolved_model = model or self.model
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {"model": resolved_model, "messages": messages, "stream": False}
        if response_format is not None:
            payload["format"] = response_format
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        started_at = time.perf_counter()
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Nao consegui ligar ao Ollama (modelo '{resolved_model}'). "
                "Confirma que o Ollama esta aberto e que o modelo esta instalado."
            ) from exc
        except ValueError as exc:
            raise RuntimeError("O Ollama devolveu uma resposta invalida.") from exc

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        message = data.get("message", {})
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("O Ollama nao devolveu texto para esta mensagem.")

        return ModelResponse(
            text=text.strip(),
            provider=self.name,
            model=resolved_model,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_ms=elapsed_ms,
            estimated_cost_usd=0.0,
            raw=data,
        )

    def embed(self, text: str) -> list[float] | None:
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        payload: dict[str, Any] = {"model": self.model, "prompt": text}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return None

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            return None
        return [float(item) for item in embedding if isinstance(item, (int, float))]


class ProviderBackedLLM:
    """Adapt a ModelProvider to the LLM interface used by AssistantEngine."""

    def __init__(self, provider: ModelProvider, system_prompt: str = "", model_source: str = "provider") -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.model_source = model_source
        self.last_response: ModelResponse | None = None
        self.responses: list[ModelResponse] = []
        self._call_sources: list[str] = []

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
        result = self.provider.chat(messages, response_format=response_format)
        self.last_response = result
        self.responses.append(result)
        self._call_sources.append(str(source or "OTHER"))
        return result.text

    @property
    def chat_call_count(self) -> int:
        return len(self.responses)

    @property
    def chat_call_sources(self) -> list[str]:
        return list(self._call_sources)

    @property
    def chat_call_tokens(self) -> list[dict[str, Any]]:
        return [
            {
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
                "estimated_cost_usd": r.estimated_cost_usd,
                "provider": r.provider,
                "model": r.model,
            }
            for r in self.responses
        ]

    def choose_tool(
        self,
        user_message: str,
        tools_description: str,
        profile_name: str | None = None,
        active_contexts: list[str] | None = None,
    ) -> dict:
        context_label = ", ".join(active_contexts or []) or profile_name or "desconhecido"
        system_prompt = (
            "Es um seletor de ferramentas para o AssistenteIA.\n"
            "A tua unica tarefa e decidir se a mensagem precisa de uma ferramenta de ficheiros.\n"
            f"Contextos ativos da conversa: {context_label}.\n"
            "Responde apenas em JSON valido, sem markdown.\n"
            "Em caso de duvida, responde sempre com {\"tool\": null}.\n\n"
            "Formato quando usar ferramenta:\n"
            '{"tool": "nome_da_ferramenta", "arguments": {"chave": "valor"}, "reason": "motivo curto"}\n\n'
            "Formato quando nao usar ferramenta:\n"
            '{"tool": null, "arguments": {}, "reason": "motivo curto"}\n\n'
            f"Ferramentas disponiveis:\n{tools_description}\n"
        )
        raw_response = self.chat(user_message, history=[], system_prompt=system_prompt, response_format="json")
        try:
            decision = json.loads(raw_response)
        except json.JSONDecodeError:
            return {"tool": None, "arguments": {}}
        if not isinstance(decision, dict):
            return {"tool": None, "arguments": {}}
        tool_name = decision.get("tool")
        arguments = decision.get("arguments", {})
        reason = decision.get("reason", "")
        if tool_name is not None and not isinstance(tool_name, str):
            tool_name = None
        if not isinstance(arguments, dict):
            arguments = {}
        if not isinstance(reason, str):
            reason = ""
        return {"tool": tool_name, "arguments": arguments, "reason": reason}

    def embed(self, text: str):
        embed = getattr(self.provider, "embed", None)
        if not callable(embed):
            return None
        return embed(text)

    @property
    def settings(self) -> "_CompatSettings":
        return _CompatSettings(model=getattr(self.provider, "model", ""), model_source=self.model_source)


@dataclass
class _CompatSettings:
    model: str
    model_source: str = "provider"
