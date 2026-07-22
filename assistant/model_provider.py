"""Minimal model-provider abstraction (Part 3 of the stabilize/evals/provider task).

This is a NEW, standalone seam — it does not replace assistant.llm.OllamaClient,
which keeps powering the production app exactly as before (app.py is
untouched). Only assistant/evals uses this module today, via
ProviderBackedLLM, to make the eval runner's `--provider`/`--model` flags
real instead of decorative.

Adding a future provider (Anthropic, OpenAI) means: implement ModelProvider,
add one line to evals/harness.py's provider registry. Nothing in memory,
routing, tools, or the UI needs to change — they only ever see the
duck-typed `chat(...)`/`choose_tool(...)`/`embed(...)` interface via
ProviderBackedLLM, exactly like they see assistant.llm.OllamaClient today.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests


@dataclass
class ModelResponse:
    text: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
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


# A pricing table, not a pricing call: no network, no API keys, just a place
# for $/1M-token rates to live once a paid provider exists. Ollama is local
# and always free, hence the empty default table.
PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {}


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
    """ModelProvider implementation over Ollama's /api/chat."""

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
                f"Não consegui ligar ao Ollama (modelo '{resolved_model}'). "
                "Confirma que o Ollama está aberto e que o modelo está instalado."
            ) from exc
        except ValueError as exc:
            raise RuntimeError("O Ollama devolveu uma resposta inválida.") from exc

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        message = data.get("message", {})
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("O Ollama não devolveu texto para esta mensagem.")

        return ModelResponse(
            text=text.strip(),
            provider=self.name,
            model=resolved_model,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_ms=elapsed_ms,
            raw=data,
        )


class ProviderBackedLLM:
    """Adapts any ModelProvider to the duck-typed llm interface AssistantEngine,
    ResponseComposer, VoiceCritic and Agent already depend on:
    chat(user_message, history=, system_prompt=, response_format=) -> str,
    plus choose_tool(...) and embed(...).

    This is the ONLY place that translates between the two shapes — memory,
    routing, tools and the UI keep depending on the old shape unchanged.
    """

    def __init__(self, provider: ModelProvider, system_prompt: str = "") -> None:
        self.provider = provider
        self.system_prompt = system_prompt
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
        # Read by AssistantEngine._llm_chat_count() (see conversation.py's
        # _single_client_chat_count) so evals telemetry (llm_calls) works
        # the same way whether the engine is backed by OllamaClient or by
        # a ProviderBackedLLM.
        return len(self.responses)

    @property
    def chat_call_sources(self) -> list[str]:
        return list(self._call_sources)

    @property
    def chat_call_tokens(self) -> list[dict[str, Any]]:
        return [
            {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "latency_ms": r.latency_ms}
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
        return None

    @property
    def settings(self) -> "_CompatSettings":
        # AssistantEngine._model_name() reads llm.settings.model for the
        # "model" telemetry field — this mirrors OllamaClient's shape just
        # enough for that one read, without pretending to be OllamaClient.
        return _CompatSettings(model=getattr(self.provider, "model", ""), model_source=f"provider:{self.provider.name}")


@dataclass
class _CompatSettings:
    model: str
    model_source: str = "provider"
