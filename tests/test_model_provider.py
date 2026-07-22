from __future__ import annotations

import json

import pytest

from assistant.model_provider import ModelResponse, OllamaProvider, ProviderBackedLLM, estimate_cost


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def test_ollama_provider_parses_response_and_tokens(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):
        assert url.endswith("/api/chat")
        assert json["model"] == "llama3.1:8b"
        return FakeResponse(
            {
                "message": {"content": " Olá! "},
                "prompt_eval_count": 12,
                "eval_count": 7,
            }
        )

    monkeypatch.setattr("assistant.model_provider.requests.post", fake_post)

    provider = OllamaProvider(model="llama3.1:8b")
    result = provider.chat([{"role": "user", "content": "Olá"}])

    assert isinstance(result, ModelResponse)
    assert result.text == "Olá!"
    assert result.provider == "ollama"
    assert result.model == "llama3.1:8b"
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert result.latency_ms >= 0


def test_ollama_provider_raises_on_empty_completion(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.model_provider.requests.post",
        lambda *a, **kw: FakeResponse({"message": {"content": "   "}}),
    )
    provider = OllamaProvider(model="llama3.1:8b")
    with pytest.raises(RuntimeError):
        provider.chat([{"role": "user", "content": "Olá"}])


def test_ollama_provider_wraps_connection_errors(monkeypatch) -> None:
    import requests

    def fake_post(*a, **kw):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("assistant.model_provider.requests.post", fake_post)
    provider = OllamaProvider(model="llama3.1:8b")
    with pytest.raises(RuntimeError):
        provider.chat([{"role": "user", "content": "Olá"}])


def test_provider_backed_llm_chat_returns_text_and_records_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.model_provider.requests.post",
        lambda *a, **kw: FakeResponse({"message": {"content": "resposta"}, "prompt_eval_count": 3, "eval_count": 2}),
    )
    provider = OllamaProvider(model="llama3.1:8b")
    llm = ProviderBackedLLM(provider, system_prompt="base")

    text = llm.chat("Olá", history=[{"role": "user", "content": "oi"}])

    assert text == "resposta"
    assert llm.last_response.input_tokens == 3
    assert llm.last_response.output_tokens == 2
    assert len(llm.responses) == 1


def test_provider_backed_llm_choose_tool_parses_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.model_provider.requests.post",
        lambda *a, **kw: FakeResponse(
            {"message": {"content": json.dumps({"tool": None, "arguments": {}, "reason": "sem ferramenta"})}}
        ),
    )
    provider = OllamaProvider(model="llama3.1:8b")
    llm = ProviderBackedLLM(provider)

    decision = llm.choose_tool("Olá", "sem ferramentas")

    assert decision == {"tool": None, "arguments": {}, "reason": "sem ferramenta"}


def test_provider_backed_llm_choose_tool_defaults_on_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.model_provider.requests.post",
        lambda *a, **kw: FakeResponse({"message": {"content": "isto nao e json"}}),
    )
    provider = OllamaProvider(model="llama3.1:8b")
    llm = ProviderBackedLLM(provider)

    decision = llm.choose_tool("Olá", "sem ferramentas")

    assert decision == {"tool": None, "arguments": {}}


def test_estimate_cost_is_zero_for_ollama() -> None:
    assert estimate_cost("ollama", "llama3.1:8b", 1000, 500) == 0.0


def test_estimate_cost_is_zero_without_a_configured_price() -> None:
    assert estimate_cost("anthropic", "unknown-model", 1000, 500) == 0.0
