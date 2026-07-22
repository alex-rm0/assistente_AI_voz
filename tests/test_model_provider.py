from __future__ import annotations

import json

import pytest
import requests

from assistant.anthropic_provider import AnthropicProvider, PAID_CALL_CONFIRMATION_ENV
from assistant.model_provider import ModelResponse, OllamaProvider, ProviderBackedLLM, estimate_cost


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> dict:
        return self._payload


def test_ollama_provider_parses_response_and_tokens(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):
        assert url.endswith("/api/chat")
        assert json["model"] == "llama3.1:8b"
        return FakeResponse(
            {
                "message": {"content": " Ola! "},
                "prompt_eval_count": 12,
                "eval_count": 7,
            }
        )

    monkeypatch.setattr("assistant.model_provider.requests.post", fake_post)

    provider = OllamaProvider(model="llama3.1:8b")
    result = provider.chat([{"role": "user", "content": "Ola"}])

    assert isinstance(result, ModelResponse)
    assert result.text == "Ola!"
    assert result.provider == "ollama"
    assert result.model == "llama3.1:8b"
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert result.latency_ms >= 0
    assert result.estimated_cost_usd == 0.0


def test_ollama_provider_raises_on_empty_completion(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.model_provider.requests.post",
        lambda *a, **kw: FakeResponse({"message": {"content": "   "}}),
    )
    provider = OllamaProvider(model="llama3.1:8b")
    with pytest.raises(RuntimeError):
        provider.chat([{"role": "user", "content": "Ola"}])


def test_ollama_provider_wraps_connection_errors(monkeypatch) -> None:
    def fake_post(*a, **kw):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("assistant.model_provider.requests.post", fake_post)
    provider = OllamaProvider(model="llama3.1:8b")
    with pytest.raises(RuntimeError):
        provider.chat([{"role": "user", "content": "Ola"}])


def test_ollama_provider_embed_parses_embedding(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.model_provider.requests.post",
        lambda *a, **kw: FakeResponse({"embedding": [0.1, 2, "x"]}),
    )

    assert OllamaProvider(model="llama3.1:8b").embed("texto") == [0.1, 2.0]


def test_provider_backed_llm_chat_returns_text_and_records_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.model_provider.requests.post",
        lambda *a, **kw: FakeResponse({"message": {"content": "resposta"}, "prompt_eval_count": 3, "eval_count": 2}),
    )
    provider = OllamaProvider(model="llama3.1:8b")
    llm = ProviderBackedLLM(provider, system_prompt="base")

    text = llm.chat("Ola", history=[{"role": "user", "content": "oi"}])

    assert text == "resposta"
    assert llm.last_response.input_tokens == 3
    assert llm.last_response.output_tokens == 2
    assert llm.chat_call_tokens[0]["provider"] == "ollama"
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

    decision = llm.choose_tool("Ola", "sem ferramentas")

    assert decision == {"tool": None, "arguments": {}, "reason": "sem ferramenta"}


def test_provider_backed_llm_choose_tool_defaults_on_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.model_provider.requests.post",
        lambda *a, **kw: FakeResponse({"message": {"content": "isto nao e json"}}),
    )
    provider = OllamaProvider(model="llama3.1:8b")
    llm = ProviderBackedLLM(provider)

    decision = llm.choose_tool("Ola", "sem ferramentas")

    assert decision == {"tool": None, "arguments": {}}


def test_anthropic_requires_api_key_before_network(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("assistant.anthropic_provider.requests.post", lambda *a, **kw: pytest.fail("network called"))

    provider = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="", allow_paid_calls=True)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        provider.chat([{"role": "user", "content": "Ola"}])


def test_anthropic_requires_paid_call_confirmation_before_network(monkeypatch) -> None:
    monkeypatch.delenv(PAID_CALL_CONFIRMATION_ENV, raising=False)
    monkeypatch.setattr("assistant.anthropic_provider.requests.post", lambda *a, **kw: pytest.fail("network called"))

    provider = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="secret", allow_paid_calls=False)

    with pytest.raises(RuntimeError, match=PAID_CALL_CONFIRMATION_ENV):
        provider.chat([{"role": "user", "content": "Ola"}])


def test_anthropic_sends_system_prompt_history_and_records_cost(monkeypatch) -> None:
    seen: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        seen["headers"] = headers
        return FakeResponse(
            {
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": "Resposta"}],
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            }
        )

    monkeypatch.setattr("assistant.anthropic_provider.requests.post", fake_post)
    provider = AnthropicProvider(
        model="claude-haiku-4-5-20251001",
        api_key="secret",
        allow_paid_calls=True,
        timeout_seconds=7,
    )

    result = provider.chat(
        [
            {"role": "system", "content": "Prompt base"},
            {"role": "user", "content": "Mensagem antiga"},
            {"role": "assistant", "content": "Resposta antiga"},
            {"role": "user", "content": "Mensagem actual"},
        ]
    )

    assert seen["url"].endswith("/v1/messages")
    assert seen["headers"]["x-api-key"] == "secret"
    assert seen["json"]["system"] == "Prompt base"
    assert seen["json"]["messages"] == [
        {"role": "user", "content": "Mensagem antiga"},
        {"role": "assistant", "content": "Resposta antiga"},
        {"role": "user", "content": "Mensagem actual"},
    ]
    assert result.text == "Resposta"
    assert result.input_tokens == 1000
    assert result.output_tokens == 500
    assert result.estimated_cost_usd == pytest.approx(0.0035)


def test_anthropic_timeout_is_clear(monkeypatch) -> None:
    def fake_post(*a, **kw):
        raise requests.Timeout("slow")

    monkeypatch.setattr("assistant.anthropic_provider.requests.post", fake_post)
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="secret", allow_paid_calls=True)

    with pytest.raises(RuntimeError, match="tempo limite"):
        provider.chat([{"role": "user", "content": "Ola"}])


def test_anthropic_rate_limit_is_clear(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.anthropic_provider.requests.post",
        lambda *a, **kw: FakeResponse({"error": "rate"}, status=429),
    )
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="secret", allow_paid_calls=True)

    with pytest.raises(RuntimeError, match="rate limit"):
        provider.chat([{"role": "user", "content": "Ola"}])


def test_anthropic_empty_response_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "assistant.anthropic_provider.requests.post",
        lambda *a, **kw: FakeResponse({"content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}),
    )
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="secret", allow_paid_calls=True)

    with pytest.raises(RuntimeError, match="nao devolveu texto"):
        provider.chat([{"role": "user", "content": "Ola"}])


def test_estimate_cost_is_zero_for_ollama() -> None:
    assert estimate_cost("ollama", "llama3.1:8b", 1000, 500) == 0.0


def test_estimate_cost_is_zero_without_a_configured_price() -> None:
    assert estimate_cost("anthropic", "unknown-model", 1000, 500) == 0.0
