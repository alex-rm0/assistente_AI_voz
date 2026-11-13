"""The conversation eval can run the app's automatic routing (--provider auto):
Claude for conversation, the local model as the reserve. Fake providers only.
"""
from __future__ import annotations

import pytest

from assistant.model_provider import ModelResponse, ProviderTimeoutError
from evals.conversation import runner


class _Provider:
    def __init__(self, name: str, model: str, cost: float = 0.0) -> None:
        self._name, self.model, self.cost = name, model, cost
        self.error: Exception | None = None
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages, *, model=None, **kwargs) -> ModelResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        text = "Well done." if self._name == "anthropic" else "Good."
        return ModelResponse(text=text, provider=self._name, model=model or self.model, input_tokens=5, output_tokens=2,
                             latency_ms=1.0, estimated_cost_usd=self.cost)

    def embed(self, text):
        return None


@pytest.fixture
def isolated_eval_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHO_ENV", "test")
    monkeypatch.setenv("ECHO_TEST_DATA_DIR", str(tmp_path / "eval-root"))


def _cases(count: int) -> list[dict]:
    return [{"id": f"c{index}", "language": "en-GB", "turns": [{"user": "I passed my exam today."}]} for index in range(count)]


def test_auto_needs_the_opt_in_and_a_cost_ceiling() -> None:
    with pytest.raises(PermissionError):
        runner.make_llm_factory("auto")
    with pytest.raises(ValueError, match="cost ceiling"):
        runner.make_llm_factory("auto", allow_cloud=True)


def test_auto_sends_conversation_to_claude_and_reports_who_answered(isolated_eval_env) -> None:
    ollama, claude = _Provider("ollama", "llama3.1:8b"), _Provider("anthropic", "claude-haiku-4-5-20251001", cost=0.001)
    factory = runner.make_llm_factory("auto", allow_cloud=True, max_cost_usd=0.5, auto_providers={"ollama": ollama, "anthropic": claude})

    report = runner.run_cases(_cases(2) + [{"id": "social", "language": "en-GB", "turns": [{"user": "Bom dia"}]}],
                              factory, provider="auto", max_cost_usd=0.5)

    turns = [turn for case in report["cases"] for turn in case["turns"]]
    assert [turn["answered_by"] for turn in turns] == ["anthropic", "anthropic", "local"]
    assert [turn["routing_reason"] for turn in turns[:2]] == ["general_conversation", "general_conversation"]
    assert (claude.calls, ollama.calls) == (2, 0)
    usage = report["summary"]["usage"]
    assert usage["answered_by"] == {"anthropic": 2, "local": 1}
    assert usage["cost_usd"] == pytest.approx(0.002)


def test_auto_falls_back_to_the_local_model_and_says_why(isolated_eval_env) -> None:
    ollama, claude = _Provider("ollama", "llama3.1:8b"), _Provider("anthropic", "claude-haiku-4-5-20251001", cost=0.001)
    claude.error = ProviderTimeoutError("slow", provider="anthropic")
    factory = runner.make_llm_factory("auto", allow_cloud=True, max_cost_usd=0.5, auto_providers={"ollama": ollama, "anthropic": claude})

    report = runner.run_cases(_cases(1), factory, provider="auto", max_cost_usd=0.5)

    turn = report["cases"][0]["turns"][0]
    assert turn["response"] == "Good." and not turn["error"]
    assert (turn["answered_by"], turn["routing_reason"], turn["routing_fallback"]) == ("ollama", "claude_timeout", "claude_timeout")
    assert report["summary"]["usage"]["cost_usd"] == 0.0


def test_auto_stops_at_the_cost_ceiling(isolated_eval_env) -> None:
    ollama, claude = _Provider("ollama", "llama3.1:8b"), _Provider("anthropic", "claude-haiku-4-5-20251001", cost=0.2)
    factory = runner.make_llm_factory("auto", allow_cloud=True, max_cost_usd=0.5, auto_providers={"ollama": ollama, "anthropic": claude})

    report = runner.run_cases(_cases(6), factory, provider="auto", max_cost_usd=0.5)

    assert report["stopped"] == "cost_limit" and len(report["cases"]) == 3
