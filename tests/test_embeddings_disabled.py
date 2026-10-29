"""KNOWN_ISSUES §4: embeddings are off by default.

The app's embedder is the chat LLM (llama3.1:8b), which Ollama rejects on
/api/embeddings with HTTP 500 every time -- the first call cost ~22 s cold
and forced a second model load for chat, while 0 memories ever got an
embedding (search always fell back to text). With the default
configuration no memory path may issue that request at all.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import requests

import app
from assistant.long_term_memory import LongTermMemory
from assistant.model_provider import OllamaProvider, ProviderBackedLLM


class _RecordingPost:
    """Stands in for requests.post: records every URL and answers like the
    real Ollama does for a chat model on /api/embeddings (HTTP 500)."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, url: str, *args, **kwargs):
        self.urls.append(str(url))
        response = requests.Response()
        response.status_code = 500
        response.url = str(url)
        return response


@pytest.fixture
def recorded_posts(monkeypatch) -> _RecordingPost:
    recorder = _RecordingPost()
    monkeypatch.setattr(requests, "post", recorder)
    return recorder


def _default_config_memory(tmp_path: Path) -> LongTermMemory:
    """Built exactly like app.main() does: real config/settings.json memory
    section, real Ollama provider chain as the embedder."""
    memory_config = app.load_settings().get("memory", {})
    embedder = ProviderBackedLLM(OllamaProvider(model="llama3.1:8b"))
    return app.build_long_term_memory(tmp_path, memory_config, embedder)


def test_default_config_has_embeddings_disabled() -> None:
    assert app.load_settings()["memory"]["embeddings_enabled"] is False


def test_no_memory_path_requests_embeddings_with_default_config(tmp_path: Path, recorded_posts) -> None:
    memory = _default_config_memory(tmp_path)

    memory.remember("O meu projeto principal é o assistenteIA.")
    memory.search("projeto principal")
    memory.context_for("projeto principal")
    memory.create_task("Rever a bateria de estabilização")
    # The call engine.close_session() -> SessionManager.end_session() makes.
    memory.remember_context_summary("Resumo de sessao: projeto=assistenteIA; atividade=teste.")

    embedding_calls = [url for url in recorded_posts.urls if "/api/embeddings" in url]
    assert embedding_calls == []


def test_missing_flag_also_means_disabled(tmp_path: Path, recorded_posts) -> None:
    embedder = ProviderBackedLLM(OllamaProvider(model="llama3.1:8b"))
    memory = app.build_long_term_memory(tmp_path, {}, embedder)

    memory.remember("nota sem flag")
    memory.search("nota")

    assert [url for url in recorded_posts.urls if "/api/embeddings" in url] == []


def test_embeddings_still_used_when_explicitly_enabled(tmp_path: Path, recorded_posts) -> None:
    """The embedding code is kept, only switched off: enabling it restores
    the original request (which today still answers 500 -> None)."""
    embedder = ProviderBackedLLM(OllamaProvider(model="llama3.1:8b"))
    memory = app.build_long_term_memory(tmp_path, {"embeddings_enabled": True}, embedder)

    memory.remember("nota com embeddings")

    assert any("/api/embeddings" in url for url in recorded_posts.urls)


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float] | None:
        self.calls += 1
        return [1.0, 0.0]


def test_search_results_match_text_fallback_when_disabled(tmp_path: Path) -> None:
    """Disabled embeddings must give exactly the text-fallback ranking the
    app already got (embedder returning None on every 500)."""
    seed = [
        "O meu projeto principal é o assistenteIA.",
        "Gosto de café sem açúcar.",
        "A reunião de direção é à segunda-feira.",
        "O assistenteIA usa Ollama localmente.",
    ]
    baseline = LongTermMemory(tmp_path / "baseline", embedder=None)
    embedder = _CountingEmbedder()
    disabled = LongTermMemory(tmp_path / "disabled", embedder=embedder, embeddings_enabled=False)
    for text in seed:
        baseline.remember(text)
        disabled.remember(text)

    for query in ("assistenteIA", "café", "reunião segunda", "Ollama", "nada a ver"):
        expected = [(r.category, r.content, r.score) for r in baseline.search(query)]
        actual = [(r.category, r.content, r.score) for r in disabled.search(query)]
        assert actual == expected, query
    assert embedder.calls == 0
