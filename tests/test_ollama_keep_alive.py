"""R2: no request set keep_alive, so Ollama unloaded llama3.1:8b after its
default 5 minutes idle and the next turn paid the reload (3.76 s load, 4.67 s
first /api/chat in the 11/11 log). Every chat now asks Ollama to keep the
model loaded (keep_alive=-1); the app unloads it on close (model_warmup).
"""
from __future__ import annotations

from assistant import model_provider
from assistant.model_provider import OllamaProvider


class _Response:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"message": {"content": "Ok."}, "prompt_eval_count": 1, "eval_count": 1}


def test_chat_payload_keeps_the_model_loaded(monkeypatch) -> None:
    sent: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append({"url": url, **json})
        return _Response()

    monkeypatch.setattr(model_provider.requests, "post", fake_post)

    OllamaProvider(model="llama3.1:8b").chat([{"role": "user", "content": "olá"}])

    assert sent[-1]["url"].endswith("/api/chat")
    assert sent[-1]["keep_alive"] == -1
