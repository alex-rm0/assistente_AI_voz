"""R2: the first voice turn paid ~9 s of cold starts (GPU STT model + first
CUDA inference, and llama3.1:8b reloading after Ollama's 5 min idle). The
models are now warmed in the background after the UI is ready (same timer
as the Pocket prewarm), and released on close without blocking it. A
failed warm-up (e.g. out of GPU memory) is logged and the app carries on;
the first turn then pays the cold start as before.
"""
from __future__ import annotations

import threading
import time

import pytest

from assistant import model_warmup
from assistant.model_provider import OllamaProvider, ProviderBackedLLM
from assistant.model_warmup import ModelWarmup, find_ollama_provider, unload_ollama_in_background, warm_ollama
from tests.test_web_ui_prototype import _FakeEngine, _qt_app, _run_until


class _Response:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _Posts:
    def __init__(self, *, status: int = 200, delay: float = 0.0, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.status = status
        self.delay = delay
        self.error = error

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, **(json or {})})
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return _Response(self.status)


class _Transcriber:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.warmed = 0
        self.closed = 0

    def prewarm_stt(self) -> bool:
        if self.error is not None:
            raise self.error
        self.warmed += 1
        return True

    def close(self) -> None:
        self.closed += 1


def _llm() -> ProviderBackedLLM:
    return ProviderBackedLLM(OllamaProvider(model="llama3.1:8b"))


def test_warm_ollama_loads_the_model_without_generating(monkeypatch) -> None:
    posts = _Posts()
    monkeypatch.setattr(model_warmup.requests, "post", posts)

    warm_ollama(OllamaProvider(model="llama3.1:8b"))

    assert posts.calls == [{"url": "http://127.0.0.1:11434/api/generate", "model": "llama3.1:8b", "prompt": "", "keep_alive": -1}]


def test_find_ollama_provider_in_plain_and_routed_llms() -> None:
    provider = OllamaProvider(model="llama3.1:8b")

    assert find_ollama_provider(ProviderBackedLLM(provider)) is provider
    assert find_ollama_provider(type("Routed", (), {"providers": {"ollama": provider}})()) is provider
    assert find_ollama_provider(object()) is None
    assert find_ollama_provider(None) is None


def test_warmup_warms_stt_then_ollama(monkeypatch, capsys) -> None:
    posts = _Posts()
    monkeypatch.setattr(model_warmup.requests, "post", posts)
    transcriber = _Transcriber()

    warmup = ModelWarmup(transcriber=transcriber, llm=_llm())
    warmup.run()

    assert transcriber.warmed == 1
    assert [call["keep_alive"] for call in posts.calls] == [-1]
    assert warmup.results == {"stt": "ready", "ollama": "ready"}
    out = capsys.readouterr().out
    assert "[MODEL_WARMUP] component=stt event=ready" in out
    assert "[MODEL_WARMUP] component=ollama event=ready" in out


@pytest.mark.parametrize("failing", ["stt", "ollama"])
def test_out_of_memory_degrades_without_crashing(monkeypatch, capsys, failing: str) -> None:
    oom = RuntimeError("CUDA out of memory")
    posts = _Posts(error=oom if failing == "ollama" else None)
    monkeypatch.setattr(model_warmup.requests, "post", posts)
    transcriber = _Transcriber(error=oom if failing == "stt" else None)

    warmup = ModelWarmup(transcriber=transcriber, llm=_llm())
    warmup.run()  # must not raise

    assert warmup.results[failing] == "failed"
    other = "ollama" if failing == "stt" else "stt"
    assert warmup.results[other] == "ready"  # the other model is still warmed
    out = capsys.readouterr().out
    assert f"[MODEL_WARMUP] component={failing} event=failed degraded=true" in out
    assert "out of memory" in out


def test_unload_sends_keep_alive_zero_without_blocking(monkeypatch) -> None:
    posts = _Posts(delay=0.5)
    monkeypatch.setattr(model_warmup.requests, "post", posts)

    started = time.perf_counter()
    thread = unload_ollama_in_background(OllamaProvider(model="llama3.1:8b"))
    elapsed = time.perf_counter() - started
    thread.join(timeout=2)

    assert elapsed < 0.1
    assert posts.calls[-1]["keep_alive"] == 0 and posts.calls[-1]["prompt"] == ""
    assert thread.daemon is False  # the request is sent even while the app exits


# --- controller wiring --------------------------------------------------------


class _EngineWithOllama(_FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.llm = _llm()


def _controller(transcriber, *, delay_ms: int = 20):
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _EngineWithOllama()
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    controller._TTS_STARTUP_PREWARM_DELAY_MS = delay_ms
    return controller


def test_controller_warms_models_after_ui_ready(monkeypatch) -> None:
    posts = _Posts()
    monkeypatch.setattr(model_warmup.requests, "post", posts)
    transcriber = _Transcriber()
    controller = _controller(transcriber)
    try:
        _run_until(lambda: False, timeout_ms=60)
        assert transcriber.warmed == 0  # nothing before the UI is ready
        controller.notify_ui_ready()
        assert _run_until(lambda: transcriber.warmed == 1 and len(posts.calls) == 1, timeout_ms=2000)
        assert posts.calls[0]["keep_alive"] == -1
    finally:
        controller.shutdown()


def test_controller_close_unloads_ollama_without_blocking(monkeypatch) -> None:
    posts = _Posts(delay=0.5)
    monkeypatch.setattr(model_warmup.requests, "post", posts)
    controller = _controller(_Transcriber())

    started = time.perf_counter()
    controller.request_shutdown()
    elapsed = time.perf_counter() - started
    controller.shutdown()

    assert elapsed < 0.1
    deadline = time.perf_counter() + 2
    while not posts.calls and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert posts.calls and posts.calls[-1]["keep_alive"] == 0
    for thread in threading.enumerate():
        if thread.name == "EchoOllamaUnload":
            thread.join(timeout=2)


# --- regression: no thread when there is nothing to warm ----------------------
# A Python thread can run the garbage collector and destroy dead Qt wrappers
# off the GUI thread: with a warm-up thread started on every STANDBY entry,
# the wake tests crashed with a fail-fast (0xC0000409) 6 times out of 8.


def _count_threads(monkeypatch) -> list:
    created: list = []
    real_thread = model_warmup.threading.Thread

    def counting_thread(*args, **kwargs):
        created.append(kwargs.get("name"))
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(model_warmup.threading, "Thread", counting_thread)
    return created


def test_no_thread_when_there_is_nothing_to_warm(monkeypatch) -> None:
    created = _count_threads(monkeypatch)

    class _Bare:  # a transcriber without GPU STT warm-up
        pass

    ModelWarmup(transcriber=_Bare(), llm=None).start()

    assert created == []


def test_default_stt_without_ollama_is_skipped_without_a_thread(monkeypatch, capsys) -> None:
    from assistant.voice_input import VoiceTranscriber

    created = _count_threads(monkeypatch)
    warmup = ModelWarmup(transcriber=VoiceTranscriber(model_name="medium", stt_provider="openai-whisper"), llm=None)
    warmup.start()

    assert created == []
    assert warmup.results == {"stt": "skipped"}
    assert "[MODEL_WARMUP] component=stt event=skipped reason=default_stt" in capsys.readouterr().out
