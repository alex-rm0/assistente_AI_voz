"""R2 (first voice turn): warm the local models a voice turn needs before the
first turn, and release them on close.

The first turn of the 11/11 session paid ~9 s of cold starts: the GPU STT
worker loaded its model (4.7 s) and ran its first CUDA inference (1.3 s),
and Ollama reloaded llama3.1:8b after its default 5 min idle (3.8 s). The
controller warms Ollama right away (external process, no UI contention) and
the GPU STT after the UI is ready (same timer as the Pocket TTS prewarm), and
releases both on close.

Plain Python threads, never Qt: nothing here touches a Qt object, and the
close path must not wait for a model load. A failed warm-up (no CUDA, out of
GPU memory, Ollama down) is logged as degraded and the app carries on; the
first turn then pays the cold start as before.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import requests

from assistant.model_provider import OLLAMA_KEEP_ALIVE, OllamaProvider


def find_ollama_provider(llm: Any) -> OllamaProvider | None:
    """The OllamaProvider behind the engine's LLM (plain or routed), if any."""
    if llm is None:
        return None
    providers = getattr(llm, "providers", None)
    if isinstance(providers, dict) and isinstance(providers.get("ollama"), OllamaProvider):
        return providers["ollama"]
    provider = getattr(llm, "provider", None)
    return provider if isinstance(provider, OllamaProvider) else None


def warm_ollama(provider: OllamaProvider, *, keep_alive: int | str = OLLAMA_KEEP_ALIVE, timeout: float = 120.0) -> None:
    """Load (keep_alive=-1) or unload (keep_alive=0) the model: an empty
    /api/generate prompt only changes what is loaded, it generates nothing.
    Local only -- never a paid call."""
    response = requests.post(
        f"{provider.base_url.rstrip('/')}/api/generate",
        json={"model": provider.model, "prompt": "", "keep_alive": keep_alive},
        timeout=timeout,
    )
    response.raise_for_status()


def unload_ollama_in_background(provider: OllamaProvider | None, *, timeout: float = 3.0) -> threading.Thread | None:
    """Ask Ollama to unload the model (chats keep it with keep_alive=-1).
    Never blocks the caller; non-daemon so the request is still sent while
    the app is exiting."""
    if provider is None:
        return None

    def run() -> None:
        try:
            warm_ollama(provider, keep_alive=0, timeout=timeout)
            print("[MODEL_WARMUP] component=ollama event=unloaded", flush=True)
        except Exception as exc:
            print(f"[MODEL_WARMUP] component=ollama event=unload_failed error={type(exc).__name__}", flush=True)

    thread = threading.Thread(target=run, name="EchoOllamaUnload", daemon=False)
    thread.start()
    return thread


class ModelWarmup:
    """Warms the GPU STT (when active) and then the Ollama model, in order,
    on one background thread."""

    def __init__(self, *, transcriber: Any = None, llm: Any = None) -> None:
        self.transcriber = transcriber
        self.provider = find_ollama_provider(llm)
        self.results: dict[str, str] = {}
        self.cancelled = False
        self._threads: dict[str, threading.Thread] = {}

    # Ollama is an external process and does not compete with the UI, so the
    # controller starts it right away; the GPU STT (in-app worker start-up)
    # waits for the UI to be ready. Each component starts once, on its own
    # daemon thread: a warm-up cut short by the app exiting leaves nothing
    # behind (Ollama finishes its load; the STT worker exits on stdin EOF).
    def start(self) -> None:
        """Both components (each starts once)."""
        self.start_ollama()
        self.start_stt()

    def start_ollama(self) -> None:
        if self.provider is not None:
            self._start("ollama", self._warm_ollama)

    def start_stt(self) -> None:
        if not callable(getattr(self.transcriber, "prewarm_stt", None)):
            return
        if getattr(self.transcriber, "stt_provider", None) == "openai-whisper":
            # An explicitly selected CPU backend has no GPU worker to warm.
            if "stt" not in self.results:
                self.results["stt"] = "skipped"
                print("[MODEL_WARMUP] component=stt event=skipped reason=explicit_cpu_backend", flush=True)
            return
        self._start("stt", self._warm_stt)

    def _start(self, component: str, target: Callable[[], None]) -> None:
        # No thread at all when there is nothing to warm: a Python thread can
        # run the garbage collector, which then destroys dead Qt wrappers off
        # the GUI thread (seen as a fail-fast crash in the wake tests).
        if component in self._threads or self.cancelled:
            return
        thread = threading.Thread(target=target, name=f"EchoModelWarmup-{component}", daemon=True)
        self._threads[component] = thread
        thread.start()

    def is_running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads.values())

    def cancel(self) -> None:
        """App close: stop a warm-up in flight without waiting for it."""
        self.cancelled = True
        stt = self._threads.get("stt")
        if stt is not None and stt.is_alive() and self.transcriber is not None:
            close = getattr(self.transcriber, "close", None)
            if callable(close):
                close()

    def run(self) -> None:
        """Both, in order, on the caller's thread: Ollama first, then STT."""
        self._warm_ollama()
        self._warm_stt()

    def _warm_ollama(self) -> None:
        if self.provider is not None and not self.cancelled:
            self._step("ollama", lambda: warm_ollama(self.provider))

    def _warm_stt(self) -> None:
        prewarm_stt = getattr(self.transcriber, "prewarm_stt", None)
        if callable(prewarm_stt) and not self.cancelled:
            self._step("stt", prewarm_stt, skipped_reason="backend_not_warmable")

    def _step(self, component: str, warm: Callable[[], Any], *, skipped_reason: str = "") -> None:
        started = time.perf_counter()
        try:
            result = warm()
        except Exception as exc:
            self.results[component] = "failed"
            print(
                f"[MODEL_WARMUP] component={component} event=failed degraded=true "
                f"error={type(exc).__name__}: {str(exc)[:160]}",
                flush=True,
            )
            return
        if result is False:
            self.results[component] = "skipped"
            print(f"[MODEL_WARMUP] component={component} event=skipped reason={skipped_reason}", flush=True)
            return
        self.results[component] = "ready"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(f"[MODEL_WARMUP] component={component} event=ready elapsed_ms={elapsed_ms}", flush=True)
