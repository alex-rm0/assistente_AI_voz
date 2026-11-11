"""Pocket TTS server start-up could not be interrupted: shutdown() waited on
the provider _lock held by the start, and a subprocess that died mid-start
left start() waiting for the whole load timeout (60 s by default), because
the stdout reader ended without telling anyone. Closing the app while the
prewarm was starting therefore kept the process alive (3 s in the battery,
up to the load timeout with a stalled server).
"""
from __future__ import annotations

import io
import sys
import threading
import time

from assistant.tts import PocketTTSLocalProvider, TTSError

LOAD_TIMEOUT_S = 8.0


class _LoadingProcess:
    """A Pocket server still loading: stdout yields nothing until the process
    is terminated (then EOF). Optionally prints "ready" after a delay."""

    instances: list["_LoadingProcess"] = []

    def __init__(self, *args, ready_after: float | None = None, **kwargs) -> None:
        self._dead = threading.Event()
        self._ready_after = ready_after
        self.stdin = io.StringIO()
        self.stdout = self._lines()
        self.stderr = iter(())
        _LoadingProcess.instances.append(self)

    def _lines(self):
        if self._ready_after is not None and not self._dead.wait(self._ready_after):
            yield '{"ok": true, "type": "ready", "model_load_ms": 5}\n'
        self._dead.wait()

    def poll(self):
        return 0 if self._dead.is_set() else None

    def terminate(self) -> None:
        self._dead.set()

    def kill(self) -> None:
        self._dead.set()

    def wait(self, timeout=None):
        return 0


def _provider(factory=_LoadingProcess) -> PocketTTSLocalProvider:
    return PocketTTSLocalProvider(process_factory=factory, load_timeout_seconds=LOAD_TIMEOUT_S, runtime_python=sys.executable)


def _start_warmup(provider: PocketTTSLocalProvider) -> tuple[threading.Thread, dict]:
    result: dict = {}

    def run() -> None:
        started = time.perf_counter()
        try:
            provider.warmup()
            result["outcome"] = "ok"
        except TTSError as exc:
            result["outcome"] = f"error: {exc}"
        result["elapsed_s"] = time.perf_counter() - started

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def _wait_for_process() -> _LoadingProcess:
    deadline = time.perf_counter() + 2
    while not _LoadingProcess.instances and time.perf_counter() < deadline:
        time.sleep(0.01)
    return _LoadingProcess.instances[-1]


def test_shutdown_during_startup_returns_at_once_and_cancels_the_start() -> None:
    _LoadingProcess.instances.clear()
    provider = _provider()
    thread, result = _start_warmup(provider)
    _wait_for_process()
    time.sleep(0.2)

    started = time.perf_counter()
    provider.shutdown()
    shutdown_s = time.perf_counter() - started
    thread.join(timeout=LOAD_TIMEOUT_S + 2)

    assert shutdown_s < 1.0, shutdown_s
    assert result["outcome"].startswith("error"), result
    assert result["elapsed_s"] < 2.0, result
    assert provider._process is None


def test_subprocess_dying_mid_startup_fails_start_at_once() -> None:
    _LoadingProcess.instances.clear()
    provider = _provider()
    thread, result = _start_warmup(provider)
    process = _wait_for_process()
    time.sleep(0.2)

    process.terminate()  # the server process exits -> stdout EOF
    thread.join(timeout=LOAD_TIMEOUT_S + 2)

    assert result["outcome"].startswith("error"), result
    assert result["elapsed_s"] < 2.0, result


def test_normal_startup_still_becomes_ready() -> None:
    _LoadingProcess.instances.clear()
    provider = _provider(lambda *a, **k: _LoadingProcess(ready_after=0.2))

    thread, result = _start_warmup(provider)
    thread.join(timeout=LOAD_TIMEOUT_S + 2)

    assert result["outcome"] == "ok", result
    assert provider._process is not None and provider._process.is_alive()
    provider.shutdown()
    assert provider._process is None
