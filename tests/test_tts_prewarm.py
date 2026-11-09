"""Pocket TTS cold start: the server only started on the first speak
request (real log: pocket_tts_server_start -> ready ~5 s), so the first
wake acknowledgement ("I'm listening") waited for it. The controller now
starts the TTS worker in the background (QThread, like the wake prewarm)
when voice is available, and again on STANDBY/conversation entry; the UI
thread never blocks and shutdown drains it.
"""
from __future__ import annotations

import threading
import time

from assistant.tts import TTSManager
from tests.test_web_ui_prototype import _FakeEngine, _FakeTTSManager, _qt_app, _run_until


class _PrewarmableTTS(_FakeTTSManager):
    def __init__(self, delay_seconds: float = 0.3) -> None:
        super().__init__()
        self.prewarm_delay = delay_seconds
        self.prewarm_threads: list[str] = []
        self.prewarm_finished = 0

    def prewarm(self) -> bool:
        self.prewarm_threads.append(threading.current_thread().name)
        time.sleep(self.prewarm_delay)
        self.prewarm_finished += 1
        return True


def _controller(tts, *, voice_available: bool = True):
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    return EchoUIController(
        _FakeEngine().respond,
        voice_available=voice_available,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.01,
        thinking_min_ms=0,
    )


def test_controller_prewarms_tts_in_background_when_voice_is_available(capsys) -> None:
    tts = _PrewarmableTTS(delay_seconds=0.3)

    started = time.perf_counter()
    controller = _controller(tts)
    construct_ms = (time.perf_counter() - started) * 1000

    assert construct_ms < 200  # never blocks the UI thread
    assert _run_until(lambda: tts.prewarm_finished == 1, timeout_ms=3000)
    assert tts.prewarm_threads and tts.prewarm_threads[0] != threading.main_thread().name
    assert _run_until(lambda: controller._tts_prewarm_thread is None, timeout_ms=2000)
    out = capsys.readouterr().out
    assert "[TTS_PREWARM] event=started" in out
    assert "[TTS_PREWARM] event=finished" in out
    controller.shutdown()


def test_no_tts_prewarm_without_voice() -> None:
    tts = _PrewarmableTTS(delay_seconds=0.0)
    controller = _controller(tts, voice_available=False)

    _run_until(lambda: False, timeout_ms=200)

    assert tts.prewarm_threads == []
    controller.shutdown()


def test_entering_standby_prewarms_tts_once_at_a_time() -> None:
    tts = _PrewarmableTTS(delay_seconds=0.3)
    controller = _controller(tts)

    controller._prewarm_tts()  # already in flight from construction -> ignored
    controller._prewarm_tts()
    assert _run_until(lambda: tts.prewarm_finished >= 1 and controller._tts_prewarm_thread is None, timeout_ms=3000)
    assert len(tts.prewarm_threads) == 1
    controller.shutdown()


def test_shutdown_while_prewarming_waits_for_it_and_reports_it_pending() -> None:
    tts = _PrewarmableTTS(delay_seconds=0.4)
    controller = _controller(tts)
    assert _run_until(lambda: bool(tts.prewarm_threads), timeout_ms=2000)

    controller.request_shutdown()
    assert "tts_prewarm" in controller.shutdown_pending_components()
    controller.shutdown(wait_ms=2000)

    assert tts.prewarm_finished == 1
    assert controller._tts_prewarm_thread is None


class _WarmupProvider:
    name = "fake"

    def __init__(self) -> None:
        self.warmups = 0

    def warmup(self) -> None:
        self.warmups += 1


def test_tts_manager_prewarm_starts_the_provider_worker() -> None:
    provider = _WarmupProvider()

    assert TTSManager(provider=provider).prewarm() is True
    assert provider.warmups == 1
