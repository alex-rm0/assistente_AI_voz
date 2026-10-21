"""Phase 7.6.7: real-runtime wake reliability + non-blocking shutdown.

Controller-level tests for the async close sequence (shutdown_async) and
the quick-rearm wake policy, plus unit tests for the fast-wake assessment
gate and the wake recognizer's confidence metadata. The real failure these
cover: a STANDBY wake attempt that spent stt_ms=9184 in a Medium fallback
only to be ignored, and an app close that froze the window ("Não
responde") while shutdown ran synchronously on the GUI thread.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import shiboken6

from assistant.voice_input import VoiceNoSpeechDetected
from assistant.voice_wake import assess_fast_wake_transcript, classify_voice_wake_intent
from assistant.wake_recognizer import WakeRecognizer


def _qt_app():
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def _run_until(predicate, timeout_ms: int = 3000) -> bool:
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    deadline = time.perf_counter() + (timeout_ms / 1000)
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


class _Engine:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def respond(self, message: str) -> str:
        self.messages.append(message)
        return "Reply."


class _StandbyTranscriber:
    """A STANDBY transcriber double with controllable timing.

    mode="capture": blocks in "audio capture" until stop_event is set, then
        reports no speech -- like the real VAD loop (section G).
    mode="busy": ignores stop_event for busy_seconds -- an uninterruptible
        Whisper inference (tiny, or the Medium fallback) -- then returns
        `result` (sections H/I).
    mode="immediate": returns `result` straight away.
    """

    def __init__(
        self,
        *,
        mode: str = "capture",
        result: str = "",
        busy_seconds: float = 0.0,
        fast_assessment: str | None = None,
    ) -> None:
        self.mode = mode
        self.result = result
        self.busy_seconds = busy_seconds
        self.fast_assessment = fast_assessment
        self.model_name = "medium"
        self.standby_vad_config = object()
        self.vad_config = object()
        self.last_vad_outcome: dict[str, object] | None = None
        self.started = threading.Event()
        self.calls = 0

    def record_and_transcribe_wake(self, stop_event, status_callback=None, *, on_speech_confirmed=None) -> str:
        self.calls += 1
        self.started.set()
        self.last_vad_outcome = {"speech_ms": 300.0, "trailing_silence_ms": 450.0, "stt_ms": 200.0}
        if self.fast_assessment is not None:
            self.last_vad_outcome["fast_assessment"] = self.fast_assessment
        if self.mode == "capture":
            while not stop_event.is_set():
                time.sleep(0.005)
            raise VoiceNoSpeechDetected("Nao detetei fala.")
        if self.mode == "busy":
            time.sleep(self.busy_seconds)
        return self.result

    def record_and_transcribe_vad(self, stop_event, status_callback=None, **_kwargs) -> str:
        while not stop_event.is_set():
            time.sleep(0.005)
        raise VoiceNoSpeechDetected("Nao detetei fala.")

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        raise AssertionError("manual path must not be used")


class _SlowPrewarm:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.calls = 0

    def prewarm(self) -> None:
        time.sleep(self.delay_seconds)
        self.calls += 1


class _TTSManager:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


def _controller(transcriber, **kwargs):
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    return EchoUIController(
        _Engine().respond,
        voice_available=True,
        voice_transcriber=transcriber,
        voice_missing_dependencies=[],
        thinking_min_ms=0,
        **kwargs,
    )


def _close(controller, session_close=None, timeout_ms: int = 5000) -> dict[str, object]:
    """Drives shutdown_async() the way EchoOSWindow.closeEvent does, and
    records how long the call itself held the (GUI) thread."""
    result: dict[str, object] = {"completed": 0}

    def on_complete() -> None:
        result["completed"] = int(result["completed"]) + 1

    started = time.perf_counter()
    controller.shutdown_async(on_complete, session_close=session_close)
    result["call_ms"] = (time.perf_counter() - started) * 1000.0
    result["drained"] = _run_until(lambda: result["completed"] == 1, timeout_ms=timeout_ms)
    return result


# --- Assessment gate ---------------------------------------------------------


def test_assess_clear_wake_is_wake() -> None:
    for phrase in ("Echo", "Echo, acorda", "acorda", "wake up", "Hey Echo!"):
        assert assess_fast_wake_transcript(phrase) == "wake", phrase


def test_assess_empty_results() -> None:
    for phrase in ("", "   ", "...", "?!"):
        assert assess_fast_wake_transcript(phrase) == "empty", repr(phrase)


def test_assess_plausible_wake_mishearings_are_near_wake() -> None:
    for phrase in ("Eco.", "Eko!", "Ecco", "Ekko", "hecho", "a corda", "acordo"):
        assert assess_fast_wake_transcript(phrase) == "near_wake", phrase


def test_assess_unrelated_and_hallucinated_results() -> None:
    # "Ainda, a" is the real tiny output replayed on the session's own
    # low-level STANDBY capture.
    for phrase in ("Ainda, a", "ola", "Obrigado.", "ok", "eu", "blah blah", "o que voce esta fazendo hoje a noite"):
        assert assess_fast_wake_transcript(phrase) == "unrelated", phrase


def test_assess_sleep_phrase_in_standby_is_unrelated() -> None:
    assert assess_fast_wake_transcript("pode ir dormir") == "unrelated"


def test_no_alias_added_without_evidence() -> None:
    """F: no real tiny transcript of a human "Echo" was available, so no
    alias was added -- near-miss spellings only earn a Medium pass, they
    never wake Echo on their own."""
    for phrase in ("Eco", "Eko", "Ecco", "ego"):
        assert classify_voice_wake_intent(phrase) == "none", phrase


# --- WakeRecognizer metadata -------------------------------------------------


class _SegmentWhisperModel:
    def transcribe(self, audio_path: str, **options: object) -> dict[str, object]:
        return {
            "text": " Ainda, a",
            "language": "pt",
            "segments": [{"no_speech_prob": 0.287, "avg_logprob": -1.959}],
        }


class _SegmentWhisperModule:
    def load_model(self, model_name: str) -> _SegmentWhisperModel:
        return _SegmentWhisperModel()


def test_wake_recognizer_reports_confidence_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "whisper", _SegmentWhisperModule())
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    result = WakeRecognizer(model_name="tiny").recognize(audio_path, input_language="pt")

    assert result.transcript == "Ainda, a"
    assert result.no_speech_prob == 0.287
    assert result.avg_logprob == -1.959
    assert result.language == "pt"


# --- Controller: wake policy -------------------------------------------------


def test_a_clear_wake_enters_active_exactly_once(capsys) -> None:
    """A/E: a clear fast wake enters ACTIVE once -- no duplicate wake
    transition, no Medium involvement at controller level."""
    transcriber = _StandbyTranscriber(mode="immediate", result="Echo", fast_assessment="wake")
    tts = _TTSManager()
    controller = _controller(transcriber, tts_manager=tts)
    transitions: list[bool] = []
    controller.voiceConversationChanged.connect(transitions.append)

    controller.startVoiceStandby()
    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=3000)
    _run_until(lambda: False, timeout_ms=150)

    assert transitions.count(True) == 1
    assert capsys.readouterr().out.count("event=wake_intent_matched") == 1
    assert _close(controller)["drained"] is True


def test_b_fast_unrelated_result_rearms_quickly_and_logs_reason(capsys) -> None:
    """B/C: an empty-string result tagged fast_assessment=unrelated is
    ignored with a reason and STANDBY rearms -- no wake, no submit."""
    transcriber = _StandbyTranscriber(mode="immediate", result="", fast_assessment="unrelated")
    controller = _controller(transcriber)

    controller.startVoiceStandby()
    assert _run_until(lambda: transcriber.calls >= 2, timeout_ms=3000)  # rearmed and listening again
    out = capsys.readouterr().out

    assert "event=wake_ignored session_id=1 reason=fast_unrelated" in out
    assert "fast_assessment=unrelated" in out
    assert controller.isVoiceConversationActive() is False
    assert "REQUEST_BEGIN" not in out
    assert _close(controller)["drained"] is True


# --- Controller: non-blocking shutdown ----------------------------------------


def test_g_shutdown_while_standby_capture_active(capsys) -> None:
    transcriber = _StandbyTranscriber(mode="capture")
    controller = _controller(transcriber)
    controller.startVoiceStandby()
    assert transcriber.started.wait(2)
    voice_thread = controller._voice_thread

    result = _close(controller)

    assert result["drained"] is True
    assert controller.shutdown_pending_components() == []
    assert not shiboken6.isValid(voice_thread) or voice_thread.isRunning() is False
    out = capsys.readouterr().out
    assert "[SHUTDOWN] event=started" in out
    assert "[SHUTDOWN] component=voice_worker event=stop_requested" in out
    assert "[SHUTDOWN] component=voice_worker event=finished" in out


def test_h_shutdown_while_fast_recognizer_busy_does_not_block_caller(capsys) -> None:
    """H: tiny inference ignores stop_event; shutdown_async must return
    immediately (the GUI thread stays free) and complete once it ends."""
    transcriber = _StandbyTranscriber(mode="busy", busy_seconds=0.4, result="")
    controller = _controller(transcriber)
    controller.startVoiceStandby()
    assert transcriber.started.wait(2)

    result = _close(controller)

    assert result["call_ms"] < 100, result
    assert result["drained"] is True
    assert controller.is_shutdown_complete() is True


def test_i_shutdown_while_medium_fallback_busy_ignores_late_wake(capsys) -> None:
    """I/M: a late "Echo" from an in-flight Medium fallback after close
    began is stale -- never wakes ACTIVE, never submits, never rearms."""
    transcriber = _StandbyTranscriber(mode="busy", busy_seconds=0.8, result="Echo")
    engine_messages: list[str] = []
    controller = _controller(transcriber)
    controller.responder = lambda message: engine_messages.append(message) or "Reply."
    transitions: list[bool] = []
    controller.voiceConversationChanged.connect(transitions.append)
    controller.startVoiceStandby()
    assert transcriber.started.wait(2)

    result = _close(controller)

    assert result["call_ms"] < 100, result
    assert result["drained"] is True
    out = capsys.readouterr().out
    assert "event=stale_callback_ignored" in out
    assert True not in transitions
    assert engine_messages == []
    assert transcriber.calls == 1  # never rearmed during the drain


def test_j_shutdown_immediately_after_standby_entered_with_prewarm_running(capsys) -> None:
    transcriber = _StandbyTranscriber(mode="capture")
    prewarm = _SlowPrewarm(delay_seconds=0.3)
    transcriber.wake_recognizer = prewarm
    controller = _controller(transcriber)

    controller.startVoiceStandby()
    prewarm_thread = controller._wake_prewarm_thread
    assert prewarm_thread is not None
    result = _close(controller)

    assert result["call_ms"] < 100, result
    assert result["drained"] is True
    assert prewarm.calls == 1
    assert controller._wake_prewarm_thread is None
    out = capsys.readouterr().out
    assert "[SHUTDOWN] component=wake_prewarm event=stop_requested" in out
    assert "[SHUTDOWN] component=wake_prewarm event=finished" in out


def test_k_l_no_thread_left_running_and_no_invalid_wrapper_access() -> None:
    """K/L: after completion every owned QThread is either gone or finished
    (nothing for Qt to destroy while running), and calling shutdown()
    again touches no stale wrapper."""
    transcriber = _StandbyTranscriber(mode="busy", busy_seconds=0.2, result="")
    transcriber.wake_recognizer = _SlowPrewarm(delay_seconds=0.2)
    controller = _controller(transcriber)
    controller.startVoiceStandby()
    assert transcriber.started.wait(2)
    threads = [controller._voice_thread, controller._wake_prewarm_thread]

    assert _close(controller)["drained"] is True
    _run_until(lambda: False, timeout_ms=100)  # let deleteLater run

    for thread in threads:
        assert thread is not None
        assert not shiboken6.isValid(thread) or thread.isRunning() is False
    controller.shutdown()  # idempotent, must not raise


def test_n_shutdown_reaches_completed_and_runs_session_close_once_after_drain(capsys) -> None:
    transcriber = _StandbyTranscriber(mode="busy", busy_seconds=0.3, result="")
    tts = _TTSManager()
    controller = _controller(transcriber, tts_manager=tts)
    controller.startVoiceStandby()
    assert transcriber.started.wait(2)
    session_close_calls: list[list[str]] = []

    result = _close(controller, session_close=lambda: session_close_calls.append(controller.shutdown_pending_components()))

    assert result["drained"] is True
    assert result["completed"] == 1
    assert session_close_calls == [[]]  # once, and only after every worker drained
    assert tts.stop_calls == 1  # Pocket/TTS manager torn down even with no active TTS worker
    out = capsys.readouterr().out
    assert "[SHUTDOWN] component=session_close event=started" in out
    assert "[SHUTDOWN] component=session_close event=finished" in out
    assert out.count("[SHUTDOWN] event=completed") == 1
    assert "total_ms=" in out


def test_shutdown_async_twice_is_safe() -> None:
    transcriber = _StandbyTranscriber(mode="capture")
    controller = _controller(transcriber)
    controller.startVoiceStandby()
    assert transcriber.started.wait(2)

    assert _close(controller)["drained"] is True
    assert _close(controller)["drained"] is True
    assert controller.is_shutdown_complete() is True


def test_no_new_listening_after_shutdown_requested() -> None:
    transcriber = _StandbyTranscriber(mode="capture")
    controller = _controller(transcriber)
    controller.request_shutdown()

    controller._begin_listening()

    assert controller._voice_thread is None
    assert _close(controller)["drained"] is True
