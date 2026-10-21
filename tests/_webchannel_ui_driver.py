"""Phase 8.1.1: real EchoOSWindow + QWebEngine + QWebChannel driver.

Run as a subprocess by tests/test_ui_response_delivery.py (the rest of the
suite already owns a QCoreApplication, and QtWebEngine needs its own
QApplication). Drives the REAL index.html/echo_ui.js through:

  1. a typed request (submitted through the page's own form),
  2. VOICE MODE -> STANDBY (with a wake-recognizer prewarm QThread -- the
     exact trigger of the real regression) -> wake -> one ACTIVE turn ->
     TTS -> rearm listening,
  3. voice mode off, then a second typed request,
  4. window close (the async shutdown path).

Prints every controller/JS log line as usual, plus one final
`DRIVER_RESULT {json}` line with DOM snapshots and the spoken texts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([sys.argv[0]])

from assistant.voice_input import VoiceNoSpeechDetected  # noqa: E402
from prototype_web_ui.window import EchoOSWindow  # noqa: E402

TYPED_TEXT = "Olá, como está?"


class _PrewarmingWakeRecognizer:
    def prewarm(self) -> None:
        time.sleep(0.5)


class _Transcriber:
    model_name = "medium"
    standby_vad_config = object()
    vad_config = object()
    barge_in_vad_config = object()
    last_vad_outcome = None

    def __init__(self) -> None:
        self.wake_recognizer = _PrewarmingWakeRecognizer()
        self.active_turns = 0

    def record_and_transcribe_wake(self, stop_event, status_callback=None, *, on_speech_confirmed=None) -> str:
        time.sleep(0.8)  # lets the prewarm thread finish first, as in the real log
        return "Acorda, Echo."

    def record_and_transcribe_vad(self, stop_event, status_callback=None, **kwargs) -> str:
        if not kwargs.get("require_confirmation_before_recording"):
            self.active_turns += 1
            if self.active_turns == 1:
                time.sleep(0.1)
                return TYPED_TEXT
        while not stop_event.is_set():
            time.sleep(0.01)
        raise VoiceNoSpeechDetected("Nao detetei fala.")

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        raise AssertionError("manual MIC path is not part of this scenario")


@dataclass
class _SpeakResult:
    audio_path: Path
    text_chars: int
    provider: str = "fake"
    voice_id: str = "fake_voice"
    provider_voice: str = "fake_voice"
    language: str = "pt-PT"
    model: str = "fake"
    device: str = "cpu"
    worker_launch_ms: float = 0.0
    model_load_ms: float = 0.0
    reference_prepare_ms: float = 0.0
    first_audio_ms: float = 0.0
    synthesis_ms: float = 0.0
    audio_duration_ms: float = 0.0
    reference_used: bool = False
    reference_cache_hit: bool = False
    fallback_used: bool = False
    fallback_reason: str = ""


class _TTSManager:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def is_available(self) -> bool:
        return True

    def speak_to_file(self, text: str, response_language: str = "") -> _SpeakResult:
        self.spoken.append(text)
        return _SpeakResult(audio_path=Path("unused.wav"), text_chars=len(text))

    def stop(self) -> None:
        pass


def _player(path, stop_checker, **kwargs) -> float:
    time.sleep(0.2)
    return 200.0


replies = {"count": 0}


def _responder(message: str) -> str:
    replies["count"] += 1
    time.sleep(0.1)
    return f"Resposta {replies['count']}."


tts = _TTSManager()
window = EchoOSWindow(responder=_responder, title="driver", voice_available=False, tts_manager=tts)
controller = window.controller
controller.voice_available = True
controller.voice_transcriber = _Transcriber()
controller.voice_missing_dependencies = []
controller._voice_runtime_status_provided = True
controller.tts_player = _player
controller.thinking_min_ms = 0
window.show()

result: dict[str, object] = {"snapshots": {}, "errors": []}
DOM_JS = (
    "(() => { const e = document.getElementById('echoResponse'); const s = getComputedStyle(e);"
    " return JSON.stringify({text: e.textContent.trim(), visible: !e.hidden && s.display !== 'none'"
    " && s.visibility !== 'hidden' && s.opacity !== '0'}); })()"
)


def _wait(predicate, then, *, timeout_ms: int = 8000, label: str = "") -> None:
    deadline = time.perf_counter() + timeout_ms / 1000

    def poll() -> None:
        if predicate():
            then()
        elif time.perf_counter() > deadline:
            result["errors"].append(f"timeout:{label}")
            finish()
        else:
            QTimer.singleShot(20, poll)

    poll()


def _snapshot(name: str, then) -> None:
    def store(raw) -> None:
        result["snapshots"][name] = json.loads(raw)
        then()

    window.page().runJavaScript(DOM_JS, store)


def _type(text: str) -> None:
    window.page().runJavaScript(
        "(() => { const i = document.getElementById('voiceIn'); i.value = %s; i.form.requestSubmit(); })()"
        % json.dumps(text)
    )


def step_wait_ready() -> None:
    state = {"ready": False}

    def check(value) -> None:
        state["ready"] = bool(value)

    def poll() -> None:
        window.page().runJavaScript("Boolean(window.echoController)", check)
        if state["ready"]:
            step_typed_first()
        else:
            QTimer.singleShot(50, poll)

    poll()


def step_typed_first() -> None:
    print("DRIVER step=typed_first")
    _type(TYPED_TEXT)
    _wait(lambda: replies["count"] >= 1 and not controller._active, lambda: QTimer.singleShot(300, after_typed_first), label="typed_first")


def after_typed_first() -> None:
    _snapshot("after_typed_first", step_voice)


def step_voice() -> None:
    print("DRIVER step=voice")
    window.page().runJavaScript("document.getElementById('voiceConversationButton').click()")
    _wait(
        lambda: replies["count"] >= 2 and len(tts.spoken) >= 2 and controller._tts_worker is None and controller._voice_worker is not None,
        lambda: QTimer.singleShot(700, after_voice_rearm),
        label="voice_turn",
    )


def after_voice_rearm() -> None:
    _snapshot("after_voice_rearm", step_voice_off)


def step_voice_off() -> None:
    print("DRIVER step=voice_off")
    controller.stopVoiceConversation()
    _wait(lambda: controller._voice_worker is None, step_typed_second, label="voice_off")


def step_typed_second() -> None:
    print("DRIVER step=typed_second")
    _type(TYPED_TEXT)
    _wait(lambda: replies["count"] >= 3 and not controller._active, lambda: QTimer.singleShot(300, after_typed_second), label="typed_second")


def after_typed_second() -> None:
    _snapshot("after_typed_second", finish)


def finish() -> None:
    result["spoken"] = list(tts.spoken)
    result["replies"] = replies["count"]
    print("DRIVER_RESULT " + json.dumps(result, ensure_ascii=True), flush=True)
    window.close()


QTimer.singleShot(0, step_wait_ready)
QTimer.singleShot(45000, lambda: (print("DRIVER_TIMEOUT", flush=True), app.exit(3)))
sys.exit(app.exec())
