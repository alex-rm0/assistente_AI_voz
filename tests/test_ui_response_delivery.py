"""Phase 8.1.1: response display regression in voice mode.

Real symptom: in Voice Mode the final response was generated, emitted
(RESPONSE_READY_EMITTED) and spoken by TTS, but never rendered -- and the
real log showed NO JS console line of any kind after
"voiceStandbyChanged recebido: true".

Root cause: STANDBY's wake-prewarm QThread connected its `finished` signal
to EchoUIController._on_wake_prewarm_finished, which was NOT a declared
@Slot. PySide then adds a dynamic slot to the controller's QMetaObject at
connect time -- and the controller is the object QWebChannel had already
published to JS with a fixed method table. From that moment every
controller -> JS signal stopped arriving (stateChanged, responseReady,
voiceConversationChanged ...), for typed AND voice turns alike.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "prototype_web_ui" / "controller.py"
DRIVER = Path(__file__).resolve().parent / "_webchannel_ui_driver.py"


def _qt_app():
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def _run_until(predicate, timeout_ms: int = 3000) -> bool:
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    deadline = time.perf_counter() + timeout_ms / 1000
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


# --- Root-cause guards (no WebEngine needed) ---------------------------------


def test_every_controller_method_connected_to_a_signal_is_a_declared_slot() -> None:
    """Any `<signal>.connect(self.<method>)` in the controller must target an
    @Slot-decorated method -- an undecorated one mutates the published
    QMetaObject and silently breaks QWebChannel delivery."""
    tree = ast.parse(CONTROLLER.read_text(encoding="utf-8"))
    controller_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EchoUIController"
    )
    methods = {node.name: node for node in controller_class.body if isinstance(node, ast.FunctionDef)}

    def is_slot(func: ast.FunctionDef) -> bool:
        for decorator in func.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Name) and target.id == "Slot":
                return True
        return False

    connected: set[str] = set()
    for node in ast.walk(controller_class):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "connect"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "self"
            and node.args[0].attr in methods
        ):
            connected.add(node.args[0].attr)

    assert "_on_wake_prewarm_finished" in connected
    undecorated = sorted(name for name in connected if not is_slot(methods[name]))
    assert undecorated == [], undecorated


class _PrewarmRecognizer:
    def prewarm(self) -> None:
        time.sleep(0.05)


class _StandbyTranscriber:
    model_name = "medium"
    standby_vad_config = object()
    vad_config = object()
    last_vad_outcome = None

    def __init__(self) -> None:
        self.wake_recognizer = _PrewarmRecognizer()

    def record_and_transcribe_wake(self, stop_event, status_callback=None, *, on_speech_confirmed=None) -> str:
        while not stop_event.is_set():
            time.sleep(0.005)
        from assistant.voice_input import VoiceNoSpeechDetected

        raise VoiceNoSpeechDetected("Nao detetei fala.")

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        raise AssertionError


def test_g_standby_prewarm_does_not_change_the_published_metaobject() -> None:
    """G: entering STANDBY (which starts and finishes the prewarm QThread)
    must leave the controller's QMetaObject exactly as QWebChannel saw it."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(
        lambda message: "Reply.",
        voice_available=True,
        voice_transcriber=_StandbyTranscriber(),
        voice_missing_dependencies=[],
        thinking_min_ms=0,
    )
    before = controller.metaObject().methodCount()

    controller.startVoiceStandby()
    assert _run_until(lambda: controller._wake_prewarm_thread is None, timeout_ms=3000)
    controller.stopVoiceStandby()
    assert _run_until(lambda: controller._voice_worker is None, timeout_ms=3000)

    assert controller.metaObject().methodCount() == before
    controller.shutdown()


# --- Real Qt / QWebEngine / QWebChannel integration ----------------------------


@pytest.fixture(scope="module")
def driver_run() -> dict[str, object]:
    if find_spec("PySide6.QtWebEngineWidgets") is None:
        pytest.skip("QtWebEngine not available")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [sys.executable, str(DRIVER)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"^DRIVER_RESULT (\{.*\})$", completed.stdout, re.MULTILINE)
    assert match is not None, output[-4000:]
    result = json.loads(match.group(1))
    lines = [line.split("message=", 1)[-1] for line in output.splitlines() if "[UI_RESPONSE]" in line]
    return {"result": result, "ui": lines, "output": output, "returncode": completed.returncode}


def _events(run, event: str) -> list[str]:
    return [line for line in run["ui"] if f"event={event}" in line]


def test_driver_completed_without_errors(driver_run) -> None:
    assert driver_run["result"]["errors"] == []
    assert driver_run["returncode"] == 0


def test_a_typed_response_reaches_js_and_renders(driver_run) -> None:
    snapshot = driver_run["result"]["snapshots"]["after_typed_first"]
    assert snapshot == {"text": "Resposta 1.", "visible": True}


def test_b_f_voice_active_response_reaches_js_and_renders(driver_run) -> None:
    emits = _events(driver_run, "emit")
    voice_emits = [line for line in emits if "source=request" in line and "voice_mode=active" in line]
    assert len(voice_emits) == 1
    snapshot = driver_run["result"]["snapshots"]["after_voice_rearm"]
    assert snapshot == {"text": "Resposta 2.", "visible": True}


def test_c_e_each_response_emitted_received_and_rendered_exactly_once(driver_run) -> None:
    # typed #1, wake acknowledgement, voice turn, typed #2
    assert len(_events(driver_run, "emit")) == 4
    assert len(_events(driver_run, "received")) == 4
    assert len(_events(driver_run, "render_start")) == 4
    completes = _events(driver_run, "render_complete")
    assert len(completes) == 4
    assert all("visible=true" in line for line in completes), completes


def test_d_tts_requested_exactly_once_per_spoken_response(driver_run) -> None:
    assert driver_run["result"]["spoken"] == ["I'm listening.", "Resposta 2."]


def test_g_controller_identity_is_stable_across_transitions(driver_run) -> None:
    ids = {re.search(r"controller_id=(\d+)", line).group(1) for line in _events(driver_run, "emit")}
    assert len(ids) == 1


def test_g_typed_after_voice_transitions_still_delivered(driver_run) -> None:
    snapshot = driver_run["result"]["snapshots"]["after_typed_second"]
    assert snapshot == {"text": "Resposta 3.", "visible": True}


def test_h_i_response_stays_visible_after_finish_and_next_listening_cycle(driver_run) -> None:
    """after_voice_rearm is taken after requestFinished, TTS completion and
    the rearm into a fresh listening cycle -- nothing may clear it."""
    assert driver_run["result"]["snapshots"]["after_voice_rearm"]["visible"] is True
    cleared = _events(driver_run, "cleared")
    assert all("reason=new_message" in line for line in cleared), cleared


def test_j_window_close_still_completes_async_shutdown(driver_run) -> None:
    assert "[SHUTDOWN] event=completed" in driver_run["output"]
