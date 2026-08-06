from __future__ import annotations

import time
import os
import traceback
import json
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot

from assistant.secret_storage import sanitize_secret_error

class EchoRequestWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, responder: Callable[[str], str], message: str) -> None:
        super().__init__()
        self.responder = responder
        self.message = message

    @Slot()
    def run(self) -> None:
        owner = getattr(self.responder, "__self__", None)
        set_listener = getattr(owner, "set_progress_listener", None)
        if callable(set_listener):
            set_listener(self.progress.emit)
        try:
            self.finished.emit(str(self.responder(self.message) or ""))
        except Exception as exc:
            print("[ECHO ERROR] stage=EchoRequestWorker.run")
            print("[ECHO ERROR] type=", type(exc).__name__)
            print("[ECHO ERROR] message=", str(exc))
            print("[ECHO ERROR] user_message=", self.message)
            traceback.print_exc()
            self.failed.emit(str(exc))
        finally:
            if callable(set_listener):
                set_listener(None)


class EchoUIController(QObject):
    """QWebChannel-facing controller for the Echo OS web UI.

    Public QWebChannel signals must be emitted by this controller in the GUI
    thread. The worker only emits private completion signals back to this object.
    """

    responseReady = Signal(str)
    stateChanged = Signal(str)
    errorOccurred = Signal(str)
    uiEvent = Signal(str)
    telemetryUpdated = Signal(str)
    modelModeChanged = Signal(str)
    requestStarted = Signal(str)
    requestFinished = Signal()

    def __init__(
        self,
        responder: Callable[[str], str],
        parent: QObject | None = None,
        *,
        clear_conversation: Callable[[], None] | None = None,
        get_telemetry: Callable[[], dict | None] | None = None,
        model_runtime: Any | None = None,
        thinking_min_ms: int = 300,
        speaking_duration_ms: int = 1000,
    ) -> None:
        super().__init__(parent)
        self.responder = responder
        self.clear_conversation_callback = clear_conversation
        self.get_telemetry_callback = get_telemetry
        self.model_runtime = model_runtime
        self.thinking_min_ms = thinking_min_ms
        self.speaking_duration_ms = speaking_duration_ms
        self._thread: QThread | None = None
        self._worker: EchoRequestWorker | None = None
        self._active = False
        self._started_at = 0.0
        self._visual_token = 0

    @Slot(str)
    def submitMessage(self, text: str) -> None:
        message = (text or "").strip()
        _debug_ui(f"[Echo UI] message_submitted={message}")
        self._thread_debug("submitMessage")
        if not message:
            self.errorOccurred.emit("Escreve uma mensagem primeiro.")
            self._emit_state("error")
            QTimer.singleShot(900, lambda: self._emit_state("idle"))
            return
        if self._active:
            _debug_ui("[Echo UI] duplicate_submission=ignored")
            return

        self._active = True
        self._started_at = time.perf_counter()
        self._visual_token += 1
        self.requestStarted.emit(message)
        self._emit_state("thinking")
        self._emit_runtime_telemetry("request_started")
        _debug_ui("[Echo UI] request_started=true")

        self._thread = QThread(self)
        self._worker = EchoRequestWorker(self.responder, message)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._handle_response, Qt.ConnectionType.QueuedConnection)
        self._worker.failed.connect(self._handle_error, Qt.ConnectionType.QueuedConnection)
        self._worker.progress.connect(self._handle_progress, Qt.ConnectionType.QueuedConnection)
        self._thread.start()

    @Slot(str)
    def receiveMessage(self, text: str) -> None:
        """Compatibility slot for the first prototype frontend."""

        self.submitMessage(text)

    @Slot()
    def clearConversation(self) -> None:
        if self.clear_conversation_callback is not None:
            self.clear_conversation_callback()
        self._emit_pending_ui_events()
        self.requestFinished.emit()

    @Slot()
    def cancelCurrentRequest(self) -> None:
        if not self._active:
            return
        owner = getattr(self.responder, "__self__", None)
        cancel = getattr(owner, "cancel_current_request", None)
        if callable(cancel):
            cancel()
            _debug_ui("[Echo UI] cancel_requested=true")
            self.uiEvent.emit(_json_payload({"type": "cancel_requested"}))
        else:
            _debug_ui("[Echo UI] cancel_requested=unsupported")

    @Slot(str)
    def setState(self, state: str) -> None:
        self._emit_state(state)

    @Slot(result=str)
    def getModelTelemetry(self) -> str:
        return self._runtime_payload_json("current")

    @Slot(str, result=str)
    def setModelMode(self, mode: str) -> str:
        return self.executeSystemCommand(_json_payload({"intent": "set_model_mode", "parameters": {"mode": mode}, "source": "ui"}))

    @Slot(bool, result=str)
    def setAutomaticClaudeEnabled(self, enabled: bool) -> str:
        return self.executeSystemCommand(
            _json_payload({"intent": "set_automatic_claude_enabled", "parameters": {"enabled": bool(enabled)}, "source": "ui"})
        )

    @Slot(float, float, result=str)
    def setModelBudget(self, daily_budget_usd: float, max_single_call_estimated_usd: float) -> str:
        return self.executeSystemCommand(
            _json_payload(
                {
                    "intent": "set_model_budget",
                    "parameters": {
                        "daily_budget_usd": daily_budget_usd,
                        "max_single_call_estimated_usd": max_single_call_estimated_usd,
                    },
                    "source": "ui",
                }
            )
        )

    @Slot(str, result=str)
    def executeSystemCommand(self, payload: str) -> str:
        if self.model_runtime is None:
            payload = {"state": "error", "note": "O runtime de modelos não está disponível."}
        else:
            try:
                command = json.loads(str(payload or "{}"))
            except json.JSONDecodeError:
                command = {"intent": str(payload or ""), "source": "ui"}
            try:
                payload = self.model_runtime.execute_command(command, source="ui")
            except Exception as exc:
                payload = {
                    "state": "error",
                    "note": sanitize_secret_error(exc),
                    "provider_error_type": "system_command_failed",
                }
        encoded = _json_payload(payload)
        if str(payload.get("state") or "") == "mode_changed":
            self.modelModeChanged.emit(encoded)
        self.telemetryUpdated.emit(encoded)
        return encoded

    @Slot(str)
    def sendTestResponse(self, text: str) -> None:
        self._thread_debug("sendTestResponse")
        response_text = str(text or "")
        self._emit_response_ready(response_text)

    def has_active_request(self) -> bool:
        return self._active

    def shutdown(self, wait_ms: int = 1500) -> None:
        if self._thread is None:
            return
        self._thread.requestInterruption()
        self._thread.quit()
        if not self._thread.wait(wait_ms):
            _debug_ui("[Echo UI] shutdown=worker_still_running")

    @Slot(str)
    def _handle_response(self, response: str) -> None:
        self._thread_debug("_handle_response")
        _debug_ui("[Echo UI] response_received=true")
        response_text = str(response or "")
        self._emit_response_ready(response_text)
        self._emit_pending_ui_events()
        self._emit_runtime_telemetry("response_ready")
        elapsed_ms = int((time.perf_counter() - self._started_at) * 1000) if self._started_at else 0
        delay_ms = max(0, self.thinking_min_ms - elapsed_ms)
        token = self._visual_token
        QTimer.singleShot(delay_ms, lambda: self._enter_speaking(token))
        self._cleanup_worker()

    @Slot(str)
    def _handle_error(self, error: str) -> None:
        self._thread_debug("_handle_error")
        _debug_ui(f"[Echo UI] error={error}")
        error_text = "Não consegui responder a este pedido."
        _debug_ui(
            "[Echo UI DEBUG] emitting errorOccurred",
            repr(error_text),
            type(error_text),
            f"controller_id={id(self)}",
        )
        self.errorOccurred.emit(error_text)
        self._emit_state("error")
        self._emit_runtime_telemetry("error", error_text)
        QTimer.singleShot(1200, lambda: self._emit_state("idle"))
        self._cleanup_worker()

    @Slot(str)
    def _handle_progress(self, event: str) -> None:
        self._thread_debug("_handle_progress")
        _debug_ui(f"[Echo UI] progress_event={event}")
        self.uiEvent.emit(_json_payload({"type": str(event or "")}))

    @Slot(int)
    def _enter_speaking(self, token: int) -> None:
        self._thread_debug("_enter_speaking")
        if token != self._visual_token:
            return
        self._emit_state("speaking")
        QTimer.singleShot(self.speaking_duration_ms, lambda: self._finish_speaking(token))

    @Slot(int)
    def _finish_speaking(self, token: int) -> None:
        self._thread_debug("_finish_speaking")
        if token != self._visual_token:
            return
        self._emit_state("idle")

    @Slot()
    def _cleanup_worker(self) -> None:
        self._thread_debug("_cleanup_worker")
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self._active = False
        duration_ms = int((time.perf_counter() - self._started_at) * 1000) if self._started_at else 0
        _debug_ui(f"[Echo UI] request_duration_ms={duration_ms}")
        self.requestFinished.emit()

    def _emit_response_ready(self, response_text: str) -> None:
        _debug_ui(
            "[Echo UI DEBUG] emitting responseReady",
            repr(response_text),
            type(response_text),
            f"controller_id={id(self)}",
        )
        self.responseReady.emit(response_text)

    def _emit_pending_ui_events(self) -> None:
        owner = getattr(self.responder, "__self__", None)
        consume = getattr(owner, "consume_ui_events", None)
        if consume is None:
            return
        try:
            events = consume()
        except Exception as exc:
            _debug_ui(f"[Echo UI DEBUG] ui_events_error={exc}")
            return
        for event in events:
            payload = str(event or "")
            _debug_ui(f"[Echo UI DEBUG] emitting uiEvent {payload}")
            self.uiEvent.emit(payload)

    def _emit_state(self, state: str) -> None:
        clean_state = (state or "").strip().lower()
        if clean_state not in {"idle", "thinking", "speaking", "error"}:
            clean_state = "error"
        _debug_ui(f"[Echo UI] state={clean_state}")
        _debug_ui(f"[Echo UI DEBUG] emitting stateChanged {clean_state!r} controller_id={id(self)}")
        self.stateChanged.emit(clean_state)

    def _runtime_payload_json(self, state: str) -> str:
        if self.model_runtime is None:
            return _json_payload({"state": state, "note": "O runtime de modelos não está disponível."})
        if state == "request_started":
            return _json_payload(self.model_runtime.request_started_payload())
        telemetry = self.get_telemetry_callback() if self.get_telemetry_callback is not None else None
        if telemetry:
            return _json_payload(self.model_runtime.telemetry_payload(telemetry, state=state))
        return _json_payload(self.model_runtime.current_payload(state=state))

    def _emit_runtime_telemetry(self, state: str, error_message: str = "") -> None:
        if self.model_runtime is None:
            return
        if state == "error":
            payload = self.model_runtime.error_payload(error_message or "Erro.", provider_error_type="ui_error")
            encoded = _json_payload(payload)
        else:
            encoded = self._runtime_payload_json(state)
        _debug_ui(f"[Echo UI DEBUG] emitting telemetryUpdated {encoded}")
        self.telemetryUpdated.emit(encoded)

    def _thread_debug(self, label: str) -> None:
        current_thread = QThread.currentThread()
        controller_thread = self.thread()
        worker_thread = self._thread
        _debug_threads(
            f"[Echo UI THREAD] {label} "
            f"current_id={id(current_thread)} "
            f"controller_thread_id={id(controller_thread)} "
            f"worker_thread_id={id(worker_thread) if worker_thread is not None else None} "
            f"on_controller_thread={current_thread == controller_thread}"
        )


def _debug_ui(*parts: object) -> None:
    if os.environ.get("ECHO_DEBUG_UI", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(*parts)


def _debug_threads(*parts: object) -> None:
    if os.environ.get("ECHO_DEBUG_THREADS", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(*parts)


def _json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
