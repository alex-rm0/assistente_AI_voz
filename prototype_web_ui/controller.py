from __future__ import annotations

import time
import os
import traceback
import json
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot

from assistant.secret_storage import sanitize_secret_error
from assistant.tts import TTSError, play_audio_file
from assistant.voice_input import VoiceInputError, VoiceNoSpeechDetected, VoiceTranscriber, check_voice_runtime

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


class EchoVoiceWorker(QObject):
    transcriptionReady = Signal(str)
    failed = Signal(str)
    statusChanged = Signal(str)

    def __init__(self, transcriber: VoiceTranscriber) -> None:
        super().__init__()
        self.transcriber = transcriber
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    @Slot()
    def run(self) -> None:
        try:
            text = self.transcriber.record_and_transcribe(self.stop_event, self.statusChanged.emit)
            self.statusChanged.emit("Concluído.")
            self.transcriptionReady.emit(text)
        except VoiceNoSpeechDetected:
            self.statusChanged.emit("Não detetei fala.")
            self.transcriptionReady.emit("")
        except Exception as exc:
            self.failed.emit(_bounded_voice_error(exc))


class EchoTTSWorker(QObject):
    playbackStarted = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        tts_manager: Any,
        text: str,
        player: Callable[..., float] = play_audio_file,
        response_language: str = "",
    ) -> None:
        super().__init__()
        self.tts_manager = tts_manager
        self.text = text
        self.player = player
        self.response_language = response_language
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()
        stop = getattr(self.tts_manager, "stop", None)
        if callable(stop):
            stop()

    @Slot()
    def run(self) -> None:
        try:
            try:
                result = self.tts_manager.speak_to_file(self.text, response_language=self.response_language)
            except TypeError:
                result = self.tts_manager.speak_to_file(self.text)
            if self.stop_event.is_set():
                self.finished.emit(_json_payload({"stopped": True}))
                return
            self.playbackStarted.emit(
                _json_payload(
                    {
                        "provider": result.provider,
                        "voice_id": result.voice_id,
                        "language": self.response_language or result.language,
                        "response_language": self.response_language,
                        "voice_language": result.language,
                        "synthesis_ms": int(result.synthesis_ms),
                        "text_chars": result.text_chars,
                        "fallback_used": result.fallback_used,
                        "fallback_reason": result.fallback_reason,
                    }
                )
            )
            playback_ms = self.player(result.audio_path, self.stop_event.is_set)
            self.finished.emit(_json_payload({"playback_ms": int(playback_ms), "voice_id": result.voice_id}))
        except Exception as exc:
            self.failed.emit(_bounded_tts_error(exc))


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
    voiceTranscriptionReady = Signal(str)
    voiceInputError = Signal(str)
    voiceInputStarted = Signal()
    voiceInputFinished = Signal()
    ttsStarted = Signal(str)
    ttsFinished = Signal(str)
    ttsError = Signal(str)

    def __init__(
        self,
        responder: Callable[[str], str],
        parent: QObject | None = None,
        *,
        clear_conversation: Callable[[], None] | None = None,
        get_telemetry: Callable[[], dict | None] | None = None,
        model_runtime: Any | None = None,
        voice_available: bool = False,
        voice_transcriber: VoiceTranscriber | None = None,
        voice_missing_dependencies: list[str] | tuple[str, ...] | None = None,
        tts_manager: Any | None = None,
        tts_player: Callable[..., float] = play_audio_file,
        get_response_language: Callable[[], str] | None = None,
        thinking_min_ms: int = 300,
    ) -> None:
        super().__init__(parent)
        self.responder = responder
        self.clear_conversation_callback = clear_conversation
        self.get_telemetry_callback = get_telemetry
        self.model_runtime = model_runtime
        self.thinking_min_ms = thinking_min_ms
        self._thread: QThread | None = None
        self._worker: EchoRequestWorker | None = None
        self._voice_thread: QThread | None = None
        self._voice_worker: EchoVoiceWorker | None = None
        self._tts_thread: QThread | None = None
        self._tts_worker: EchoTTSWorker | None = None
        self.tts_manager = tts_manager
        self.tts_player = tts_player
        self.get_response_language_callback = get_response_language
        self.voice_available = bool(voice_available)
        self.voice_transcriber = voice_transcriber
        self._voice_runtime_status_provided = voice_missing_dependencies is not None
        self.voice_missing_dependencies = list(voice_missing_dependencies or [])
        self._active = False
        self._started_at = 0.0
        self._visual_token = 0
        # Set by _handle_progress with the most recent rewrite progress
        # event for this request; consulted by _handle_response to decide
        # whether the normal thinking->speaking->idle animation should be
        # skipped in favour of going straight to idle (cancellation) or a
        # brief error flash (timeout) -- see _handle_response.
        self._last_progress_event = ""

    @Slot(str)
    def submitMessage(self, text: str) -> None:
        message = (text or "").strip()
        _lifecycle_trace("REQUEST_BEGIN")
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
        self._last_progress_event = ""
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
    def startVoiceInput(self) -> None:
        self._thread_debug("startVoiceInput")
        if self._voice_worker is not None:
            self.cancelVoiceInput()
            return
        if self._active:
            self.voiceInputError.emit("Espera pela resposta atual terminar antes de gravares.")
            return
        if not self.voice_available or self.voice_transcriber is None:
            self.voiceInputError.emit("A voz não está disponível nesta sessão.")
            self._emit_state("idle")
            return
        missing = self.voice_missing_dependencies if self._voice_runtime_status_provided else check_voice_runtime()
        if missing:
            self.voiceInputError.emit("A voz não está pronta porque falta: " + ", ".join(missing))
            self._emit_state("idle")
            return

        self._visual_token += 1
        self._voice_thread = QThread(self)
        self._voice_worker = EchoVoiceWorker(self.voice_transcriber)
        self._voice_worker.moveToThread(self._voice_thread)
        self._voice_thread.started.connect(self._voice_worker.run, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.statusChanged.connect(self._handle_voice_status, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.transcriptionReady.connect(self._handle_voice_transcription, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.failed.connect(self._handle_voice_error, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.transcriptionReady.connect(self._cleanup_voice_worker, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.failed.connect(self._cleanup_voice_worker, Qt.ConnectionType.QueuedConnection)
        self.voiceInputStarted.emit()
        self._emit_state("listening")
        self._voice_thread.start()

    @Slot()
    def cancelVoiceInput(self) -> None:
        self._thread_debug("cancelVoiceInput")
        if self._voice_worker is None:
            return
        self._handle_voice_status("A processar...")
        self._voice_worker.stop()

    @Slot(result=bool)
    def isVoiceInputAvailable(self) -> bool:
        return bool(self.voice_available and self.voice_transcriber is not None and not self.voice_missing_dependencies)

    @Slot(result=bool)
    def isTtsAvailable(self) -> bool:
        if self.tts_manager is None:
            return False
        manager_available = getattr(self.tts_manager, "is_available", None)
        if callable(manager_available):
            try:
                return bool(manager_available(response_language=self._response_language()))
            except TypeError:
                return bool(manager_available())
        provider = getattr(self.tts_manager, "provider", None)
        available = getattr(provider, "is_available", None)
        return bool(callable(available) and available())

    @Slot(str)
    def speakText(self, text: str) -> None:
        self._thread_debug("speakText")
        message = str(text or "").strip()
        if not message:
            self.ttsError.emit("Não há resposta para falar.")
            return
        if self.tts_manager is None:
            self.ttsError.emit("A fala ainda não está ligada nesta sessão.")
            return
        manager_available = getattr(self.tts_manager, "is_available", None)
        response_language = self._response_language()
        if callable(manager_available):
            try:
                available = bool(manager_available(response_language=response_language))
            except TypeError:
                available = bool(manager_available())
            if not available:
                self.ttsError.emit(_tts_unavailable_message(response_language))
                self._emit_state("idle")
                return
        if self._tts_worker is not None:
            self.stopSpeaking()
        self._visual_token += 1
        self._tts_thread = QThread(self)
        self._tts_worker = EchoTTSWorker(self.tts_manager, message, self.tts_player, response_language)
        self._tts_worker.moveToThread(self._tts_thread)
        self._tts_thread.started.connect(self._tts_worker.run, Qt.ConnectionType.QueuedConnection)
        self._tts_worker.playbackStarted.connect(self._handle_tts_started, Qt.ConnectionType.QueuedConnection)
        self._tts_worker.finished.connect(self._handle_tts_finished, Qt.ConnectionType.QueuedConnection)
        self._tts_worker.failed.connect(self._handle_tts_error, Qt.ConnectionType.QueuedConnection)
        _debug_tts("[TTS]", "event=request", f"text_chars={len(message)}", f"response_language={response_language}")
        self._tts_thread.start()

    @Slot()
    def stopSpeaking(self) -> None:
        self._thread_debug("stopSpeaking")
        if self._tts_worker is None:
            return
        self._tts_worker.stop()
        self._emit_state("idle")

    def _response_language(self) -> str:
        if callable(self.get_response_language_callback):
            try:
                return str(self.get_response_language_callback() or "")
            except Exception:
                return ""
        return ""

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
    def reportWorkspaceCapacity(self, payload: str) -> None:
        """Phase 4 (Cognitive Workspace Blueprint v2, section 11): the
        ONLY frontend -> backend layout signal. `payload` is a JSON object
        with exactly {pressure, visible_object_count, major_surface_count,
        overflow_risk} -- semantic capacity, never a pixel or coordinate
        (echo_workspace_objects.js computes this; see its own comments for
        why no geometry field exists to send one in). Fire-and-forget: no
        response/uiEvent is expected back for this call itself, the signal
        is simply stored for the NEXT turn's policy checks to read."""
        owner = getattr(self.responder, "__self__", None)
        handler = getattr(owner, "report_workspace_capacity", None)
        if not callable(handler):
            return
        try:
            data = json.loads(str(payload or "{}"))
        except json.JSONDecodeError:
            return
        try:
            handler(data)
        except Exception as exc:
            _debug_ui(f"[Echo UI DEBUG] workspace_capacity_error={exc}")

    @Slot(str)
    def sendTestResponse(self, text: str) -> None:
        self._thread_debug("sendTestResponse")
        response_text = str(text or "")
        self._emit_response_ready(response_text)

    def has_active_request(self) -> bool:
        return self._active

    def shutdown(self, wait_ms: int = 1500) -> None:
        if self._voice_worker is not None:
            self._voice_worker.stop()
            self._cleanup_voice_worker()
        if self._tts_worker is not None:
            self._tts_worker.stop()
            self._cleanup_tts_worker()
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
        try:
            response_text = str(response or "")
            self._emit_response_ready(response_text)
            self._emit_pending_ui_events()
            self._emit_runtime_telemetry("response_ready")
            last_progress_event = self._last_progress_event
            self._last_progress_event = ""
            if last_progress_event == "rewrite_cancelled":
                # The request already resolved to "Pedido cancelado." -- go
                # straight to idle instead of playing the normal thinking->
                # speaking animation over a response nobody asked to see spoken.
                self._emit_state("idle")
            elif last_progress_event == "rewrite_timeout":
                self._emit_state("error")
                QTimer.singleShot(900, lambda: self._emit_state("idle"))
            else:
                elapsed_ms = int((time.perf_counter() - self._started_at) * 1000) if self._started_at else 0
                delay_ms = max(0, self.thinking_min_ms - elapsed_ms)
                QTimer.singleShot(delay_ms, lambda: self._emit_state("idle"))
        except Exception as exc:
            _debug_ui(f"[Echo UI DEBUG] response_finalization_error={exc}")
            self._emit_state("error")
            QTimer.singleShot(900, lambda: self._emit_state("idle"))
        finally:
            self._cleanup_worker()

    @Slot(str)
    def _handle_error(self, error: str) -> None:
        self._thread_debug("_handle_error")
        _debug_ui(f"[Echo UI] error={error}")
        error_text = "Não consegui responder a este pedido."
        try:
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
        except Exception as exc:
            _debug_ui(f"[Echo UI DEBUG] error_finalization_error={exc}")
            QTimer.singleShot(0, lambda: self._emit_state("idle"))
        finally:
            self._cleanup_worker()

    @Slot(str)
    def _handle_progress(self, event: str) -> None:
        self._thread_debug("_handle_progress")
        _debug_ui(f"[Echo UI] progress_event={event}")
        self._last_progress_event = str(event or "")
        self.uiEvent.emit(_json_payload({"type": str(event or "")}))

    @Slot(str)
    def _handle_voice_status(self, status: str) -> None:
        clean_status = str(status or "").strip()
        lowered = clean_status.lower()
        if "ouvir" in lowered:
            self._emit_state("listening")
        elif "process" in lowered or "conclu" in lowered:
            self._emit_state("thinking")
        self.uiEvent.emit(_json_payload({"type": "voice_status", "status": clean_status}))

    @Slot(str)
    def _handle_voice_transcription(self, text: str) -> None:
        self._thread_debug("_handle_voice_transcription")
        self.voiceTranscriptionReady.emit(str(text or ""))
        self._emit_state("idle")

    @Slot(str)
    def _handle_voice_error(self, error: str) -> None:
        self._thread_debug("_handle_voice_error")
        error_text = str(error or "Não consegui transcrever o áudio.")
        self.voiceInputError.emit(error_text)
        self._emit_state("error")
        QTimer.singleShot(900, lambda: self._emit_state("idle"))

    @Slot()
    def _cleanup_voice_worker(self) -> None:
        self._thread_debug("_cleanup_voice_worker")
        if self._voice_thread is not None:
            self._voice_thread.quit()
            self._voice_thread.wait()
            self._voice_thread.deleteLater()
        if self._voice_worker is not None:
            self._voice_worker.deleteLater()
        self._voice_thread = None
        self._voice_worker = None
        self.voiceInputFinished.emit()

    @Slot(str)
    def _handle_tts_started(self, payload: str) -> None:
        self._thread_debug("_handle_tts_started")
        _debug_tts("[TTS]", "event=playback_started", _bounded_tts_payload(payload))
        self._emit_state("speaking")
        self.ttsStarted.emit(str(payload or "{}"))

    @Slot(str)
    def _handle_tts_finished(self, payload: str) -> None:
        self._thread_debug("_handle_tts_finished")
        _debug_tts("[TTS]", "event=playback_finished", _bounded_tts_payload(payload), "success=true")
        self._emit_state("idle")
        self._cleanup_tts_worker()
        self.ttsFinished.emit(str(payload or "{}"))

    @Slot(str)
    def _handle_tts_error(self, error: str) -> None:
        self._thread_debug("_handle_tts_error")
        error_text = str(error or "Não consegui falar a resposta.")
        _debug_tts("[TTS]", "event=error", "success=false", f"failure_reason={error_text[:160]}")
        self._cleanup_tts_worker()
        self.ttsError.emit(error_text)
        if _is_tts_capability_unavailable(error_text):
            self._emit_state("idle")
            return
        self._emit_state("error")
        QTimer.singleShot(900, lambda: self._emit_state("idle"))

    @Slot()
    def _cleanup_tts_worker(self) -> None:
        self._thread_debug("_cleanup_tts_worker")
        if self._tts_thread is not None:
            self._tts_thread.quit()
            self._tts_thread.wait()
            self._tts_thread.deleteLater()
        if self._tts_worker is not None:
            self._tts_worker.deleteLater()
        self._tts_thread = None
        self._tts_worker = None

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
        _lifecycle_trace("REQUEST_FINISHED_EMITTED")
        self.requestFinished.emit()

    def _emit_response_ready(self, response_text: str) -> None:
        _debug_ui(
            "[Echo UI DEBUG] emitting responseReady",
            repr(response_text),
            type(response_text),
            f"controller_id={id(self)}",
        )
        _lifecycle_trace("RESPONSE_READY_EMITTED")
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

    # The 9 semantic states from the visual-state redesign. "listening" has
    # no emitter yet (no voice-capture state machine exists) -- it's
    # whitelisted so the existing setState QWebChannel slot can already
    # drive it once that's wired up, without touching this whitelist again.
    _KNOWN_STATES = {
        "idle", "listening", "thinking", "reading", "working", "speaking", "error",
    }

    def _emit_state(self, state: str) -> None:
        clean_state = (state or "").strip().lower()
        if clean_state not in self._KNOWN_STATES:
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
        try:
            if state == "error":
                payload = self.model_runtime.error_payload(error_message or "Erro.", provider_error_type="ui_error")
                encoded = _json_payload(payload)
            else:
                encoded = self._runtime_payload_json(state)
            _debug_ui(f"[Echo UI DEBUG] emitting telemetryUpdated {encoded}")
            self.telemetryUpdated.emit(encoded)
        except Exception as exc:
            _debug_ui(f"[Echo UI DEBUG] telemetry_error={exc}")

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


def _debug_tts(*parts: object) -> None:
    print(*parts)


def _lifecycle_trace(event: str) -> None:
    print("[LIFECYCLE]", event)


def _json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _bounded_voice_error(exc: Exception) -> str:
    text = sanitize_secret_error(exc)
    if isinstance(exc, VoiceInputError):
        return text[:240]
    return f"Não consegui usar a voz: {text}"[:240]


def _bounded_tts_error(exc: Exception) -> str:
    text = sanitize_secret_error(exc)
    if isinstance(exc, TTSError):
        return text[:240]
    return f"Não consegui falar a resposta: {text}"[:240]


def _bounded_tts_payload(payload: str) -> str:
    try:
        data = json.loads(str(payload or "{}"))
    except json.JSONDecodeError:
        return "payload=invalid"
    fields = []
    for key in (
        "provider",
        "voice_id",
        "language",
        "response_language",
        "voice_language",
        "text_chars",
        "synthesis_ms",
        "playback_ms",
        "fallback_used",
        "fallback_reason",
    ):
        if key in data:
            fields.append(f"{key}={str(data.get(key))[:80]}")
    return " ".join(fields)


def _is_tts_capability_unavailable(message: str) -> bool:
    text = str(message or "").strip().lower()
    return any(
        marker in text
        for marker in (
            "nenhuma voz tts local",
            "voz tts selecionada não está disponível",
            "fala ainda não está ligada",
            "não encontrei uma voz tts local",
            "voz portuguesa ainda não está disponível",
        )
    )


def _tts_unavailable_message(response_language: str) -> str:
    if str(response_language or "").strip() == "pt-PT":
        return "A voz portuguesa ainda não está disponível."
    return "Não encontrei uma voz TTS local compatível."
