from __future__ import annotations

import time
import os
import traceback
import json
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import shiboken6
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot

from assistant.model_warmup import ModelWarmup, find_ollama_provider, unload_ollama_in_background
from assistant.secret_storage import sanitize_secret_error
from assistant.response_language import localize
from assistant.tts import TTSError, play_audio_file
from assistant.voice_input import VoiceInputError, VoiceNoSpeechDetected, VoiceTranscriber, check_voice_runtime
from assistant.voice_wake import classify_voice_wake_intent, wake_acknowledgement_text
from assistant.turn_input import CaptureScope, TurnInput


class _WakeRecognizerPrewarmThread(QThread):
    """Phase 7.6.6 (section 6): runs a wake recognizer's prewarm() call on
    its own QThread, never blocking the caller (entering STANDBY). A plain
    QThread subclass with an overridden run() -- no signals needed, since
    prewarm()'s only effect is warming the SHARED WakeRecognizer's own
    internal (lock-guarded) model cache, which a later real
    record_and_transcribe_wake() call (on ITS OWN worker thread) observes
    naturally through that same shared object."""

    def __init__(self, prewarm: Callable[[], None], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._prewarm = prewarm

    def run(self) -> None:
        try:
            self._prewarm()
        except Exception:
            pass  # a failed prewarm just means the first real wake attempt loads it instead


class EchoRequestWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, responder: Callable[[str], str], message: str, *, turn_input: TurnInput | None = None) -> None:
        super().__init__()
        self.responder = responder
        self.message = message
        self.turn_input = turn_input or TurnInput.typed(message)

    @Slot()
    def run(self) -> None:
        owner = getattr(self.responder, "__self__", None)
        set_listener = getattr(owner, "set_progress_listener", None)
        if callable(set_listener):
            set_listener(self.progress.emit)
        try:
            respond_turn = getattr(owner, "respond_turn", None)
            response = respond_turn(self.turn_input) if callable(respond_turn) else self.responder(self.message)
            self.finished.emit(str(response or ""))
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
    transcriptionReady = Signal(object)
    failed = Signal(str)
    statusChanged = Signal(str)
    # Phase 7.6 (section 8/9): emitted exactly once, from the audio-capture
    # thread via record_until_speech_complete's on_speech_confirmed
    # callback, the instant a barge-in-mode worker confirms genuine user
    # speech -- Qt marshals this safely across threads like every other
    # signal here. Only meaningful when mode == "barge_in".
    speechConfirmed = Signal()

    def __init__(self, transcriber: VoiceTranscriber, *, mode: str = "manual", session_id=None) -> None:
        super().__init__()
        self.transcriber = transcriber
        self.stop_event = threading.Event()
        # Phase 7.5/7.6 (section 1/13): "manual" keeps manual MIC's existing
        # click-to-stop behavior completely unchanged. "active"/"standby"
        # get automatic end-of-speech detection (record_and_transcribe_vad).
        # "barge_in" also uses VAD, but with a conservative barge-in-
        # specific config and the speech-confirmed handoff above. Duck-
        # typed against the transcriber's actual capabilities rather than a
        # hard attribute check, so a transcriber double lacking the VAD/
        # barge-in interface (e.g. existing test fakes) safely falls back
        # instead of crashing -- no fake needs to be updated just to keep
        # working.
        self.mode = mode
        self.session_id = session_id
        from uuid import uuid4
        self.capture_id = uuid4().hex

    def stop(self) -> None:
        self.stop_event.set()

    @Slot()
    def run(self) -> None:
        with CaptureScope(session_id=self.session_id, mode=self.mode, capture_id=self.capture_id) as capture:
            self._run_capture(capture)

    def _run_capture(self, capture) -> None:
        try:
            vad_method = getattr(self.transcriber, "record_and_transcribe_vad", None)
            wake_method = getattr(self.transcriber, "record_and_transcribe_wake", None)
            if self.mode == "barge_in" and callable(vad_method):
                text = vad_method(
                    self.stop_event,
                    self.statusChanged.emit,
                    vad_config=getattr(self.transcriber, "barge_in_vad_config", None),
                    on_speech_confirmed=self.speechConfirmed.emit,
                    require_confirmation_before_recording=True,
                )
            elif self.mode == "standby" and callable(wake_method):
                # Phase 7.6.6 (section 3): STANDBY's dedicated fast wake-
                # recognition path -- a small, cheap model tries first, and
                # only falls back to the transcriber's own Medium model
                # once if that doesn't read as a wake command. Duck-typed
                # (like vad_method below) so a transcriber double lacking
                # this method (every existing test fake) safely falls
                # through to the Phase 7.6.1 path just below instead of
                # crashing -- no existing fake needs updating.
                text = wake_method(self.stop_event, self.statusChanged.emit)
            elif self.mode == "standby" and callable(vad_method):
                # Phase 7.6.1 (section 2/3): standby's wake/sleep-word
                # capture uses its OWN, more permissive VAD config -- a
                # short wake word like "Echo" is genuinely too brief for
                # the normal conversational min_speech_ms. ACTIVE mode
                # (below) is deliberately left untouched.
                text = vad_method(
                    self.stop_event,
                    self.statusChanged.emit,
                    vad_config=getattr(self.transcriber, "standby_vad_config", None),
                )
            elif self.mode == "active" and callable(vad_method):
                text = vad_method(self.stop_event, self.statusChanged.emit)
            else:
                text = self.transcriber.record_and_transcribe(self.stop_event, self.statusChanged.emit)
            self.statusChanged.emit("Concluído.")
            self.transcriptionReady.emit(capture.finish(text))
        except VoiceNoSpeechDetected:
            self.statusChanged.emit("Não detetei fala.")
            self.transcriptionReady.emit(capture.finish(""))
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
        *,
        require_interruptible: bool = False,
    ) -> None:
        super().__init__()
        self.tts_manager = tts_manager
        self.text = text
        self.player = player
        self.response_language = response_language
        # Phase 7.6.4 (section 7): Voice Conversation Mode must never
        # silently degrade to the non-interruptible winsound fallback and
        # pretend STOP/barge-in still work. When True, play_audio_file()
        # raises instead of falling back to winsound -- the existing
        # except/failed.emit()/_handle_tts_error path below then keeps the
        # text response visible and safely recovers voice mode, exactly as
        # it already does for any other TTS failure.
        self.require_interruptible = require_interruptible
        self.stop_event = threading.Event()
        # Phase 7.6.3 (section 14): WHY this worker was stopped (if it was)
        # -- set by whoever calls stop(reason=...); read back in run() once
        # self.player() returns, to log a [TTS_PLAYBACK] line that
        # correlates with the caller's own [VOICE_STOP]/[BARGE_IN] context.
        self._stop_reason = ""
        self._stop_requested_at: float | None = None

    def stop(self, *, teardown_manager: bool = False, reason: str = "manual_stop") -> None:
        # Phase 7.3.4 (section 5): stopping THIS request/playback must never
        # tear down the shared, persistent Pocket worker -- it used to call
        # tts_manager.stop() unconditionally, which cascades through
        # LocalTTSProvider.stop() into every provider's .stop(), including
        # PocketTTSLocalProvider.shutdown() (kills the long-lived subprocess
        # and nulls self._process). Since stopSpeaking() is wired to fire
        # automatically before every new user message and on "clear
        # conversation" while TTS is playing (see echo_ui.js), that
        # unconditional teardown silently forced a full cold Pocket restart
        # on the very next SPEAK -- an ordinary mid-session UI action, not a
        # worker failure or app exit. teardown_manager=True is reserved for
        # genuine session-ending shutdown (see EchoUIController.shutdown()).
        self._stop_reason = reason
        self._stop_requested_at = time.perf_counter()
        self.stop_event.set()
        if not teardown_manager:
            return
        stop = getattr(self.tts_manager, "stop", None)
        if callable(stop):
            stop()

    def _on_playback_event(self, event: str, fields: dict[str, object]) -> None:
        # Phase 7.6.3 (section 14): the ONE place [TTS_PLAYBACK] lines are
        # emitted -- play_audio_file() reports backend/timing, this adds
        # the WHY (reason/stop_latency_ms) that only the caller knows.
        payload = dict(fields)
        if event == "stopped":
            if self._stop_requested_at is not None:
                payload["stop_latency_ms"] = int((time.perf_counter() - self._stop_requested_at) * 1000)
            payload["reason"] = self._stop_reason or "manual_stop"
        elif event == "completed":
            payload["reason"] = "natural_completion"
        _log_tts_playback(event, **payload)

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
                        "provider_voice": result.provider_voice,
                        "language": self.response_language or result.language,
                        "response_language": self.response_language,
                        "voice_language": result.language,
                        "model": result.model,
                        "device": result.device,
                        "worker_launch_ms": int(result.worker_launch_ms),
                        "model_load_ms": int(result.model_load_ms),
                        "reference_prepare_ms": int(result.reference_prepare_ms),
                        "first_audio_ms": int(result.first_audio_ms),
                        "synthesis_ms": int(result.synthesis_ms),
                        "audio_duration_ms": int(result.audio_duration_ms),
                        "text_chars": result.text_chars,
                        "reference_used": result.reference_used,
                        "reference_cache_hit": result.reference_cache_hit,
                        "fallback_used": result.fallback_used,
                        "fallback_reason": result.fallback_reason,
                    }
                )
            )
            try:
                playback_ms = self.player(
                    result.audio_path,
                    self.stop_event.is_set,
                    on_playback_event=self._on_playback_event,
                    allow_winsound_fallback=not self.require_interruptible,
                )
            except TypeError:
                # A custom/test player double that doesn't know about the
                # new diagnostics hook -- fall back to the original 2-arg
                # call so no existing test fake needs updating.
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
    _sttStatusChanged = Signal(str)
    ttsStarted = Signal(str)
    ttsFinished = Signal(str)
    ttsError = Signal(str)
    # Phase 7.4: the ONE frontend-facing signal for voice-conversation-mode
    # state (section 3) -- JS never tracks its own independent copy of
    # "is voice mode on", it only ever renders the value this carries (plus
    # whatever isVoiceConversationActive() reports on load/reconnect).
    voiceConversationChanged = Signal(bool)
    # Phase 7.6 (section 2/16): the frontend-facing signal for Voice Standby
    # Mode -- mirrors voiceConversationChanged's role for ACTIVE mode. JS
    # renders a tri-state label from these two booleans combined; the
    # backend guarantees they are never both True at once.
    voiceStandbyChanged = Signal(bool)

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
        # Phase 7.6.6.1 (ownership rule): the controller holds this
        # reference for the thread's ENTIRE valid lifetime and NOWHERE
        # ELSE -- set the instant a prewarm thread is created, cleared the
        # instant it finishes (_on_wake_prewarm_finished), before deleteLater
        # is even requested. Anything that reads this attribute (shutdown()
        # included) can therefore trust: not None => still a valid, live
        # QThread wrapper; None => nothing to wait on, nothing to guard.
        self._wake_prewarm_thread: QThread | None = None
        # Phase 7.6.6.1 (shutdown requirement 1): flips True the instant
        # shutdown() begins tearing down the prewarm thread -- prevents a
        # NEW one from being started afterward (e.g. a stray rearm/standby
        # re-entry racing app close).
        self._wake_prewarm_shutting_down = False
        # Phase 7.6.7 (section 8/9): shutdown bookkeeping. request_shutdown()
        # only SIGNALS every owned worker to stop (never waits), and
        # shutdown_async() then lets the Qt event loop keep running while
        # they drain -- so app close never parks the GUI thread in a
        # synchronous wait.
        self._shutdown_requested = False
        self._shutdown_completed = False
        self._shutdown_started_at: float | None = None
        self._shutdown_on_complete: Callable[[], None] | None = None
        self._shutdown_session_close: Callable[[], None] | None = None
        self._shutdown_pending_seen: set[str] = set()
        self._shutdown_deadline_ms = self._SHUTDOWN_DRAIN_DEADLINE_MS
        self._tts_manager_shut_down = False
        self.tts_manager = tts_manager
        self.tts_player = tts_player
        self.get_response_language_callback = get_response_language
        self.voice_available = bool(voice_available)
        self.voice_transcriber = voice_transcriber
        self._sttStatusChanged.connect(self._handle_stt_status, Qt.ConnectionType.QueuedConnection)
        if callable(getattr(voice_transcriber, "set_stt_status_callback", None)):
            voice_transcriber.set_stt_status_callback(self._sttStatusChanged.emit)
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
        # Phase 7.4 (section 3/13): the SINGLE source of truth for voice-
        # conversation-mode state, plus a generation token that increments
        # every start/stop. Any async callback (STT transcript, LLM
        # response, TTS finished/error, a rearm timer) that wants to act on
        # behalf of the loop must check BOTH `_voice_conversation_active`
        # and that its own captured session id still equals
        # `_voice_session_id` before auto-submitting, auto-speaking, or
        # restarting listening -- this is what makes stopping (or a
        # stop+restart) safe from any state without resurrecting stale work.
        self._voice_conversation_active = False
        self._voice_session_id = 0
        # The voice_session_id captured at the moment THIS turn started, if
        # it started while conversation mode was active -- None for a plain
        # typed turn submitted while the mode is off. _handle_response only
        # auto-speaks when this still equals _voice_session_id.
        self._voice_turn_session_id: int | None = None
        self._voice_conversation_failures = 0
        # Phase 7.4.1 (section 3): the voice_session_id captured at the
        # moment THIS worker was created via _begin_listening(), if
        # conversation mode was active then -- None for a manual (non-
        # conversation) recording, which must never be treated as stale.
        # _handle_voice_transcription/_handle_voice_error check this against
        # the CURRENT _voice_session_id before touching the response area,
        # state, or the loop -- a worker started under an old/stopped
        # session that reports late (e.g. "recording too short" arriving
        # after the user already turned the mode off) is silently dropped
        # instead of surfacing as the main Echo response.
        self._voice_worker_session_id: int | None = None
        # Phase 7.6 (section 2): the mode ("active"/"standby") captured
        # alongside _voice_worker_session_id when THIS worker was created --
        # None for manual. _is_voice_worker_stale() now checks that the
        # CURRENT mode still matches this, not just the session id, so a
        # standby worker's late result is never misread as an active one
        # (or vice versa) across a wake/sleep transition.
        self._voice_worker_mode: str | None = None
        # Phase 7.6 (section 2): STANDBY is a new, separate concept from
        # ACTIVE (_voice_conversation_active) -- never both True at once.
        # OFF is simply neither. See _current_voice_mode().
        self._voice_standby_active = False
        # Phase 7.6 (section 8-12): true for the lifetime of a worker
        # created by _begin_barge_in_monitor() (mic monitoring while Anna
        # speaks), and _voice_worker_barge_in_confirmed flips True the
        # instant it actually detects genuine interruption -- both reset on
        # every new worker. Together they let _handle_tts_finished/_error
        # tell "detector never confirmed anything, safe to cancel it" apart
        # from "detector confirmed and is now capturing the interruption,
        # must not be touched or duplicated".
        self._voice_worker_is_barge_in = False
        self._voice_worker_barge_in_confirmed = False
        # Diagnostics only, never read by the voice loop itself:
        # [VOICE_TURN_TIMING] perf_counter() marks for the voice turn in
        # flight (None when there is none), and the two [VOICE_LOST_SPEECH]
        # flags -- an unconfirmed barge-in monitor cancelled because TTS
        # ended, and "the next capture follows a confirmed barge-in".
        self._voice_turn_timing: dict[str, float] | None = None
        self._barge_in_cancelled_at_tts_end = False
        self._barge_in_recent = False
        # Phase 7.6.2 (section 3): set when stopSpeaking() interrupts an
        # actual SPEAKING TTS worker -- consumed (and logged) the next time
        # _begin_listening() actually reopens the mic, giving the requested
        # [VOICE_STOP] event=returned_to_listening a real completion point
        # rather than firing immediately/optimistically at request time.
        self._voice_stop_awaiting_listen_log = False
        # Pocket TTS cold start: the server used to start on the first speak
        # request (~5 s), delaying the first wake ack. Same ownership rule as
        # _wake_prewarm_thread: set when started, cleared when finished.
        self._tts_prewarm_thread: QThread | None = None
        self._tts_prewarm_started_at: float | None = None
        self._ui_ready_received = False
        self._tts_startup_prewarm_timer = QTimer(self)
        self._tts_startup_prewarm_timer.setSingleShot(True)
        self._tts_startup_prewarm_timer.timeout.connect(self._prewarm_tts)
        # R2 (assistant/model_warmup.py): Ollama is an external process that
        # does not compete with the UI, so it warms right away (also for
        # typed chats); the GPU STT warms on the same timer as Pocket.
        engine = getattr(self.responder, "__self__", None)
        self._model_warmup = ModelWarmup(transcriber=self.voice_transcriber, llm=getattr(engine, "llm", None))
        self._tts_startup_prewarm_timer.timeout.connect(self._prewarm_models)
        # Started directly (a plain background thread, no Qt timer that could
        # fire after this controller's C++ object is gone).
        self._model_warmup.start_ollama()

    _TTS_STARTUP_PREWARM_DELAY_MS = 3000

    @Slot()
    def _prewarm_models(self) -> None:
        # Background, each model once, never after shutdown began; failures
        # are logged by ModelWarmup and the first turn pays the cold start.
        if self._shutdown_requested:
            return
        self._model_warmup.start_ollama()
        self._model_warmup.start_stt()

    def _release_local_models(self) -> None:
        # App close, never waiting: stop a warm-up in flight and ask Ollama
        # to unload the model the chats keep loaded (keep_alive=-1).
        if self._model_warmup is not None:
            self._model_warmup.cancel()
        engine = getattr(self.responder, "__self__", None)
        unload_ollama_in_background(find_ollama_provider(getattr(engine, "llm", None)))

    def notify_ui_ready(self) -> None:
        if self._ui_ready_received or self._shutdown_requested:
            return
        self._ui_ready_received = True
        if self.voice_available:
            self._tts_startup_prewarm_timer.start(self._TTS_STARTUP_PREWARM_DELAY_MS)

    @Slot(str)
    def submitMessage(self, text: str) -> None:
        self._submit_turn(TurnInput.typed(text))

    def _submit_turn(self, turn: TurnInput) -> None:
        text = turn.text
        message = (text or "").strip()
        if message != turn.text:
            turn = replace(turn, text=message)
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
        if self._voice_worker is not None:
            # Phase 7.4 (section 15): never let mic capture and turn
            # processing overlap. Covers both a typed message arriving
            # while a voice-conversation listening cycle is still open,
            # and the (harmless) re-entrant case where the STT worker's own
            # completion is what called submitMessage() in the first place.
            self.cancelVoiceInput()

        self._active = True
        self._started_at = time.perf_counter()
        self._visual_token += 1
        self._last_progress_event = ""
        # Phase 7.4 (section 13): capture the CURRENT voice session only
        # when the turn actually starts while conversation mode is active --
        # this is what _handle_response checks before auto-speaking, so a
        # plain typed turn with the mode off never auto-speaks (unchanged
        # existing behavior), and a turn started under an OLD session can
        # never auto-speak under a newer one.
        self._voice_turn_session_id = self._voice_session_id if self._voice_conversation_active else None
        if self._voice_turn_session_id is not None:
            self._log_voice("turn_submitted")
        self.requestStarted.emit(message)
        self._emit_state("thinking")
        self._emit_runtime_telemetry("request_started")
        _debug_ui("[Echo UI] request_started=true")

        self._thread = QThread(self)
        self._worker = EchoRequestWorker(self.responder, message, turn_input=turn)
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
        if self._voice_conversation_active or self._voice_standby_active:
            # Phase 7.6: manual MIC and the conversation loop (active or
            # standby) must never race for the same mic -- the loop already
            # owns it.
            self.voiceInputError.emit("O modo de conversa por voz já está ativo.")
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
        self._begin_listening()

    def _voice_dependencies_missing(self) -> list[str]:
        if not self.voice_available or self.voice_transcriber is None:
            return ["voice_unavailable"]
        return list(self.voice_missing_dependencies if self._voice_runtime_status_provided else check_voice_runtime())

    def _current_voice_mode(self) -> str:
        """Phase 7.6 (section 2): the ONE tri-state read of voice-loop
        state -- "active"/"standby"/"off". _voice_conversation_active and
        _voice_standby_active remain the actual stored booleans (kept
        mutually exclusive at every transition site) so every existing
        direct reference to _voice_conversation_active keeps working
        unchanged; this is just the combined view."""
        if self._voice_conversation_active:
            return "active"
        if self._voice_standby_active:
            return "standby"
        return "off"

    def _begin_listening(self) -> None:
        """The actual "create and start an EchoVoiceWorker" mechanics --
        shared by the manual startVoiceInput() Slot (which has its own
        user-facing availability/busy checks above) and the Phase 7.4/7.6
        conversation-loop rearm path (_rearm_voice_listening), which must
        never surface a manual-mic-flavored error message for an internal
        automatic retry. Which of active/standby/manual applies is read
        fresh from _current_voice_mode() every time -- this is what makes
        the SAME rearm path correctly resume either loop."""
        if self._shutdown_requested:
            # Phase 7.6.7: nothing may open the mic once app close began --
            # a late rearm timer/TTS-finished handler during the drain
            # would otherwise start a fresh worker the drain then waits on.
            return
        self._visual_token += 1
        if self._voice_stop_awaiting_listen_log:
            self._voice_stop_awaiting_listen_log = False
            self._log_voice_stop("returned_to_listening")
        current_mode = self._current_voice_mode()
        self._voice_worker_session_id = self._voice_session_id if current_mode != "off" else None
        self._voice_worker_mode = current_mode if current_mode != "off" else None
        self._voice_worker_is_barge_in = False
        self._voice_worker_barge_in_confirmed = False
        worker_mode = current_mode if current_mode != "off" else "manual"
        self._voice_thread = QThread(self)
        self._voice_worker = EchoVoiceWorker(self.voice_transcriber, mode=worker_mode, session_id=self._voice_session_id)
        self._voice_worker.moveToThread(self._voice_thread)
        self._voice_thread.started.connect(self._voice_worker.run, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.statusChanged.connect(self._handle_voice_status, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.transcriptionReady.connect(self._handle_voice_transcription, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.failed.connect(self._handle_voice_error, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.transcriptionReady.connect(self._cleanup_voice_worker, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.failed.connect(self._cleanup_voice_worker, Qt.ConnectionType.QueuedConnection)
        self.voiceInputStarted.emit()
        self._emit_state("listening")
        if current_mode != "off":
            self._log_voice("listening_started")
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

    # --- Phase 7.4: explicit Voice Conversation Mode -------------------
    # Section 1-3: one explicit, backend-owned mode -- LISTENING ->
    # TRANSCRIBING -> THINKING -> SYNTHESIZING -> SPEAKING -> LISTENING,
    # reusing the existing "listening"/"thinking"/"working"/"speaking"
    # _emit_state values throughout rather than inventing new ones (section
    # 19). Never builds a separate voice-only answer path (section 4):
    # auto-submit reuses submitMessage(), auto-speak reuses speakText().

    @Slot(result=bool)
    def startVoiceConversation(self) -> bool:
        self._thread_debug("startVoiceConversation")
        if self._voice_conversation_active:
            return True
        missing = self._voice_dependencies_missing()
        if missing:
            self.voiceInputError.emit(
                "A voz não está disponível nesta sessão."
                if missing == ["voice_unavailable"]
                else "A voz não está pronta porque falta: " + ", ".join(missing)
            )
            return False
        if self._active or self._tts_worker is not None or self._voice_worker is not None:
            self.voiceInputError.emit("Espera que a resposta atual termine antes de ativares o modo de conversa por voz.")
            return False
        self._voice_standby_active = False
        self._voice_conversation_active = True
        self._voice_session_id += 1
        self._voice_conversation_failures = 0
        self._log_voice("started")
        self._prewarm_tts()
        self._prewarm_models()
        self.voiceConversationChanged.emit(True)
        self._begin_listening()
        return True

    @Slot()
    def stopVoiceConversation(self) -> None:
        self._thread_debug("stopVoiceConversation")
        self._stop_voice_conversation(reason="user_request")

    @Slot(result=bool)
    def isVoiceConversationActive(self) -> bool:
        return bool(self._voice_conversation_active)

    # --- Phase 7.6: Voice Standby Mode (wake/sleep) ---------------------
    # Section 2/15: a genuinely separate concept from ACTIVE -- the mic is
    # open and VAD-listening, but a captured utterance is interpreted as a
    # wake/ignore decision (see _handle_standby_transcript), never routed
    # into the normal task/response pipeline. The manual UI toggle enters
    # STANDBY (not ACTIVE directly); saying a wake phrase from standby is
    # what promotes the loop into ACTIVE (_wake_into_active_conversation).
    # startVoiceConversation()/stopVoiceConversation() above are left
    # completely unchanged in meaning (direct-to-ACTIVE) so every existing
    # Phase 7.4/7.5 test keeps working untouched.

    @Slot(result=bool)
    def startVoiceStandby(self) -> bool:
        self._thread_debug("startVoiceStandby")
        if self._voice_standby_active or self._voice_conversation_active:
            return True
        missing = self._voice_dependencies_missing()
        if missing:
            self.voiceInputError.emit(
                "A voz não está disponível nesta sessão."
                if missing == ["voice_unavailable"]
                else "A voz não está pronta porque falta: " + ", ".join(missing)
            )
            return False
        if self._active or self._tts_worker is not None or self._voice_worker is not None:
            self.voiceInputError.emit("Espera que a resposta atual termine antes de ativares a voz.")
            return False
        self._voice_session_id += 1
        self._voice_conversation_failures = 0
        self._begin_voice_standby(reason="user_request")
        return True

    @Slot()
    def stopVoiceStandby(self) -> None:
        self._thread_debug("stopVoiceStandby")
        self._stop_voice_standby(reason="user_request")

    @Slot(result=bool)
    def isVoiceStandbyActive(self) -> bool:
        return bool(self._voice_standby_active)

    @Slot(result=str)
    def voiceMode(self) -> str:
        return self._current_voice_mode()

    def _begin_voice_standby(self, *, reason: str) -> None:
        self._voice_standby_active = True
        self._log_wake("standby_entered", reason=reason)
        self.voiceStandbyChanged.emit(True)
        self._prewarm_wake_recognizer()
        self._prewarm_tts()
        self._prewarm_models()
        if self._active or self._tts_worker is not None:
            # Something else is genuinely still busy (a typed turn, TTS
            # being stopped as part of this same sleep transition) -- ITS
            # OWN completion handler already checks _current_voice_mode()
            # and will begin listening once it settles.
            return
        if self._voice_worker is not None:
            # This is commonly the SAME worker whose own transcript (e.g.
            # the "go to sleep" phrase itself) triggered this very call --
            # its cleanup slot, connected to the same signal, hasn't run
            # yet, so starting a new worker over it here would race/corrupt
            # that cleanup. Schedule a rearm instead; by the time the timer
            # fires, cleanup will have completed.
            self._schedule_voice_rearm()
            return
        self._begin_listening()

    def _prewarm_wake_recognizer(self) -> None:
        # Phase 7.6.6 (section 6): the first-ever wake attempt after app
        # startup pays the fast recognizer's model-load cost inline unless
        # something loads it earlier -- kicked off here, on a plain daemon
        # thread (NOT blocking this Slot / the UI thread), the instant
        # STANDBY is entered, so by the time the user actually says "Echo"
        # the model is very likely already loaded. WakeRecognizer.prewarm()
        # is itself lock-guarded (assistant/wake_recognizer.py), so a real
        # wake attempt racing this thread just reuses whichever one
        # finishes loading first -- never a second, redundant load. Duck-
        # typed: a transcriber double without a wake_recognizer (every
        # existing test fake) is simply skipped, never touching Pocket TTS
        # or any other model (section 6: "do NOT prewarm... unrelated
        # models"). Uses a QThread (like every other background operation
        # in this controller -- EchoRequestWorker/EchoVoiceWorker/
        # EchoTTSWorker all run on QThread, never a raw threading.Thread)
        # rather than Python's own threading.Thread: mixing raw threads
        # into a process this heavily built around Qt's own thread/event
        # loop machinery caused real instability under a long real-Qt test
        # run, whereas a QThread behaves like everything else here.
        #
        # Phase 7.6.6.1: never start a second prewarm while one is already
        # in flight -- re-entering STANDBY quickly (sleep -> wake -> sleep)
        # must not spawn a pile of redundant threads; the existing one
        # already warms the SAME shared WakeRecognizer. Also never start a
        # new one once shutdown has begun tearing this down (requirement 1).
        if self._wake_prewarm_thread is not None or self._wake_prewarm_shutting_down:
            return
        recognizer = getattr(self.voice_transcriber, "wake_recognizer", None)
        prewarm = getattr(recognizer, "prewarm", None)
        if not callable(prewarm):
            return
        thread = _WakeRecognizerPrewarmThread(prewarm, self)
        self._wake_prewarm_thread = thread
        thread.finished.connect(self._on_wake_prewarm_finished)
        self._log_wake_prewarm("started")
        thread.start()

    @Slot()
    def _on_wake_prewarm_finished(self) -> None:
        # Phase 8.1.1: MUST stay a declared @Slot. Connecting a signal to an
        # undecorated method makes PySide add a dynamic slot to this
        # controller's QMetaObject at runtime -- and this is the object
        # QWebChannel already published to JS with a fixed method table.
        # Real symptom: every controller -> JS signal (stateChanged,
        # responseReady, voiceConversationChanged...) silently stopped
        # arriving the moment STANDBY started the prewarm thread, so voice
        # responses were spoken by TTS but never rendered.
        #
        # Phase 7.6.6.1: the ownership fix -- clear the controller's OWN
        # reference to this thread FIRST, before ever requesting its C++
        # deletion, so nothing (shutdown() included) can observe a
        # not-None-but-actually-deleted wrapper. sender() is the thread
        # itself (this slot is only ever connected to one thread's own
        # finished signal); guarding with identity (`is thread`) means a
        # stale/duplicate signal delivery can never clear a NEWER thread's
        # reference out from under it.
        thread = self.sender()
        if self._wake_prewarm_thread is thread:
            self._wake_prewarm_thread = None
        self._log_wake_prewarm("finished")
        if thread is not None:
            thread.deleteLater()

    @Slot()
    def _prewarm_tts(self) -> None:
        self._tts_startup_prewarm_timer.stop()
        # Starts the TTS manager's long-lived worker (Pocket server) off the
        # UI thread so the first spoken reply / wake ack does not wait for
        # its start-up. Duck-typed (a TTS double without prewarm() is
        # skipped); one at a time; never once shutdown has begun. When the
        # server is already running prewarm() returns at once. A real speak
        # racing this simply waits on the provider's own lock for the same
        # server, never a second one.
        if self._tts_prewarm_thread is not None or self._wake_prewarm_shutting_down or self._shutdown_requested:
            return
        prewarm = getattr(self.tts_manager, "prewarm", None)
        if not callable(prewarm):
            return
        thread = _WakeRecognizerPrewarmThread(prewarm, self)
        self._tts_prewarm_thread = thread
        self._tts_prewarm_started_at = time.perf_counter()
        thread.finished.connect(self._on_tts_prewarm_finished)
        print("[TTS_PREWARM] event=started")
        thread.start()

    @Slot()
    def _on_tts_prewarm_finished(self) -> None:
        # Must stay a declared @Slot (see _on_wake_prewarm_finished).
        thread = self.sender()
        if self._tts_prewarm_thread is thread:
            self._tts_prewarm_thread = None
        elapsed_ms = int((time.perf_counter() - self._tts_prewarm_started_at) * 1000) if self._tts_prewarm_started_at else 0
        print(f"[TTS_PREWARM] event=finished elapsed_ms={elapsed_ms}")
        if thread is not None:
            thread.deleteLater()

    def _stop_voice_standby(self, *, reason: str) -> None:
        if not self._voice_standby_active:
            return
        self._voice_standby_active = False
        self._voice_session_id += 1
        self._voice_conversation_failures = 0
        self._voice_stop_awaiting_listen_log = False
        self._log_wake("standby_exited", reason=reason)
        if self._voice_worker is not None:
            self.cancelVoiceInput()
        self.voiceStandbyChanged.emit(False)

    def _handle_standby_transcript(self, clean: str) -> None:
        if not clean:
            # Phase 7.6.7 (section 4): the fast recognizer's empty/unrelated
            # result now comes back as "" (no Medium pass) -- still log its
            # timing so a real session shows the quick-rearm latency.
            outcome = getattr(self.voice_transcriber, "last_vad_outcome", None) or {}
            fast_assessment = outcome.get("fast_assessment")
            if fast_assessment in ("empty", "unrelated"):
                self._log_wake_timing(classification_ms=0.0, wake_to_active_ms=0.0)
                self._log_wake("wake_ignored", reason=f"fast_{fast_assessment}")
            self._schedule_voice_rearm()
            return
        self._log_wake("wake_speech_detected")
        classify_started_at = time.perf_counter()
        intent = classify_voice_wake_intent(clean)
        classification_ms = (time.perf_counter() - classify_started_at) * 1000.0
        if intent == "wake":
            self._log_wake("wake_intent_matched", language=self._response_language() or "unknown")
            self._wake_into_active_conversation(classification_ms=classification_ms)
            return
        self._log_wake_timing(classification_ms=classification_ms, wake_to_active_ms=0.0)
        self._log_wake("wake_ignored")
        self._schedule_voice_rearm()

    def _wake_into_active_conversation(self, *, classification_ms: float = 0.0) -> None:
        # NOTE: deliberately does NOT check `self._voice_worker is not None`
        # -- this is always called synchronously from within that SAME
        # worker's own transcriptionReady handling (_handle_voice_transcription
        # -> _handle_standby_transcript), before its cleanup slot (connected
        # to the same signal) has run. Checking it here would always see the
        # about-to-be-cleaned-up worker as "something else busy" and bail
        # out forever, looping standby -> wake -> bail -> rearm indefinitely.
        wake_transition_started_at = time.perf_counter()
        if self._active or self._tts_worker is not None:
            self._schedule_voice_rearm()
            return
        self._voice_standby_active = False
        self.voiceStandbyChanged.emit(False)
        self._voice_conversation_active = True
        self._voice_session_id += 1
        self._voice_conversation_failures = 0
        self._log_voice("started", reason="wake_command")
        # Section 12: ACTIVE becomes true immediately -- voiceConversationChanged
        # fires (and the frontend's own "VOICE MODE · ON" visual flips) right
        # here, BEFORE any acknowledgement TTS, not after it.
        self.voiceConversationChanged.emit(True)
        wake_to_active_ms = (time.perf_counter() - wake_transition_started_at) * 1000.0
        self._log_wake_timing(classification_ms=classification_ms, wake_to_active_ms=wake_to_active_ms)
        # Section 6: speak a short deterministic acknowledgement rather than
        # silently opening the mic -- gives the user (no screen focus in a
        # voice-first flow) clear confirmation the wake worked. Self-
        # capture is avoided for free: speakText()'s existing lifecycle
        # already guarantees the mic stays closed until _handle_tts_finished
        # fires, which is what actually opens listening afterward.
        self._speak_deterministic(wake_acknowledgement_text(self._response_language()))

    def _enter_voice_standby(self, *, reason: str) -> None:
        self._stop_voice_conversation(reason=reason, enter_standby=True)

    def _speak_deterministic(self, text: str) -> None:
        """A canned, non-LLM acknowledgement (e.g. the wake greeting) --
        shown as the visible response AND spoken via the existing TTS
        lifecycle, but never routed through submitMessage()/the request
        pipeline (no task/routing/research runs for it)."""
        self._emit_response_ready(text, source="deterministic")
        self.speakText(text)

    def _stop_voice_conversation(self, *, reason: str, enter_standby: bool = False) -> None:
        if not self._voice_conversation_active:
            return
        self._voice_conversation_active = False
        self._voice_session_id += 1
        self._voice_conversation_failures = 0
        if not enter_standby:
            self._voice_stop_awaiting_listen_log = False
        self._log_voice("stopped", reason=reason)
        # Section 12: stop safely from ANY state -- LISTENING/TRANSCRIBING
        # (cancel the STT worker), SYNTHESIZING/SPEAKING (stop TTS
        # playback). THINKING has no worker of its own to stop here: the
        # normal request is left to resolve on its own (never cancelled),
        # but _handle_response's session-id check (now stale, since
        # _voice_conversation_active just flipped False) already ensures it
        # will not auto-speak or rearm listening once it arrives.
        if self._voice_worker is not None:
            self.cancelVoiceInput()
        if self._tts_worker is not None:
            self.stopSpeaking()
        self.voiceConversationChanged.emit(False)
        if enter_standby:
            # Section 7: "go to sleep" leaves ACTIVE for STANDBY, not fully
            # OFF -- a spoken wake phrase can bring it back.
            self._begin_voice_standby(reason=reason)

    def _schedule_voice_rearm(self, delay_ms: int | None = None) -> None:
        session_id = self._voice_session_id
        if delay_ms is not None:
            delay = delay_ms
        elif self._current_voice_mode() == "standby":
            delay = self._VOICE_STANDBY_REARM_DELAY_MS
        else:
            delay = self._VOICE_REARM_DELAY_MS
        self._log_voice("rearm_listening", delay_ms=delay)
        QTimer.singleShot(delay, lambda: self._rearm_voice_listening(session_id))

    def _rearm_voice_listening(self, session_id: int) -> None:
        if self._current_voice_mode() == "off" or session_id != self._voice_session_id:
            self._log_voice("stale_callback_ignored", context="rearm")
            return
        if self._active or self._voice_worker is not None or self._tts_worker is not None:
            # Something else (a typed message, another rearm already in
            # flight) claimed the turn in the meantime -- never start a
            # second, competing listening cycle (section 24.P).
            return
        if self._voice_dependencies_missing():
            # Voice became unavailable mid-session (e.g. runtime dropped
            # out) -- stop the mode instead of silently spinning forever.
            if self._voice_conversation_active:
                self._stop_voice_conversation(reason="voice_unavailable")
            else:
                self._stop_voice_standby(reason="voice_unavailable")
            return
        self._begin_listening()

    def _auto_speak_response(self, response_text: str, session_id: int) -> None:
        if not self._voice_conversation_active or session_id != self._voice_session_id:
            self._log_voice("stale_callback_ignored", context="auto_speak")
            self._log_voice_turn_timing("stale")
            return
        self._log_voice("tts_requested")
        if self._voice_turn_timing is not None:
            self._voice_turn_timing["tts_requested_at"] = time.perf_counter()
        self.speakText(response_text)

    def _is_voice_worker_stale(self) -> bool:
        """Phase 7.4.1/7.6 (section 3): true when the worker CURRENTLY
        reporting (transcription/error) was started under a voice mode
        (active OR standby) and session that has since been stopped or
        superseded -- e.g. stopVoiceConversation()/a sleep command ran
        while it was still recording, so its eventual "too short"/"no
        speech"/genuine-error result arrives after the fact, OR the mode
        itself changed (standby woke into active, active slept into
        standby) since this worker started. Always False for a manual
        (non-conversation) recording, whose captured session id is None --
        this must never change manual MIC behavior."""
        if self._voice_worker_session_id is None:
            return False
        return self._current_voice_mode() != self._voice_worker_mode or self._voice_worker_session_id != self._voice_session_id

    def _settle_idle_if_nothing_pending(self) -> None:
        """Phase 7.4.1 (section 3/8): cancelVoiceInput() already flashes the
        UI to "thinking" the instant a voice worker is cancelled (see
        cancelVoiceInput() above), before it's known whether that worker's
        eventual result will turn out to be stale. A suppressed stale
        callback must not restart the loop or touch the response, but it
        must also not leave the UI stuck showing "thinking" forever --
        settle back to idle once nothing else (a typed turn, TTS) has
        claimed it in the meantime.

        Phase 7.6: a stale worker can also arrive after a MODE change
        rather than a full stop -- e.g. a sleep command bailed out of
        _begin_voice_standby() because the just-cancelled active worker
        hadn't finished tearing down yet. In that case "settle" means
        schedule a rearm (not go idle): _voice_worker is still this SAME
        stale worker's own reference at this point (its cleanup slot,
        connected to the same signal, hasn't run yet), so starting a new
        listening cycle synchronously here would race/corrupt that
        cleanup -- _schedule_voice_rearm's timer naturally waits it out."""
        if self._active or self._tts_worker is not None:
            return
        if self._current_voice_mode() == "off":
            self._emit_state("idle")
            return
        self._schedule_voice_rearm()

    def _log_voice(self, event: str, **fields: object) -> None:
        # Section 23: bounded diagnostics -- never logs transcript/response
        # content, only counts/session ids/reasons.
        parts = [f"[VOICE_LOOP] event={event}", f"session_id={self._voice_session_id}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts))

    def _log_wake(self, event: str, **fields: object) -> None:
        # Phase 7.6 (section 18): bounded diagnostics for wake/sleep --
        # never logs transcript content, only counts/session ids/reasons.
        parts = [f"[VOICE_WAKE] event={event}", f"session_id={self._voice_session_id}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts))

    def _log_wake_prewarm(self, event: str) -> None:
        # Phase 7.6.6.1: bounded, one line per lifecycle event -- lets a
        # real shutdown log prove exactly which of started/finished/
        # shutdown_wait/shutdown_already_finished/reference_cleared
        # actually happened, instead of guessing at the QThread's state.
        print(f"[VOICE_WAKE_PREWARM] event={event}")

    def _log_wake_timing(self, *, classification_ms: float, wake_to_active_ms: float) -> None:
        # Phase 7.6.5 (section 8): bounded breakdown of where wake latency
        # actually goes -- speech_start_to_end_ms/trailing_silence_ms/
        # stt_ms come straight from the transcriber's last_vad_outcome
        # (populated by record_until_speech_complete / record_and_
        # transcribe_vad / record_and_transcribe_wake), classification_ms/
        # wake_to_active_ms are measured here around the deterministic
        # (non-LLM) classify + ACTIVE-flip steps. "Measure first, don't
        # guess" (section 8).
        #
        # Phase 7.6.6 (section 13): extended with recognizer=/model=/
        # model_load_ms=/fallback_to_medium= so a real log makes it
        # obvious which recognizer actually produced the transcript --
        # defaults (recognizer="medium", model=self.voice_transcriber's
        # own Medium model_name, model_load_ms=0, fallback_to_medium=False)
        # cover a transcriber double that only implements the older
        # record_and_transcribe_vad path (no wake_recognizer fields on its
        # last_vad_outcome at all).
        outcome = getattr(self.voice_transcriber, "last_vad_outcome", None) or {}
        speech_ms = outcome.get("speech_ms", 0.0) or 0.0
        trailing_silence_ms = outcome.get("trailing_silence_ms", 0.0) or 0.0
        stt_ms = outcome.get("stt_ms", 0.0) or 0.0
        recognizer = outcome.get("recognizer") or "medium"
        model = outcome.get("wake_model") or getattr(self.voice_transcriber, "model_name", "unknown")
        model_load_ms = outcome.get("wake_model_load_ms", 0.0) or 0.0
        fallback_to_medium = bool(outcome.get("fallback_to_medium", False))
        # Phase 7.6.7: why the fast stage did/didn't hand off to Medium, and
        # whether that Medium pass paid a cold model load (the real 9184ms
        # stt_ms was Medium's FIRST use of the session).
        fast_assessment = outcome.get("fast_assessment") or "n/a"
        medium_cold_load = outcome.get("medium_cold_load")
        total_ms = speech_ms + trailing_silence_ms + stt_ms + classification_ms + wake_to_active_ms
        print(
            "[VOICE_WAKE_TIMING] "
            f"recognizer={recognizer} "
            f"model={model} "
            f"model_load_ms={int(model_load_ms)} "
            f"speech_start_to_end_ms={int(speech_ms)} "
            f"trailing_silence_ms={int(trailing_silence_ms)} "
            f"end_of_speech_to_stt_start_ms={int(trailing_silence_ms)} "
            f"stt_ms={int(stt_ms)} "
            f"fallback_to_medium={str(fallback_to_medium).lower()} "
            f"fast_assessment={fast_assessment} "
            f"medium_cold_load={'n/a' if medium_cold_load is None else str(bool(medium_cold_load)).lower()} "
            f"classification_ms={classification_ms:.2f} "
            f"wake_to_active_ms={wake_to_active_ms:.2f} "
            f"total_ms={int(total_ms)}"
        )

    def _start_voice_turn_timing(self) -> None:
        # Diagnostics only: copies the capture/STT marks of the transcript
        # about to be submitted; the rest is filled in as the turn advances
        # (_handle_response -> _auto_speak_response -> _handle_tts_started).
        outcome = getattr(self.voice_transcriber, "last_vad_outcome", None) or {}
        timing: dict[str, float] = {"transcript_at": time.perf_counter()}
        capture_ended_at = outcome.get("capture_ended_at")
        if isinstance(capture_ended_at, (int, float)):
            timing["capture_ended_at"] = float(capture_ended_at)
            timing["speech_end_at"] = float(capture_ended_at) - float(outcome.get("trailing_silence_ms") or 0.0) / 1000.0
        stt_started_at = outcome.get("stt_started_at")
        if isinstance(stt_started_at, (int, float)):
            timing["stt_started_at"] = float(stt_started_at)
        if isinstance(outcome.get("stt_ms"), (int, float)):
            timing["stt_ms"] = float(outcome["stt_ms"])
        self._voice_turn_timing = timing

    def _log_voice_turn_timing(self, outcome: str, *, tts_payload: str = "") -> None:
        # One bounded line per voice turn, durations only (no content):
        # end of speech -> STT start, STT, engine turn (LLM included),
        # TTS request -> first audio, and end of speech -> first sound.
        # mic_closed_ms is the window in which nothing listens (from the
        # end of the capture until playback starts and the barge-in
        # monitor opens the mic again). Fields that were not reached for
        # this outcome are "n/a".
        timing = self._voice_turn_timing
        self._voice_turn_timing = None
        if timing is None:
            return

        def span(start: str, end: str) -> object:
            if start in timing and end in timing:
                return int((timing[end] - timing[start]) * 1000)
            return "n/a"

        def value(key: str) -> object:
            return int(timing[key]) if key in timing else "n/a"

        provider_first_audio_ms: object = "n/a"
        provider_synthesis_ms: object = "n/a"
        try:
            data = json.loads(tts_payload) if tts_payload else {}
        except json.JSONDecodeError:
            data = {}
        if "first_audio_ms" in data:
            provider_first_audio_ms = data.get("first_audio_ms")
        if "synthesis_ms" in data:
            provider_synthesis_ms = data.get("synthesis_ms")
        print(
            "[VOICE_TURN_TIMING] "
            f"session_id={self._voice_session_id} "
            f"outcome={outcome} "
            f"stt_model={getattr(self.voice_transcriber, 'model_name', 'unknown')} "
            f"speech_end_to_stt_start_ms={span('speech_end_at', 'stt_started_at')} "
            f"stt_ms={value('stt_ms')} "
            f"engine_ms={value('engine_ms')} "
            f"response_to_tts_request_ms={span('response_at', 'tts_requested_at')} "
            f"tts_to_first_audio_ms={span('tts_requested_at', 'first_sound_at')} "
            f"provider_first_audio_ms={provider_first_audio_ms} "
            f"provider_synthesis_ms={provider_synthesis_ms} "
            f"speech_end_to_first_sound_ms={span('speech_end_at', 'first_sound_at')} "
            f"mic_closed_ms={span('capture_ended_at', 'first_sound_at')}"
        )

    def _log_lost_speech(self, reason: str, *, approx_ms: object, **fields: object) -> None:
        # Diagnostics only: speech a VAD saw that nothing will transcribe.
        # The mic is closed between the end of a capture and the start of
        # playback, so speech in that window cannot be detected at all --
        # its length is [VOICE_TURN_TIMING] mic_closed_ms.
        parts = [f"[VOICE_LOST_SPEECH] reason={reason}", f"session_id={self._voice_session_id}", f"approx_ms={approx_ms}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts))

    def _log_lost_barge_in_candidate(self) -> None:
        # The barge-in monitor had a speech candidate (loud enough, not yet
        # sustained long enough to confirm) when TTS ended and it was
        # cancelled -- that speech is dropped. May also be TTS bleed; the
        # peak/threshold fields let a log reader tell.
        outcome = getattr(self.voice_transcriber, "last_vad_outcome", None)
        if not isinstance(outcome, dict) or not outcome.get("candidate_entered") or outcome.get("confirmed"):
            return
        self._log_lost_speech(
            "barge_in_candidate_dropped_at_tts_end",
            approx_ms=int(outcome.get("max_sustained_ms") or 0),
            required_ms=outcome.get("required_ms"),
            peak_rms=f"{float(outcome.get('peak_rms') or 0.0):.5f}",
            effective_threshold=f"{float(outcome.get('effective_threshold') or 0.0):.5f}",
        )

    def _log_barge_in(self, event: str, **fields: object) -> None:
        # Phase 7.6 (section 18): bounded diagnostics for barge-in.
        parts = [f"[BARGE_IN] event={event}", f"session_id={self._voice_session_id}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts))

    def _log_barge_in_monitor_summary(self) -> None:
        # Phase 7.6.2 (section 5): one bounded aggregate line per barge-in
        # capture -- reads assistant/voice_input.py's last_vad_outcome
        # (already populated by record_until_speech_complete for this SAME
        # worker) rather than threading a new return channel through the
        # whole call chain. Silently skipped if the worker failed before
        # ever populating an outcome (e.g. a missing dependency).
        outcome = getattr(self.voice_transcriber, "last_vad_outcome", None)
        if not isinstance(outcome, dict) or "peak_rms" not in outcome:
            return
        self._log_barge_in(
            "monitor_summary",
            peak_rms=f"{float(outcome.get('peak_rms') or 0.0):.5f}",
            baseline_rms=f"{float(outcome.get('baseline_rms') or 0.0):.5f}",
            effective_threshold=f"{float(outcome.get('effective_threshold') or 0.0):.5f}",
            absolute_floor=outcome.get("threshold"),
            max_sustained_ms=int(outcome.get("max_sustained_ms") or 0),
            required_ms=outcome.get("required_ms"),
            candidate_entered=str(bool(outcome.get("candidate_entered"))).lower(),
            confirmed=str(bool(outcome.get("confirmed"))).lower(),
        )

    def _log_voice_stop(self, event: str, **fields: object) -> None:
        # Phase 7.6.2 (section 3): bounded, state-aware diagnostics for the
        # manual STOP route -- lets a real log confirm the controller
        # actually requested a TTS stop, independent of whether the audio
        # backend honored it.
        parts = [f"[VOICE_STOP] event={event}", f"session_id={self._voice_session_id}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts))

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
            self._fail_speak_and_maybe_rearm("Não há resposta para falar.")
            return
        if self._tts_worker is not None:
            _debug_tts("[TTS]", "duplicate_request_ignored=true")
            return
        if self.tts_manager is None:
            self._fail_speak_and_maybe_rearm("A fala ainda não está ligada nesta sessão.")
            return
        manager_available = getattr(self.tts_manager, "is_available", None)
        response_language = self._response_language()
        if callable(manager_available):
            try:
                available = bool(manager_available(response_language=response_language))
            except TypeError:
                available = bool(manager_available())
            if not available:
                # Section 16: never substitute a cross-language voice (e.g.
                # speaking Portuguese with the English Anna voice) just
                # because no compatible TTS exists for response_language --
                # this early return is the SAME refusal manual SPEAK
                # already relied on; the only new behavior is rearming
                # listening afterwards when the conversation loop is on,
                # instead of leaving Echo stuck showing the notice.
                self._fail_speak_and_maybe_rearm(_tts_unavailable_message(response_language))
                return
        self._visual_token += 1
        self._tts_thread = QThread(self)
        self._tts_worker = EchoTTSWorker(
            self.tts_manager,
            message,
            self.tts_player,
            response_language,
            require_interruptible=self._current_voice_mode() != "off",
        )
        self._tts_worker.moveToThread(self._tts_thread)
        self._tts_thread.started.connect(self._tts_worker.run, Qt.ConnectionType.QueuedConnection)
        self._tts_worker.playbackStarted.connect(self._handle_tts_started, Qt.ConnectionType.QueuedConnection)
        self._tts_worker.finished.connect(self._handle_tts_finished, Qt.ConnectionType.QueuedConnection)
        self._tts_worker.failed.connect(self._handle_tts_error, Qt.ConnectionType.QueuedConnection)
        _debug_tts("[TTS]", "event=request", f"text_chars={len(message)}", f"response_language={response_language}")
        self._emit_state("working")
        self._tts_thread.start()

    def _fail_speak_and_maybe_rearm(self, message: str) -> None:
        self.ttsError.emit(message)
        if self._current_voice_mode() != "off":
            self._schedule_voice_rearm()
        else:
            self._emit_state("idle")

    @Slot()
    def stopSpeaking(self) -> None:
        self._thread_debug("stopSpeaking")
        # Phase 7.6.2 (section 2/3): state-aware, bounded diagnostics for
        # the exact real bug report -- STOP appeared to leave TTS playing.
        # The controller ROUTING here was already correct (this Slot
        # unconditionally calls _tts_worker.stop() below); the actual root
        # cause was in play_audio_file's own blocking winsound.PlaySound
        # call ignoring stop_checker mid-playback (fixed separately in
        # assistant/tts.py). These logs make the routing itself provable
        # from real logs going forward.
        was_speaking = self._tts_worker is not None
        request_state = "speaking" if was_speaking else ("listening" if self._voice_worker is not None else "idle")
        self._log_voice_stop("requested", state=request_state)
        if self._voice_worker is not None and self._voice_worker_is_barge_in and not self._voice_worker_barge_in_confirmed:
            # Phase 7.6 (section 12): a barge-in monitor was running
            # alongside playback but never confirmed anything -- an
            # explicit user STOP should cancel it too rather than leave it
            # monitoring a room that's about to go quiet;
            # _handle_tts_finished's normal rearm opens a FRESH listening
            # cycle once playback actually stops. A CONFIRMED barge-in
            # worker is left alone -- it is the capture we want to keep.
            self.cancelVoiceInput()
            self._log_voice_stop("barge_monitor_cancelled")
        if self._tts_worker is None:
            return
        if was_speaking and self._current_voice_mode() != "off":
            self._voice_stop_awaiting_listen_log = True
        self._tts_worker.stop(reason="manual_stop")
        self._log_voice_stop("tts_stop_requested")
        # Section 14: STOP during voice-mode speech interrupts playback
        # only -- _handle_tts_finished (which still fires once the worker's
        # run() loop unwinds after stop_event) is the ONE place that
        # decides idle-vs-rearm, so this must not race it with its own
        # "idle" transition when the loop is active or in standby.
        if self._current_voice_mode() == "off":
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
        self._emit_response_ready(response_text, source="test")

    def has_active_request(self) -> bool:
        return self._active

    def request_shutdown(self) -> None:
        """Phase 7.6.7 (section 9): the non-blocking first half of app
        close -- flips every voice mode off (session-token protection makes
        any late worker callback stale) and asks each owned worker to stop,
        but never waits on any of them. Idempotent."""
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self._tts_startup_prewarm_timer.stop()
        self._shutdown_started_at = time.perf_counter()
        _log_shutdown(event="started")
        self._release_local_models()
        self._stop_voice_conversation(reason="app_shutdown")
        self._stop_voice_standby(reason="app_shutdown")
        # A Whisper model load can't be cancelled mid-way; this only blocks
        # NEW prewarms -- a running one is drained like any other worker.
        self._wake_prewarm_shutting_down = True
        if self._wake_prewarm_thread is not None:
            _log_shutdown(component="wake_prewarm", event="stop_requested")
        if self._tts_prewarm_thread is not None:
            # Pocket startup is cancellable: stopping the manager closes the
            # in-flight subprocess and releases the prewarm worker promptly.
            _log_shutdown(component="tts_prewarm", event="stop_requested")
            self._shutdown_tts_manager()
        if self._voice_worker is not None:
            self._voice_worker.stop()
            _log_shutdown(component="voice_worker", event="stop_requested")
        if self._tts_worker is not None:
            self._tts_worker.stop(reason="shutdown")
            _log_shutdown(component="tts_worker", event="stop_requested")
            # A worker mid-synthesis is blocked on the Pocket subprocess,
            # not on stop_event -- tearing the manager down is what unblocks
            # it (the same order the pre-7.6.7 shutdown used).
            self._shutdown_tts_manager()
        if self._thread is not None:
            self._thread.requestInterruption()
            _log_shutdown(component="request_worker", event="stop_requested")
        self._shutdown_pending_seen = set(self.shutdown_pending_components())

    def shutdown_pending_components(self) -> list[str]:
        """Owned QThreads not yet finished AND cleaned up -- each reference
        is cleared by that worker's own cleanup slot (or, for the prewarm
        thread, _on_wake_prewarm_finished), so "not None" means a
        synchronous wait right now could still block."""
        pending: list[str] = []
        if self._wake_prewarm_thread is not None:
            pending.append("wake_prewarm")
        if self._tts_prewarm_thread is not None:
            pending.append("tts_prewarm")
        if self._voice_thread is not None:
            pending.append("voice_worker")
        if self._tts_thread is not None:
            pending.append("tts_worker")
        if self._thread is not None:
            pending.append("request_worker")
        return pending

    def is_shutdown_complete(self) -> bool:
        return self._shutdown_completed

    def shutdown_async(
        self,
        on_complete: Callable[[], None] | None = None,
        *,
        session_close: Callable[[], None] | None = None,
    ) -> None:
        """Phase 7.6.7 (section 9): the close sequence the window uses.
        Requests cancellation immediately, then polls on a QTimer -- the Qt
        event loop keeps running, so queued finished/cleanup slots are
        delivered normally and every QThread ends on its own (no
        terminate(), no destroyed-while-running wrapper). Only once nothing
        is left (or the drain deadline passes) does the synchronous
        shutdown() run, by which point it has nothing left to block on.
        session_close (the app's own end-of-session persistence) runs after
        that on the same [SHUTDOWN] timeline, then on_complete fires
        exactly once."""
        self._shutdown_on_complete = on_complete
        self._shutdown_session_close = session_close
        self.request_shutdown()
        self._poll_shutdown_drain()

    def _poll_shutdown_drain(self) -> None:
        pending = set(self.shutdown_pending_components())
        for component in sorted(self._shutdown_pending_seen - pending):
            _log_shutdown(component=component, event="finished", elapsed_ms=self._shutdown_elapsed_ms())
        self._shutdown_pending_seen = pending
        if pending and self._shutdown_elapsed_ms() < self._shutdown_deadline_ms:
            QTimer.singleShot(self._SHUTDOWN_POLL_MS, self._poll_shutdown_drain)
            return
        if pending:
            # Still safe: shutdown() below keeps its own bounded waits and
            # never force-terminates anything.
            _log_shutdown(event="drain_deadline_exceeded", pending=",".join(sorted(pending)))
        self.shutdown(log_completed=False)
        session_close = self._shutdown_session_close
        self._shutdown_session_close = None
        if session_close is not None:
            # Runs on the GUI thread but after the window is hidden (see
            # EchoOSWindow.closeEvent), so a slow step here -- the session
            # summary's synchronous Ollama embedding call -- can no longer
            # show up as a frozen "Não responde" window.
            _log_shutdown(component="session_close", event="started", elapsed_ms=self._shutdown_elapsed_ms())
            try:
                session_close()
            except Exception as exc:
                _debug_ui(f"[Echo UI] shutdown=session_close_failed error={type(exc).__name__}")
            _log_shutdown(component="session_close", event="finished", elapsed_ms=self._shutdown_elapsed_ms())
        self.complete_shutdown()
        callback = self._shutdown_on_complete
        self._shutdown_on_complete = None
        if callback is not None:
            callback()

    def shutdown(self, wait_ms: int = 1500, *, log_completed: bool = True) -> None:
        """The synchronous teardown. Direct callers (tests) keep the
        original blocking semantics; the window's close path only calls it
        after shutdown_async() has drained every worker, so each step here
        is effectively instantaneous on that path."""
        self.request_shutdown()
        had_prewarm = self._wake_prewarm_thread is not None
        self._shutdown_wake_prewarm_thread(wait_ms)
        if had_prewarm:
            _log_shutdown(component="wake_prewarm", event="finished", elapsed_ms=self._shutdown_elapsed_ms())
        self._shutdown_tts_prewarm_thread(wait_ms)
        if self._voice_worker is not None:
            self._voice_worker.stop()
            self._cleanup_voice_worker()
            _log_shutdown(component="voice_worker", event="finished", elapsed_ms=self._shutdown_elapsed_ms())
        self._shutdown_tts_manager()
        close_stt = getattr(self.voice_transcriber, "close", None)
        if callable(close_stt):
            close_stt()
        if self._tts_worker is not None:
            self._tts_worker.stop(reason="shutdown")
            self._cleanup_tts_worker()
            _log_shutdown(component="tts_worker", event="finished", elapsed_ms=self._shutdown_elapsed_ms())
        if self._thread is not None:
            self._thread.requestInterruption()
            self._thread.quit()
            if self._thread.wait(wait_ms):
                _log_shutdown(component="request_worker", event="finished", elapsed_ms=self._shutdown_elapsed_ms())
            else:
                _debug_ui("[Echo UI] shutdown=worker_still_running")
                _log_shutdown(component="request_worker", event="still_running", elapsed_ms=self._shutdown_elapsed_ms())
        if log_completed:
            self.complete_shutdown()

    def _shutdown_tts_manager(self) -> None:
        """Tears down the shared TTS manager (the long-lived Pocket worker
        subprocess) exactly once. Previously this only happened when a TTS
        worker was active at close time, so a Pocket server started by an
        earlier SPEAK was never stopped. Pocket's own close() bounds its
        process wait (2s, then kill)."""
        if self._tts_manager_shut_down:
            return
        self._tts_manager_shut_down = True
        stop = getattr(self.tts_manager, "stop", None)
        if not callable(stop):
            return
        _log_shutdown(component="tts_manager", event="stop_requested")
        try:
            stop()
        except Exception as exc:
            _debug_ui(f"[Echo UI] shutdown=tts_manager_stop_failed error={type(exc).__name__}")
        _log_shutdown(component="tts_manager", event="finished", elapsed_ms=self._shutdown_elapsed_ms())

    def log_shutdown_step(self, component: str, event: str) -> None:
        """Lets the window put its own post-controller steps (session
        close) on the same [SHUTDOWN] timeline."""
        _log_shutdown(component=component, event=event, elapsed_ms=self._shutdown_elapsed_ms())

    def complete_shutdown(self) -> None:
        if self._shutdown_completed:
            return
        self._shutdown_completed = True
        _log_shutdown(event="completed", total_ms=self._shutdown_elapsed_ms())

    def _shutdown_elapsed_ms(self) -> int:
        if self._shutdown_started_at is None:
            return 0
        return int((time.perf_counter() - self._shutdown_started_at) * 1000)

    def _shutdown_tts_prewarm_thread(self, wait_ms: int) -> None:
        # Same ownership/isValid rule as _shutdown_wake_prewarm_thread; runs
        # before _shutdown_tts_manager() so the manager stop never blocks on
        # a server start still holding the provider lock.
        thread = self._tts_prewarm_thread
        if thread is None:
            return
        if shiboken6.isValid(thread) and thread.isRunning() and not thread.wait(wait_ms):
            _debug_ui("[Echo UI] shutdown=tts_prewarm_still_running")
        self._tts_prewarm_thread = None
        _log_shutdown(component="tts_prewarm", event="finished", elapsed_ms=self._shutdown_elapsed_ms())

    def _shutdown_wake_prewarm_thread(self, wait_ms: int) -> None:
        """Phase 7.6.6.1: the shutdown-time half of the ownership rule --
        the controller's OWN reference (self._wake_prewarm_thread) is the
        single source of truth for whether a live QThread wrapper exists.
        Never touches a wrapper without first confirming (via shiboken6.
        isValid) the underlying C++ object is still there -- a real crash
        (`libshiboken: Internal C++ object ... already deleted`) was caused
        by calling .wait() on a reference that had already been scheduled
        for deletion by an earlier finished->deleteLater connection, with
        nothing ever clearing the controller's own attribute. That gap is
        closed at the source now (_on_wake_prewarm_finished clears it the
        instant the thread finishes, before requesting deletion), so this
        isValid check is secondary/defensive, not the primary fix -- it
        guards the (already very unlikely) case of a wrapper that is
        somehow still referenced but no longer backed by a live object.

        Idempotent: prevents new prewarm starts (the guard in
        _prewarm_wake_recognizer), then does at most one wait and clears
        the reference exactly once -- calling shutdown() a second time
        finds self._wake_prewarm_thread already None and returns
        immediately (no double wait, no double delete)."""
        self._wake_prewarm_shutting_down = True
        thread = self._wake_prewarm_thread
        if thread is None:
            return
        if not shiboken6.isValid(thread):
            self._log_wake_prewarm("shutdown_already_finished")
            self._wake_prewarm_thread = None
            return
        if thread.isRunning():
            self._log_wake_prewarm("shutdown_wait")
            if not thread.wait(wait_ms):
                _debug_ui("[Echo UI] shutdown=wake_prewarm_still_running")
        else:
            self._log_wake_prewarm("shutdown_already_finished")
        self._wake_prewarm_thread = None
        self._log_wake_prewarm("reference_cleared")

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
            # Phase 7.4 (section 6/12/13): auto-speak ONLY when this turn
            # actually started under the voice-conversation session that is
            # STILL current -- covers both "mode was stopped while
            # thinking" (session flips, active flips False) and "mode was
            # stopped then restarted while thinking" (session no longer
            # matches even though active is True again) in one check.
            should_auto_speak = (
                self._voice_conversation_active
                and self._voice_turn_session_id is not None
                and self._voice_turn_session_id == self._voice_session_id
                and bool(response_text.strip())
                and last_progress_event not in {"rewrite_cancelled", "rewrite_timeout"}
            )
            if should_auto_speak:
                self._log_voice("response_ready")
            if self._voice_turn_timing is not None:
                now = time.perf_counter()
                self._voice_turn_timing["response_at"] = now
                if self._started_at:
                    self._voice_turn_timing["engine_ms"] = (now - self._started_at) * 1000.0
                if not should_auto_speak:
                    self._log_voice_turn_timing("no_tts")
            if last_progress_event == "rewrite_cancelled":
                # The request already resolved to "Pedido cancelado." -- go
                # straight to idle instead of playing the normal thinking->
                # speaking animation over a response nobody asked to see spoken.
                self._emit_state("idle")
            elif last_progress_event == "rewrite_timeout":
                self._emit_state("error")
                QTimer.singleShot(900, lambda: self._emit_state("idle"))
            elif should_auto_speak:
                voice_session_snapshot = self._voice_session_id
                elapsed_ms = int((time.perf_counter() - self._started_at) * 1000) if self._started_at else 0
                delay_ms = max(0, self.thinking_min_ms - elapsed_ms)
                QTimer.singleShot(delay_ms, lambda: self._auto_speak_response(response_text, voice_session_snapshot))
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
            self._voice_turn_session_id = None

    @Slot(str)
    def _handle_error(self, error: str) -> None:
        self._thread_debug("_handle_error")
        _debug_ui(f"[Echo UI] error={error}")
        error_text = "Não consegui responder a este pedido."
        # A genuine turn-processing failure (not STT/TTS) -- still recovers
        # the voice loop rather than stranding it, same reasoning as
        # section 17/18's TTS/STT failure recovery.
        should_rearm = self._voice_conversation_active and self._voice_turn_session_id == self._voice_session_id
        self._log_voice_turn_timing("engine_error")
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
            if should_rearm:
                self._schedule_voice_rearm(delay_ms=900)
            else:
                QTimer.singleShot(1200, lambda: self._emit_state("idle"))
        except Exception as exc:
            _debug_ui(f"[Echo UI DEBUG] error_finalization_error={exc}")
            QTimer.singleShot(0, lambda: self._emit_state("idle"))
        finally:
            self._cleanup_worker()
            self._voice_turn_session_id = None

    @Slot(str)
    def _handle_progress(self, event: str) -> None:
        self._thread_debug("_handle_progress")
        _debug_ui(f"[Echo UI] progress_event={event}")
        self._last_progress_event = str(event or "")
        self.uiEvent.emit(_json_payload({"type": str(event or "")}))

    @Slot(str)
    def _handle_stt_status(self, status: str) -> None:
        if self._shutdown_requested:
            return
        if self._response_language().startswith("pt"):
            status = "O reconhecimento de voz está a funcionar em modo lento."
        self.uiEvent.emit(_json_payload({"type": "voice_status", "status": status}))

    @Slot(str)
    def _handle_voice_status(self, status: str) -> None:
        clean_status = str(status or "").strip()
        lowered = clean_status.lower()
        if "ouvir" in lowered:
            self._emit_state("listening")
        elif "process" in lowered or "conclu" in lowered:
            self._emit_state("thinking")
        key = {
            "Preparar...": "voice.status.preparing",
            "A ouvir...": "voice.status.listening",
            "A processar...": "voice.status.processing",
            "Concluído.": "voice.status.done",
        }.get(clean_status)
        if key:
            clean_status = localize(key, self._response_language())
        self.uiEvent.emit(_json_payload({"type": "voice_status", "status": clean_status}))

    @Slot(object)
    def _handle_voice_transcription(self, text) -> None:
        self._thread_debug("_handle_voice_transcription")
        if isinstance(text, TurnInput):
            capture = text
        else:
            capture = TurnInput(text=str(text or ""), source="voice", capture_mode=self._voice_worker_mode,
                                session_id=str(self._voice_session_id))
        clean = capture.text.strip()
        expected_capture = getattr(self._voice_worker, "capture_id", None)
        if (self._is_voice_worker_stale() or (isinstance(text, TurnInput) and expected_capture is not None and capture.capture_id != expected_capture)
                or (capture.session_id is not None and capture.session_id != str(self._voice_session_id))):
            # Section 3: this worker belongs to a session/mode that was
            # already stopped or superseded -- a late result (even a
            # genuine transcript) must never reach the composer/UI or
            # auto-submit.
            self._log_voice("stale_callback_ignored", context="transcription")
            self._settle_idle_if_nothing_pending()
            return
        if capture.capture_mode == "standby":
            # Phase 7.6 (section 5): STANDBY never emits voiceTranscriptionReady
            # or touches the composer/response area -- the transcript is
            # purely a wake/ignore decision, not conversational content.
            self._handle_standby_transcript(clean)
            return
        self.voiceTranscriptionReady.emit(clean)
        if not self._voice_conversation_active:
            self._emit_state("idle")
            return
        if not clean:
            # Section 10/4: no speech detected -- a control-flow event, not
            # content -- never submit an empty turn, never show a fatal
            # error, just rearm. Phase 7.5 (section 9): with VAD in place, a
            # listening cycle only reaches this branch after a real capture
            # window (bounded below by the trailing-silence window, above by
            # the max-utterance ceiling) rather than a rapid fixed-interval
            # retry -- the Phase 7.4.1 backoff existed specifically to guard
            # against rapid repeated cycles, which VAD already structurally
            # prevents, so it was retired as redundant complexity.
            self._log_voice("no_speech_rearm")
            self._schedule_voice_rearm()
            return
        if classify_voice_wake_intent(clean) == "sleep":
            # Phase 7.6 (section 7): a sleep command during ACTIVE mode is
            # control-flow, not a task/query -- never routed through
            # submitMessage()/the response pipeline.
            self._log_wake("sleep_intent_matched")
            self._enter_voice_standby(reason="sleep_command")
            return
        self._voice_conversation_failures = 0
        self._barge_in_recent = False
        self._log_voice("transcript_ready", chars=len(clean))
        self._start_voice_turn_timing()
        self._submit_turn(capture)

    @Slot(str)
    def _handle_voice_error(self, error: str) -> None:
        self._thread_debug("_handle_voice_error")
        if self._is_voice_worker_stale():
            # Section 3: this worker belongs to a session/mode that was
            # already stopped or superseded (e.g. voice mode turned off, or
            # a sleep/wake transition happened, while it was still
            # recording) -- its late failure, even a genuine one, must
            # never replace the main Echo response, flip to the error
            # state, or restart listening.
            self._log_voice("stale_callback_ignored", context="voice_error")
            self._settle_idle_if_nothing_pending()
            return
        current_mode = self._current_voice_mode()
        error_text = str(error or "Não consegui transcrever o áudio.")
        if current_mode != "off" and _is_benign_voice_retry_error(error_text):
            # Section 11/4: a too-short/cancelled capture is a normal,
            # expected control-flow occurrence (accidental click/noise/an
            # interruption from stopping or a typed turn) -- never fill the
            # main response area with a scary error for it, just rearm (see
            # the section-9 note above -- no backoff needed once VAD bounds
            # cycle length structurally).
            if self._barge_in_recent and "curta" in error_text.lower():
                outcome = getattr(self.voice_transcriber, "last_vad_outcome", None) or {}
                self._log_lost_speech(
                    "short_capture_after_barge_in",
                    approx_ms=int(float(outcome.get("speech_ms") or 0.0)),
                    min_speech_ms=getattr(getattr(self.voice_transcriber, "vad_config", None), "min_speech_ms", "n/a"),
                    carried_over=getattr(self.voice_transcriber, "_barge_in_carryover", None) is not None,
                )
            self._barge_in_recent = False
            self._log_voice("benign_retry")
            self._schedule_voice_rearm()
            return
        if current_mode == "standby":
            # Standby is a passive wait -- a genuine STT error still
            # recovers the loop (or stops it after repeated failures), but
            # without flashing the "error" state, which would be a
            # confusing distraction while nothing was even being said to
            # Echo on purpose.
            self.voiceInputError.emit(error_text)
            self._voice_conversation_failures += 1
            if self._voice_conversation_failures >= self._VOICE_MAX_CONSECUTIVE_FAILURES:
                self._stop_voice_standby(reason="stt_failures")
                return
            self._schedule_voice_rearm(delay_ms=900)
            return
        self.voiceInputError.emit(error_text)
        if self._voice_conversation_active:
            # Section 18: a genuine STT error (missing dependency, mic
            # failure, Whisper crash...) still recovers the loop, but stops
            # it outright after too many in a row instead of retrying
            # forever.
            self._voice_conversation_failures += 1
            if self._voice_conversation_failures >= self._VOICE_MAX_CONSECUTIVE_FAILURES:
                self._stop_voice_conversation(reason="stt_failures")
                return
            self._emit_state("error")
            self._schedule_voice_rearm(delay_ms=900)
            return
        self._emit_state("error")
        QTimer.singleShot(900, lambda: self._emit_state("idle"))

    @Slot()
    def _cleanup_voice_worker(self) -> None:
        self._thread_debug("_cleanup_voice_worker")
        if self._voice_worker_is_barge_in:
            self._log_barge_in_monitor_summary()
            if self._barge_in_cancelled_at_tts_end:
                self._log_lost_barge_in_candidate()
        self._barge_in_cancelled_at_tts_end = False
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
        if self._voice_turn_timing is not None and "tts_requested_at" in self._voice_turn_timing:
            self._voice_turn_timing["first_sound_at"] = time.perf_counter()
            self._log_voice_turn_timing("spoken", tts_payload=payload)
        if self._voice_conversation_active:
            self._log_voice("speaking_started")
        self._emit_state("speaking")
        self.ttsStarted.emit(str(payload or "{}"))
        self._begin_barge_in_monitor()

    def _cancel_unconfirmed_barge_in_monitor(self) -> bool:
        """Phase 7.6 (section 12): True (and cancels it) if a barge-in
        monitor is still running but never confirmed real speech -- called
        whenever TTS ends (naturally, on error, or via STOP) so the
        detector doesn't keep listening for up to max_utterance_ms with
        nothing left to interrupt. A CONFIRMED barge-in worker is left
        completely alone: it IS the capture we want to keep running."""
        if self._voice_worker is not None and self._voice_worker_is_barge_in and not self._voice_worker_barge_in_confirmed:
            self._barge_in_cancelled_at_tts_end = True
            self.cancelVoiceInput()
            return True
        return False

    @Slot(str)
    def _handle_tts_finished(self, payload: str) -> None:
        self._thread_debug("_handle_tts_finished")
        _debug_tts("[TTS]", "event=playback_finished", _bounded_tts_payload(payload), "success=true")
        if self._voice_conversation_active:
            self._log_voice("speaking_finished")
        self._cleanup_tts_worker()
        self.ttsFinished.emit(str(payload or "{}"))
        if self._cancel_unconfirmed_barge_in_monitor():
            # That cancelled worker's OWN (benign) completion handler will
            # schedule the next rearm once cleanup finishes -- scheduling
            # one here too would just be a redundant, no-op duplicate.
            return
        # Section 8/9: this fires whether playback finished naturally OR
        # was interrupted by STOP (see stopSpeaking()) -- either way, mic
        # capture must stay closed until here, then reopen after the fixed
        # rearm delay so Echo's own tail/room reverberation isn't captured.
        if self._current_voice_mode() != "off":
            self._schedule_voice_rearm()
            return
        self._emit_state("idle")

    @Slot(str)
    def _handle_tts_error(self, error: str) -> None:
        self._thread_debug("_handle_tts_error")
        error_text = str(error or "Não consegui falar a resposta.")
        _debug_tts("[TTS]", "event=error", "success=false", f"failure_reason={error_text[:160]}")
        self._log_voice_turn_timing("tts_error")
        self._cleanup_tts_worker()
        self.ttsError.emit(error_text)
        if self._cancel_unconfirmed_barge_in_monitor():
            return
        if self._current_voice_mode() != "off":
            # Section 17: keep the textual response (already shown), keep
            # the loop alive -- rearm instead of leaving Echo stuck on a
            # TTS failure (Pocket crash/timeout/all-providers-unavailable).
            self._schedule_voice_rearm()
            return
        if _is_tts_capability_unavailable(error_text):
            self._emit_state("idle")
            return
        self._emit_state("error")
        QTimer.singleShot(900, lambda: self._emit_state("idle"))

    def _begin_barge_in_monitor(self) -> None:
        """Phase 7.6 (section 8/11): while Anna is speaking in ACTIVE mode,
        monitor the mic with a conservative VAD for a genuine interruption.
        Duck-typed against the transcriber's ACTUAL capability (mirrors the
        Phase 7.5 use_vad fallback): a transcriber double that doesn't
        support the full barge-in interface (the vad_config/
        on_speech_confirmed/require_confirmation_before_recording kwargs --
        e.g. every existing pre-Phase-7.6 test fake) safely does nothing
        here instead of crashing a worker thread with a TypeError."""
        if not self._voice_conversation_active or self._voice_worker is not None:
            return
        vad_method = getattr(self.voice_transcriber, "record_and_transcribe_vad", None)
        if not callable(vad_method):
            return
        try:
            import inspect

            params = inspect.signature(vad_method).parameters
        except (TypeError, ValueError):
            return
        if "on_speech_confirmed" not in params or "require_confirmation_before_recording" not in params:
            return
        self._log_barge_in("detector_started")
        self._voice_worker_session_id = self._voice_session_id
        self._voice_worker_mode = "active"
        self._voice_worker_is_barge_in = True
        self._voice_worker_barge_in_confirmed = False
        self._voice_thread = QThread(self)
        self._voice_worker = EchoVoiceWorker(self.voice_transcriber, mode="barge_in", session_id=self._voice_worker_session_id)
        self._voice_worker.moveToThread(self._voice_thread)
        self._voice_thread.started.connect(self._voice_worker.run, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.speechConfirmed.connect(self._handle_barge_in_confirmed, Qt.ConnectionType.QueuedConnection)
        # Deliberately NOT connected to _handle_voice_status: a barge-in
        # monitor's "A ouvir..." status must never flip the UI to
        # "listening" while Anna is still visibly/audibly speaking.
        self._voice_worker.transcriptionReady.connect(self._handle_voice_transcription, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.failed.connect(self._handle_voice_error, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.transcriptionReady.connect(self._cleanup_voice_worker, Qt.ConnectionType.QueuedConnection)
        self._voice_worker.failed.connect(self._cleanup_voice_worker, Qt.ConnectionType.QueuedConnection)
        self._voice_thread.start()

    @Slot()
    def _handle_barge_in_confirmed(self) -> None:
        self._thread_debug("_handle_barge_in_confirmed")
        if self._voice_worker is None or not self._voice_worker_is_barge_in:
            return
        if self._is_voice_worker_stale():
            return
        self._voice_worker_barge_in_confirmed = True
        self._barge_in_recent = True
        self._log_barge_in("confirmed")
        if self._tts_worker is not None:
            # Direct worker-level stop, NOT self.stopSpeaking() -- that
            # method would also cancel THIS SAME barge-in worker (see its
            # own comment), which is exactly the capture we want to keep
            # running to completion (section 8: "start/continue normal VAD
            # capture").
            self._tts_worker.stop(reason="barge_in")
            self._log_barge_in("tts_interrupted")
        self._log_barge_in("capture_handoff")

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

    def _emit_response_ready(self, response_text: str, *, source: str = "request") -> None:
        _debug_ui(
            "[Echo UI DEBUG] emitting responseReady",
            repr(response_text),
            type(response_text),
            f"controller_id={id(self)}",
        )
        _lifecycle_trace("RESPONSE_READY_EMITTED")
        # Phase 8.1.1: the Python half of the [UI_RESPONSE] trace -- the JS
        # half (received/render_start/render_complete/cleared) is logged by
        # echo_ui.js through the page console. Counts only, no content.
        print(
            "[UI_RESPONSE] event=emit "
            f"chars={len(response_text)} source={source} "
            f"voice_mode={self._current_voice_mode()} controller_id={id(self)}"
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

    # The 9 semantic states from the visual-state redesign. "listening" has
    # no emitter yet (no voice-capture state machine exists) -- it's
    # whitelisted so the existing setState QWebChannel slot can already
    # drive it once that's wired up, without touching this whitelist again.
    _KNOWN_STATES = {
        "idle", "listening", "thinking", "reading", "working", "speaking", "error",
    }

    # Phase 7.4 (section 9): small, fixed post-speech delay before reopening
    # the mic -- avoids capturing speaker/room reverberation tail. Not a
    # user-facing setting yet (section 9 explicitly says not to make it one).
    # ACTIVE conversation VAD/timing is explicitly out of scope for Phase
    # 7.6.5 -- this delay is unchanged for ACTIVE.
    _VOICE_REARM_DELAY_MS = 350
    # Phase 7.6.5 (section 14): STANDBY has no reverberation tail to guard
    # against in the same way -- it isn't reopening the mic right after
    # Echo's OWN speech, just after an ignored/failed wake attempt -- so the
    # same margin is unnecessary caution that only adds to perceived wake
    # latency. Lowered to 150ms (within the suggested 100-200ms range),
    # still well clear of a rapid noisy retry loop.
    _VOICE_STANDBY_REARM_DELAY_MS = 150
    # Phase 7.6.7: how often the async close checks whether workers have
    # drained, and how long it lets them (a cold Medium wake fallback was
    # measured at ~9s) before falling back to shutdown()'s bounded waits.
    _SHUTDOWN_POLL_MS = 25
    _SHUTDOWN_DRAIN_DEADLINE_MS = 15000
    # Section 18: stop the loop after this many consecutive GENUINE STT
    # errors (dependency/hardware failures -- NOT no-speech/too-short,
    # which never reach this counter at all) instead of retrying forever.
    _VOICE_MAX_CONSECUTIVE_FAILURES = 3

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
            payload = self.model_runtime.request_started_payload()
        else:
            telemetry = self.get_telemetry_callback() if self.get_telemetry_callback is not None else None
            if telemetry:
                payload = self.model_runtime.telemetry_payload(telemetry, state=state)
            else:
                payload = self.model_runtime.current_payload(state=state)
        return _json_payload(self._with_response_language(payload))

    def _with_response_language(self, payload: dict) -> dict:
        # The page learns the fixed response_language from this existing
        # telemetry channel (startup getModelTelemetry + every turn) -- no
        # dedicated signal. Only set when known.
        response_language = self._response_language()
        if response_language and isinstance(payload, dict):
            return {**payload, "response_language": response_language}
        return payload

    def _emit_runtime_telemetry(self, state: str, error_message: str = "") -> None:
        if self.model_runtime is None:
            return
        try:
            if state == "error":
                payload = self.model_runtime.error_payload(error_message or "Erro.", provider_error_type="ui_error")
                encoded = _json_payload(self._with_response_language(payload))
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


def _log_shutdown(**fields: object) -> None:
    # Phase 7.6.7 (section 8): bounded, one line per shutdown step --
    # component/event/elapsed_ms since request_shutdown() only.
    print(" ".join(["[SHUTDOWN]", *(f"{key}={value}" for key, value in fields.items())]))


def _log_tts_playback(event: str, **fields: object) -> None:
    # Phase 7.6.3 (section 14): bounded playback-backend diagnostics -- one
    # line per started/stopped/completed transition, never per audio frame.
    parts = [f"[TTS_PLAYBACK] event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts))


def _json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _bounded_voice_error(exc: Exception) -> str:
    text = sanitize_secret_error(exc)
    if isinstance(exc, VoiceInputError):
        return text[:240]
    return f"Não consegui usar a voz: {text}"[:240]


def _is_benign_voice_retry_error(message: str) -> bool:
    """Phase 7.4/7.4.1 (section 11/4): "recording too short" and "no audio
    captured at all" are plain VoiceInputError messages (assistant/
    voice_input.py raises them directly, not the VoiceNoSpeechDetected
    subclass), so they reach _handle_voice_error like any other failure --
    but both are ordinary, expected control-flow occurrences (accidental
    click/ambient noise, or a capture cancelled almost instantly by
    stopping voice mode / a typed turn arriving before any audio was even
    buffered), not real STT problems, so neither must ever count toward the
    conversation-mode failure limit or show a scary error."""
    text = str(message or "").strip().lower()
    return "curta" in text or "captado" in text


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
        "model",
        "device",
        "model_load_ms",
        "text_chars",
        "synthesis_ms",
        "playback_ms",
        "reference_used",
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
    # Phase 7.7 (section 1): the pt-PT branch is inherently Portuguese
    # (it only ever fires when response_language IS pt-PT) -- but the
    # fallback branch used to ALSO be hardcoded Portuguese regardless of
    # response_language, so an en-GB session with no compatible voice saw
    # a Portuguese notice. localize() makes that branch respect en-GB too.
    if str(response_language or "").strip() == "pt-PT":
        return "A voz portuguesa ainda não está disponível."
    return localize("tts.unavailable", response_language)
