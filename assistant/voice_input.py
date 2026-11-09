from __future__ import annotations

import shutil
import tempfile
import threading
import time
import wave
from array import array
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from shutil import which
from typing import Any

from assistant.audio_device import (
    choose_best_input_device,
    classify_rms_level,
    list_input_devices,
    microphone_test_report,
    resolve_input_device,
)
from assistant.wake_recognizer import WakeRecognizer
from assistant.voice_wake import assess_fast_wake_transcript, classify_voice_wake_intent


DEFAULT_INITIAL_PROMPT = (
    "Transcricao em portugues de Portugal europeu. "
    "Contexto possivel: viagens, datas, tarefas, pesquisa, orcamento, alojamento, "
    "Portugal, Estocolmo, projetos pessoais e conversa com o Echo."
)
DEFAULT_DEBUG_AUDIO_PATH = Path("data/debug/last_voice_input.wav")


class VoiceInputError(RuntimeError):
    """Raised when local voice input cannot run."""


class VoiceNoSpeechDetected(VoiceInputError):
    """Raised when recording worked but no reliable speech was detected."""


class MicrophoneCheckError(RuntimeError):
    """Raised when the microphone cannot be opened."""


@dataclass(frozen=True)
class VoiceAudioDiagnostics:
    path: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_seconds: float
    rms: float
    peak: float
    near_silence_percent: float
    clipping_detected: bool
    level: str


@dataclass(frozen=True)
class VoiceStartDiagnostics:
    mic_click_to_stream_ms: float | None = None
    stream_to_first_frame_ms: float | None = None
    first_frame_to_listening_status_ms: float | None = None
    preroll_frames: int = 0
    preroll_duration_seconds: float = 0.0
    selected_device_index: int | None = None
    selected_device_name: str = ""


@dataclass(frozen=True)
class VoiceSttDecision:
    accepted: bool
    rejection_reason: str = ""
    no_speech_probability: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


@dataclass(frozen=True)
class VadConfig:
    """Phase 7.5: tuning for the lightweight RMS-based end-of-speech
    detector used by Voice Conversation Mode (record_until_speech_complete).
    Not a Whisper-based VAD (section 2) -- a handful of float comparisons
    per ~40ms audio block. speech_rms_threshold in particular is an
    estimate that needs validating against a real microphone/room -- it has
    not been tuned with live audio by this change."""

    speech_rms_threshold: float = 0.01
    # Debounce against a single click/pop registering as "speech started".
    speech_start_min_ms: int = 120
    # Section 5: suggested 700-1200ms, starting at 900ms.
    trailing_silence_ms: int = 900
    # Section 6: total accumulated speech below this is treated as an
    # accidental noise burst, not a real utterance.
    min_speech_ms: int = 300
    # Section 7: safety ceiling only -- not a target duration.
    max_utterance_ms: int = 60_000
    # Phase 7.6.3 (section 10): optional adaptive baseline -- barge-in
    # only, None (the default) elsewhere. When set, `speech_rms_threshold`
    # becomes a FLOOR (the effective threshold is never lower than it,
    # however quiet the room is) rather than the sole trigger level; the
    # effective threshold is max(speech_rms_threshold, rolling_baseline *
    # adaptive_multiplier), where the baseline is an exponential moving
    # average of RMS updated only from blocks NOT currently over threshold
    # (so the user's own confirmed speech never drags the baseline up).
    # This is a bounded heuristic, not echo cancellation/speaker
    # separation -- it adapts to whatever ambient/TTS-bleed level the real
    # room+speaker+mic setup produces instead of requiring a single fixed
    # number to be correct for every setup.
    adaptive_multiplier: float | None = None
    adaptive_baseline_alpha: float = 0.05
    # Phase 7.6.3: a brief window (ms), counted from the START of a
    # capture, during which EVERY block (loud or quiet) seeds the baseline
    # and NONE count toward candidate detection. Without this, bleed
    # louder than the absolute floor from block 1 (no quiet block ever
    # occurs to teach the baseline) can false-confirm before the baseline
    # gets any chance to learn it. Only meaningful alongside
    # adaptive_multiplier; ignored otherwise.
    adaptive_settle_ms: int = 150


class _SpeechActivityTracker:
    """Phase 7.5 (section 3): a tiny per-audio-block state machine --
    waiting_for_speech -> speech_active -> (trailing_silence) ->
    utterance_complete. Runs entirely on the sounddevice audio-callback
    thread; update() is a few float comparisons per ~40ms block, orders of
    magnitude cheaper than a Whisper inference (section 2/16)."""

    def __init__(self, config: VadConfig, blocksize: int, sample_rate: int) -> None:
        self.config = config
        self.block_ms = (blocksize / sample_rate) * 1000.0 if sample_rate else 0.0
        self.state = "waiting_for_speech"
        self.above_threshold_ms = 0.0
        self.trailing_silence_ms = 0.0
        self.speech_ms = 0.0
        self.total_ms = 0.0
        self.done = threading.Event()
        self.reason = ""
        # Phase 7.6.2 (section 5): aggregate-only diagnostics -- never
        # logged per-frame, only summarized once a capture ends -- so a
        # real barge-in failure ("detector_started" and then nothing) can
        # be diagnosed: was mic input ever loud enough at all, how close
        # did it get to the sustained-duration requirement, what was the
        # loudest single block observed.
        self.peak_rms = 0.0
        self.max_above_threshold_ms = 0.0
        self.candidate_entered = False
        # Phase 7.6.3 (section 10/14): rolling baseline for the optional
        # adaptive threshold -- 0.0 until the first quiet block is seen.
        # last_effective_threshold mirrors whatever was actually compared
        # against on the MOST RECENT update() call, for diagnostics.
        self.baseline_rms = 0.0
        self.last_effective_threshold = config.speech_rms_threshold
        self._baseline_initialized = False

    def _effective_threshold(self) -> float:
        if self.config.adaptive_multiplier is None:
            return self.config.speech_rms_threshold
        return max(self.config.speech_rms_threshold, self.baseline_rms * self.config.adaptive_multiplier)

    def _feed_baseline(self, rms: float) -> None:
        if not self._baseline_initialized:
            self.baseline_rms = rms
            self._baseline_initialized = True
        else:
            alpha = self.config.adaptive_baseline_alpha
            self.baseline_rms = (1 - alpha) * self.baseline_rms + alpha * rms

    def update(self, rms: float) -> None:
        if self.done.is_set():
            return
        self.total_ms += self.block_ms
        self.peak_rms = max(self.peak_rms, rms)
        if self.config.adaptive_multiplier is not None and self.total_ms <= self.config.adaptive_settle_ms:
            # Settling: unconditionally learn whatever level is ALREADY
            # present (e.g. Anna's own bleed, present from the instant
            # monitoring starts) before candidate detection begins -- a
            # too-low floor would otherwise never see a "quiet" block to
            # learn from if bleed exceeds it from block 1.
            self._feed_baseline(rms)
            self.last_effective_threshold = self._effective_threshold()
            if self.total_ms >= self.config.max_utterance_ms:
                self._finish("max_duration")
            return
        threshold = self._effective_threshold()
        self.last_effective_threshold = threshold
        is_loud = rms >= threshold
        if self.config.adaptive_multiplier is not None and not is_loud:
            # Only quiet blocks feed the baseline -- the user's own
            # confirmed speech (or a sustained false-candidate streak)
            # must never drag the baseline up to chase itself.
            self._feed_baseline(rms)
        if self.state == "waiting_for_speech":
            if is_loud:
                self.above_threshold_ms += self.block_ms
                self.candidate_entered = True
                self.max_above_threshold_ms = max(self.max_above_threshold_ms, self.above_threshold_ms)
                if self.above_threshold_ms >= self.config.speech_start_min_ms:
                    self.state = "speech_active"
                    self.speech_ms = self.above_threshold_ms
            else:
                self.above_threshold_ms = 0.0
        elif self.state == "speech_active":
            if is_loud:
                self.speech_ms += self.block_ms
                self.trailing_silence_ms = 0.0
            else:
                self.trailing_silence_ms += self.block_ms
                if self.trailing_silence_ms >= self.config.trailing_silence_ms:
                    self.state = "utterance_complete"
                    self._finish("end_of_speech")
                    return
        if not self.done.is_set() and self.total_ms >= self.config.max_utterance_ms:
            self._finish("max_duration")

    def _finish(self, reason: str) -> None:
        self.reason = reason
        self.done.set()


def check_microphone(
    sample_rate: int = 44100,
    input_device: str | int | None = "default",
    auto_select: bool = True,
    silent_rms_threshold: float = 0.001,
    channels: int = 1,
    probe_duration: float = 0.5,
) -> str:
    """Probe the configured microphone and return a user-facing report."""

    try:
        report = microphone_test_report(
            input_device=input_device,
            auto_select=auto_select,
            silent_rms_threshold=silent_rms_threshold,
            sample_rate=sample_rate,
            channels=channels,
            probe_duration=probe_duration,
        )
    except Exception as exc:
        raise MicrophoneCheckError(f"Nao consegui abrir o microfone: {exc}") from exc

    return report


def analyze_voice_wav(
    audio_path: Path | str = DEFAULT_DEBUG_AUDIO_PATH,
    silent_rms_threshold: float = 0.001,
) -> VoiceAudioDiagnostics:
    """Return bounded diagnostics for the debug WAV without exposing audio content."""

    path = Path(audio_path)
    with wave.open(str(path), "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        sample_rate = int(wav_file.getframerate())
        frame_count = int(wav_file.getnframes())
        frames = wav_file.readframes(frame_count)

    duration = float(frame_count) / float(sample_rate) if sample_rate > 0 else 0.0
    samples = _decode_pcm_samples(frames, sample_width)
    if not samples:
        rms = 0.0
        peak = 0.0
        near_silence_percent = 100.0
        clipping_detected = False
    else:
        max_value = float(_pcm_max_value(sample_width))
        normalized = [float(sample) / max_value for sample in samples]
        rms = (sum(value * value for value in normalized) / len(normalized)) ** 0.5
        peak = max(abs(value) for value in normalized)
        silence_limit = max(float(silent_rms_threshold) * 2.0, 1.0 / max_value)
        near_silent = sum(1 for value in normalized if abs(value) <= silence_limit)
        near_silence_percent = (near_silent / len(normalized)) * 100.0
        clipping_detected = any(abs(value) >= 0.999 for value in normalized)

    return VoiceAudioDiagnostics(
        path=str(path),
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frame_count=frame_count,
        duration_seconds=duration,
        rms=float(rms),
        peak=float(peak),
        near_silence_percent=float(near_silence_percent),
        clipping_detected=bool(clipping_detected),
        level=classify_rms_level(float(rms), silent_rms_threshold),
    )


def format_voice_audio_diagnostics(diagnostics: VoiceAudioDiagnostics) -> str:
    """Format WAV diagnostics for developer/manual troubleshooting."""

    clipping = "sim" if diagnostics.clipping_detected else "nao"
    return "\n".join(
        [
            "Diagnostico do ultimo audio:",
            f"- ficheiro: {diagnostics.path}",
            f"- sample rate: {diagnostics.sample_rate} Hz",
            f"- canais: {diagnostics.channels}",
            f"- duracao: {diagnostics.duration_seconds:.2f}s",
            f"- RMS: {diagnostics.rms:.5f}",
            f"- pico: {diagnostics.peak:.5f}",
            f"- quase silencio: {diagnostics.near_silence_percent:.1f}%",
            f"- clipping: {clipping}",
            f"- estado: {diagnostics.level}",
        ]
    )


def play_last_voice_input(debug_audio_path: Path | str = DEFAULT_DEBUG_AUDIO_PATH) -> str:
    """Open the last captured voice WAV with the operating system player."""

    audio_path = Path(debug_audio_path)
    if not audio_path.exists():
        return "Ainda nao existe audio gravado para reproduzir."
    try:
        import os

        os.startfile(str(audio_path.resolve()))  # type: ignore[attr-defined]
    except Exception as exc:
        return f"Nao consegui reproduzir o ultimo audio: {exc}"
    return f"A reproduzir o ultimo audio gravado: {audio_path}"


class VoiceTranscriber:
    """Records microphone audio and transcribes it with a local Whisper model."""

    def __init__(
        self,
        model_name: str = "base",
        language: str = "pt",
        sample_rate: int = 44100,
        input_device: str | int | None = "default",
        auto_select_input: bool = True,
        silent_rms_threshold: float = 0.001,
        channels: int = 1,
        probe_duration: float = 0.5,
        min_record_seconds: float = 2.0,
        preroll_ms: int = 500,
        ready_delay_ms: int = 200,
        debug_audio_path: Path | str = DEFAULT_DEBUG_AUDIO_PATH,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT,
        input_language: str = "auto",
        vad_config: VadConfig | None = None,
        barge_in_vad_config: VadConfig | None = None,
        standby_vad_config: VadConfig | None = None,
        wake_recognizer: WakeRecognizer | None = None,
    ) -> None:
        self.model_name = model_name
        self.language = language
        # Phase 7.5 (section 10): a SEPARATE concept from `language` above --
        # `language` is the legacy field manual MIC's record_and_transcribe()
        # still uses unchanged (forced "pt" by default, via window.py's own
        # `voice_language or "pt"` coalescing). `input_language` only
        # affects the NEW record_and_transcribe_vad() path used by Voice
        # Conversation Mode, and must never be derived from response_language.
        self.input_language = input_language
        self.vad_config = vad_config or VadConfig()
        # Phase 7.6 (section 9/10) / 7.6.2 (section 6) / 7.6.3 (section 9):
        # deliberately more conservative than `vad_config`, so ordinary
        # TTS-speaker bleed into the mic or a cough/keyboard click doesn't
        # get mistaken for a genuine interruption -- but now calibrated
        # against REAL evidence instead of synthetic estimates.
        #
        # History: Phase 7.6's initial 0.035/250ms was a pure guess and
        # never confirmed on real hardware. Phase 7.6.2's 0.02/180ms was a
        # reasoned-but-still-unmeasured halfway adjustment -- a real
        # session then reported a concrete monitor_summary:
        #   peak_rms=0.00665 threshold=0.02 candidate_entered=false
        # i.e. the user's actual speech-over-Anna peak (0.00665) never
        # even reached the 0.02 floor, let alone sustained it -- the
        # threshold, not the duration, was the confirmed blocker (a
        # candidate never even started).
        #
        # Fix: speech_rms_threshold is now a FLOOR (0.0045 -- below the one
        # real observed peak, comfortably above pure silence) combined with
        # an adaptive baseline (3x the recent ambient/TTS-bleed RMS) via
        # adaptive_multiplier, so the effective threshold rises on its own
        # in a louder room/speaker setup instead of requiring a single
        # fixed number to be correct for every real environment (section
        # 10). speech_start_min_ms is left at 180ms unchanged -- the
        # evidence points at the threshold, not the duration, and section
        # 11 explicitly says this may already be reasonable.
        self.barge_in_vad_config = barge_in_vad_config or VadConfig(
            speech_rms_threshold=0.0045,
            speech_start_min_ms=180,
            trailing_silence_ms=self.vad_config.trailing_silence_ms,
            min_speech_ms=self.vad_config.min_speech_ms,
            max_utterance_ms=self.vad_config.max_utterance_ms,
            adaptive_multiplier=3.0,
        )
        # Phase 7.6.1 (section 2/3): real-mic validation showed standby was
        # reusing `vad_config` (min_speech_ms~300ms, tuned for ordinary
        # conversational turns) -- a short wake word like "Echo" is
        # genuinely only ~150-300ms of speech, so it was routinely measured
        # at 200-280ms and rejected as a benign "too short" retry BEFORE
        # Whisper ever ran, silently discarding real wake attempts. This is
        # a SEPARATE, more permissive config specifically for STANDBY's
        # wake/sleep-word capture -- ACTIVE conversation keeps `vad_config`
        # completely unchanged (section 9: never lower it globally to make
        # wake work). Still rejects a bare click/pop: speech_start_min_ms
        # still requires sustained (not instantaneous) loudness, and
        # min_speech_ms still requires more accumulated speech than a
        # single ~40-80ms spike can produce.
        # Phase 7.6.5 (section 9): 650ms of trailing silence was measured as
        # the single largest fixed contributor to perceived wake latency --
        # a short wake word ("Echo") only needs enough of a pause to be
        # confident the utterance is over, not the same margin a full
        # conversational turn needs. Lowered to 450ms (within the
        # 350-500ms suggested range); ACTIVE's own trailing_silence_ms
        # (`self.vad_config`) is completely untouched.
        self.standby_vad_config = standby_vad_config or VadConfig(
            speech_rms_threshold=self.vad_config.speech_rms_threshold,
            speech_start_min_ms=100,
            trailing_silence_ms=450,
            min_speech_ms=150,
            max_utterance_ms=self.vad_config.max_utterance_ms,
        )
        # Phase 7.6.6 (section 3/5): STANDBY's dedicated fast wake/sleep-
        # word recognizer -- a small, cheap Whisper model ("tiny" by
        # default, see assistant/wake_recognizer.py for the benchmark that
        # justifies it), completely separate from self._model (Medium,
        # ACTIVE conversation-grade). Owns its own model cache so it is
        # loaded once and reused, never per utterance (section 5), and
        # never touches/reloads self._model.
        self.wake_recognizer = wake_recognizer or WakeRecognizer()
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.auto_select_input = auto_select_input
        self.silent_rms_threshold = silent_rms_threshold
        self.channels = channels
        self.probe_duration = probe_duration
        self.min_record_seconds = min_record_seconds
        self.preroll_ms = max(0, int(preroll_ms))
        self.ready_delay_ms = max(0, int(ready_delay_ms))
        self.debug_audio_path = Path(debug_audio_path)
        # Debug only: when set (app.py, voice.debug_save_captures), accepted
        # captures are kept here -- see _save_accepted_capture().
        self.capture_dir: Path | None = None
        self.initial_prompt = initial_prompt
        self._model: Any | None = None
        self.last_audio_duration_seconds = 0.0
        self.last_audio_rms = 0.0
        self.last_audio_peak = 0.0
        self.last_audio_near_silence_percent = 0.0
        self.last_audio_clipping_detected = False
        self.last_audio_diagnostics: VoiceAudioDiagnostics | None = None
        self.last_start_diagnostics = VoiceStartDiagnostics()
        self.last_device_index: int | None = None
        self.last_device_name = ""
        # Phase 7.5: structured mirror of the bounded [VAD] "speech_ended"
        # log line -- set by record_until_speech_complete() (even when it
        # goes on to raise) so callers/tests can introspect the VAD outcome
        # without parsing stdout.
        self.last_vad_outcome: dict[str, object] | None = None

    def record_until_stopped(
        self,
        stop_event: threading.Event,
        status_callback: Callable[[str], None] | None = None,
    ) -> Path:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise VoiceInputError(
                "Voice Input precisa das dependencias sounddevice e numpy instaladas."
            ) from exc

        frames: list[Any] = []
        lock = threading.Lock()
        recording_active = threading.Event()
        blocksize = max(1, int(self.sample_rate * 0.04))
        preroll_blocks = max(1, int((self.sample_rate * (self.preroll_ms / 1000)) / blocksize) + 2)
        preroll_frames = deque(maxlen=preroll_blocks)
        started_at = time.perf_counter()
        stream_opened_at: float | None = None
        first_frame_at: float | None = None
        listening_status_at: float | None = None
        selected_device = resolve_input_device(self.input_device)
        selected_device_name = _input_device_name(selected_device)
        self.last_device_index = selected_device
        self.last_device_name = selected_device_name

        def callback(indata, frame_count, time_info, status) -> None:
            nonlocal first_frame_at
            if status:
                return
            if first_frame_at is None:
                first_frame_at = time.perf_counter()
            chunk = indata.copy()
            with lock:
                if recording_active.is_set():
                    frames.append(chunk)
                else:
                    preroll_frames.append(chunk)

        try:
            if status_callback is not None:
                status_callback("Preparar...")
            last_stream_error: Exception | None = None
            for candidate_device in _input_device_candidates(selected_device, self.auto_select_input):
                try:
                    with self._open_input_stream(sd, candidate_device, callback, blocksize) as stream:
                        stream_opened_at = time.perf_counter()
                        selected_device = getattr(stream, "device", candidate_device)
                        selected_device_name = _input_device_name(selected_device)
                        self.last_device_index = selected_device
                        self.last_device_name = selected_device_name
                        self._wait_for_ready_delay(stop_event)
                        with lock:
                            retained_preroll = list(preroll_frames)
                            frames.extend(retained_preroll)
                            preroll_frames.clear()
                            recording_active.set()
                        if status_callback is not None:
                            listening_status_at = time.perf_counter()
                            status_callback("A ouvir...")
                        stop_event.wait()
                    break
                except Exception as exc:
                    last_stream_error = exc
                    if not self.auto_select_input:
                        raise
            else:
                if last_stream_error is not None:
                    raise last_stream_error
        except Exception as exc:
            raise VoiceInputError(f"Nao consegui gravar audio do microfone: {exc}") from exc

        with lock:
            captured_frames = list(frames)

        if not captured_frames:
            raise VoiceInputError("Nao foi captado audio suficiente.")

        audio = np.concatenate(captured_frames, axis=0)
        self.last_audio_duration_seconds = float(len(audio)) / float(self.sample_rate)
        self.last_audio_rms = _rms_int16_audio(audio)
        self.last_start_diagnostics = VoiceStartDiagnostics(
            mic_click_to_stream_ms=_elapsed_ms(started_at, stream_opened_at),
            stream_to_first_frame_ms=_elapsed_ms(stream_opened_at, first_frame_at),
            first_frame_to_listening_status_ms=_elapsed_ms(first_frame_at, listening_status_at),
            preroll_frames=len(retained_preroll) if "retained_preroll" in locals() else 0,
            preroll_duration_seconds=_frames_duration(retained_preroll, self.sample_rate)
            if "retained_preroll" in locals()
            else 0.0,
            selected_device_index=selected_device,
            selected_device_name=selected_device_name,
        )
        self._log_start_diagnostics()
        temp_file = tempfile.NamedTemporaryFile(prefix="assistenteia_voice_", suffix=".wav", delete=False)
        temp_file.close()
        output = Path(temp_file.name)
        self._write_wav(output, audio)
        self.debug_audio_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_wav(self.debug_audio_path, audio)
        self.last_audio_diagnostics = analyze_voice_wav(
            self.debug_audio_path,
            silent_rms_threshold=self.silent_rms_threshold,
        )
        self.last_audio_peak = self.last_audio_diagnostics.peak
        self.last_audio_near_silence_percent = self.last_audio_diagnostics.near_silence_percent
        self.last_audio_clipping_detected = self.last_audio_diagnostics.clipping_detected
        if self.last_audio_duration_seconds < self.min_record_seconds:
            output.unlink(missing_ok=True)
            raise VoiceInputError(
                "A gravacao foi demasiado curta. "
                f"Gravaste {self.last_audio_duration_seconds:.2f}s; "
                f"o minimo configurado e {self.min_record_seconds:.2f}s."
            )
        return output

    def record_until_speech_complete(
        self,
        stop_event: threading.Event,
        status_callback: Callable[[str], None] | None = None,
        *,
        vad_config: VadConfig | None = None,
        on_speech_confirmed: Callable[[], None] | None = None,
        require_confirmation_before_recording: bool = False,
    ) -> Path:
        """Phase 7.5 (section 1/3): Voice Conversation Mode's automatic
        end-of-speech capture -- records until real trailing silence follows
        genuine speech (or the safety ceiling is hit), instead of blocking
        until an external stop_event. The user never has to press STOP to
        end a turn. Manual MIC keeps using record_until_stopped, completely
        unchanged (section 13) -- see controller.py's EchoVoiceWorker's
        mode flag, which picks between the two.

        Phase 7.6 additions for barge-in (section 8/9/11): `on_speech_confirmed`
        is invoked exactly once, the instant the tracker first confirms real
        speech (crosses speech_start_min_ms) -- the controller uses this to
        stop TTS playback immediately, while THIS SAME call keeps running
        and returns the full utterance once trailing silence completes
        ("start/continue normal VAD capture", not a separate hand-off).
        `require_confirmation_before_recording=True` (barge-in only) keeps
        all audio in the bounded preroll ring -- never committed to the
        eventual WAV -- until that same confirmation moment, so Echo's own
        TTS output bleeding into the mic while only monitoring (before any
        real user speech) never ends up in what gets sent to Whisper."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise VoiceInputError(
                "Voice Input precisa das dependencias sounddevice e numpy instaladas."
            ) from exc

        config = vad_config or self.vad_config
        frames: list[Any] = []
        lock = threading.Lock()
        recording_active = threading.Event()
        blocksize = max(1, int(self.sample_rate * 0.04))
        preroll_blocks = max(1, int((self.sample_rate * (self.preroll_ms / 1000)) / blocksize) + 2)
        preroll_frames: deque = deque(maxlen=preroll_blocks)
        tracker = _SpeechActivityTracker(config, blocksize, self.sample_rate)
        selected_device = resolve_input_device(self.input_device)
        selected_device_name = _input_device_name(selected_device)
        self.last_device_index = selected_device
        self.last_device_name = selected_device_name
        speech_started_logged = False

        def note_speech_confirmed(rms: float) -> None:
            nonlocal speech_started_logged
            if speech_started_logged:
                return
            speech_started_logged = True
            _log_vad("speech_started", rms=f"{rms:.5f}", elapsed_ms=int(tracker.total_ms))
            if on_speech_confirmed is not None:
                try:
                    on_speech_confirmed()
                except Exception:
                    pass  # never let a caller's callback crash the audio thread

        def callback(indata, frame_count, time_info, status) -> None:
            if status:
                return
            chunk = indata.copy()
            with lock:
                if recording_active.is_set():
                    frames.append(chunk)
                    rms = _rms_int16_audio(chunk)
                    was_waiting = tracker.state == "waiting_for_speech"
                    tracker.update(rms)
                    if was_waiting and tracker.state != "waiting_for_speech":
                        note_speech_confirmed(rms)
                    return
                preroll_frames.append(chunk)
                if not require_confirmation_before_recording:
                    return
                # Barge-in mode: keep running the tracker against the
                # bounded preroll ring WITHOUT committing anything to the
                # eventual WAV until real speech is confirmed.
                rms = _rms_int16_audio(chunk)
                was_waiting = tracker.state == "waiting_for_speech"
                tracker.update(rms)
                if was_waiting and tracker.state != "waiting_for_speech":
                    frames.extend(list(preroll_frames))
                    preroll_frames.clear()
                    recording_active.set()
                    note_speech_confirmed(rms)

        try:
            if status_callback is not None:
                status_callback("Preparar...")
            last_stream_error: Exception | None = None
            for candidate_device in _input_device_candidates(selected_device, self.auto_select_input):
                try:
                    with self._open_input_stream(sd, candidate_device, callback, blocksize) as stream:
                        selected_device = getattr(stream, "device", candidate_device)
                        selected_device_name = _input_device_name(selected_device)
                        self.last_device_index = selected_device
                        self.last_device_name = selected_device_name
                        self._wait_for_ready_delay(stop_event)
                        if not require_confirmation_before_recording:
                            with lock:
                                frames.extend(list(preroll_frames))
                                preroll_frames.clear()
                                recording_active.set()
                        if status_callback is not None:
                            status_callback("A ouvir...")
                        while not stop_event.is_set() and not tracker.done.is_set():
                            time.sleep(0.02)
                    break
                except Exception as exc:
                    last_stream_error = exc
                    if not self.auto_select_input:
                        raise
            else:
                if last_stream_error is not None:
                    raise last_stream_error
        except Exception as exc:
            raise VoiceInputError(f"Nao consegui gravar audio do microfone: {exc}") from exc

        reason = tracker.reason if tracker.done.is_set() else "manual_stop"
        capture_ended_at = time.perf_counter()
        with lock:
            captured_frames = list(frames)

        _log_vad(
            "speech_ended",
            speech_ms=int(tracker.speech_ms),
            trailing_silence_ms=int(tracker.trailing_silence_ms),
            total_capture_ms=int(tracker.total_ms),
            reason=reason,
        )
        self.last_vad_outcome = {
            "reason": reason,
            "speech_ms": tracker.speech_ms,
            "trailing_silence_ms": tracker.trailing_silence_ms,
            "total_capture_ms": tracker.total_ms,
            # Phase 7.6.2 (section 5): aggregate diagnostics -- the
            # controller reads these to emit one bounded [BARGE_IN]
            # event=monitor_summary line per barge-in capture, letting a
            # real "detector_started" with no further events be diagnosed
            # (was the mic ever loud enough at all? how close to the
            # sustained-duration requirement did it get?).
            "peak_rms": tracker.peak_rms,
            "max_sustained_ms": tracker.max_above_threshold_ms,
            "candidate_entered": tracker.candidate_entered,
            "threshold": config.speech_rms_threshold,
            "required_ms": config.speech_start_min_ms,
            "confirmed": tracker.speech_ms > 0,
            # Phase 7.6.3 (section 10/14): only meaningful when
            # adaptive_multiplier is set (barge-in) -- baseline_rms is the
            # rolling ambient/TTS-bleed estimate, effective_threshold is
            # what was ACTUALLY compared against (may be higher than the
            # configured floor once the baseline rose above it).
            "baseline_rms": tracker.baseline_rms,
            "effective_threshold": tracker.last_effective_threshold,
            # Diagnostics only ([VOICE_TURN_TIMING]): perf_counter() when
            # the capture stopped (end of speech + trailing silence).
            "capture_ended_at": capture_ended_at,
        }

        if not captured_frames:
            raise VoiceInputError("Nao foi captado audio suficiente.")

        audio = np.concatenate(captured_frames, axis=0)
        self.last_audio_duration_seconds = float(len(audio)) / float(self.sample_rate)
        self.last_audio_rms = _rms_int16_audio(audio)

        if tracker.speech_ms <= 0:
            # Section 8: no real speech was ever detected -- never write a
            # WAV or invoke Whisper for pure silence.
            raise VoiceNoSpeechDetected("Nao detetei fala.")
        if tracker.speech_ms < config.min_speech_ms:
            # Section 6: a brief noise burst crossed the loudness threshold
            # but never amounted to a real utterance -- benign, same as
            # manual MIC's "recording too short", never invokes Whisper.
            raise VoiceInputError(
                "A fala detetada foi demasiado curta. "
                f"Fala detetada: {tracker.speech_ms:.0f}ms; o minimo e {config.min_speech_ms}ms."
            )

        temp_file = tempfile.NamedTemporaryFile(prefix="assistenteia_voice_", suffix=".wav", delete=False)
        temp_file.close()
        output = Path(temp_file.name)
        self._write_wav(output, audio)
        self.debug_audio_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_wav(self.debug_audio_path, audio)
        self.last_audio_diagnostics = analyze_voice_wav(
            self.debug_audio_path,
            silent_rms_threshold=self.silent_rms_threshold,
        )
        self.last_audio_peak = self.last_audio_diagnostics.peak
        self.last_audio_near_silence_percent = self.last_audio_diagnostics.near_silence_percent
        self.last_audio_clipping_detected = self.last_audio_diagnostics.clipping_detected
        return output

    def record_and_transcribe_vad(
        self,
        stop_event: threading.Event,
        status_callback: Callable[[str], None] | None = None,
        *,
        input_language: str | None = None,
        vad_config: VadConfig | None = None,
        on_speech_confirmed: Callable[[], None] | None = None,
        require_confirmation_before_recording: bool = False,
    ) -> str:
        """Phase 7.5/7.6: Voice Conversation Mode's (and barge-in's)
        end-to-end automatic-turn capture -- record_until_speech_complete()
        followed by transcription with the given (or this instance's
        default) input_language. Whisper only ever runs here on an
        already-complete utterance (section 2), never while TTS is still
        playing (section 9).

        Phase 7.6.5 (section 8): records the Whisper wall-clock time into
        last_vad_outcome["stt_ms"] -- alongside the speech_ms/
        trailing_silence_ms record_until_speech_complete already populates
        there -- so callers (the controller's wake-timing diagnostics) can
        report a real speech_start_to_end_ms/end_of_speech_to_stt_start_ms/
        stt_ms breakdown instead of guessing where wake latency goes.
        Recorded in a finally so a rejected/no-speech STT result (an
        ignored wake attempt) still gets timed."""
        audio_path = self.record_until_speech_complete(
            stop_event,
            status_callback=status_callback,
            vad_config=vad_config,
            on_speech_confirmed=on_speech_confirmed,
            require_confirmation_before_recording=require_confirmation_before_recording,
        )
        stt_started_at = time.perf_counter()
        if self.last_vad_outcome is not None:
            self.last_vad_outcome["stt_started_at"] = stt_started_at
        try:
            text = self.transcribe(
                audio_path, input_language=input_language if input_language is not None else self.input_language
            )
        finally:
            if self.last_vad_outcome is not None:
                self.last_vad_outcome["stt_ms"] = (time.perf_counter() - stt_started_at) * 1000.0
        self._save_accepted_capture(text)
        return text

    def _save_accepted_capture(self, text: str) -> None:
        # Debug only (voice.debug_save_captures, off by default): keeps
        # each ACCEPTED capture (transcribe() raises for a rejected one) as
        # <stamp>.wav plus its transcript in <stamp>.txt, for
        # scripts/benchmark_stt.py. Never affects the turn itself.
        if self.capture_dir is None or not text:
            return
        try:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
            shutil.copyfile(self.debug_audio_path, self.capture_dir / f"{stamp}.wav")
            (self.capture_dir / f"{stamp}.txt").write_text(text + "\n", encoding="utf-8")
        except OSError:
            pass

    def record_and_transcribe_wake(
        self,
        stop_event: threading.Event,
        status_callback: Callable[[str], None] | None = None,
        *,
        on_speech_confirmed: Callable[[], None] | None = None,
    ) -> str:
        """Phase 7.6.6: STANDBY's dedicated fast wake/sleep-word path
        (section 3's "VAD -> WakeRecognizer -> deterministic wake parser").
        Reuses the EXACT SAME VAD capture as record_and_transcribe_vad
        (standby_vad_config) -- only the recognizer differs.

        self.wake_recognizer (a small, cheap model -- "tiny" by default,
        NOT self._model/Medium) transcribes first. If its transcript
        already matches a wake command, that result is trusted immediately
        -- no Medium call at all (section 9: "do not always run both", no
        duplicate STT work). Phase 7.6.7: self.transcribe() (self._model,
        the SAME cached Medium instance ACTIVE conversation uses) is now
        invoked at most ONCE and only when the fast transcript is a
        plausible near-miss of a wake word (assess_fast_wake_transcript ->
        "near_wake"); an empty or clearly unrelated fast result returns ""
        so the controller simply rearms STANDBY. Never invokes
        an LLM and never treats the fast transcript as sufficient on its
        own without going through the SAME deterministic wake/sleep parser
        the controller uses (section 10) -- this method makes no wake/
        sleep decision itself, it only decides which transcript to trust."""
        audio_path = self.record_until_speech_complete(
            stop_event,
            status_callback=status_callback,
            vad_config=self.standby_vad_config,
            on_speech_confirmed=on_speech_confirmed,
        )
        outcome = self.last_vad_outcome
        audio_rms = self.last_audio_diagnostics.rms if self.last_audio_diagnostics else None
        fast_result = self.wake_recognizer.recognize(audio_path, input_language=self.input_language)
        fast_assessment = assess_fast_wake_transcript(fast_result.transcript)
        if outcome is not None:
            outcome["recognizer"] = "wake_fast"
            outcome["wake_model"] = fast_result.model
            outcome["wake_model_load_ms"] = fast_result.model_load_ms
            outcome["stt_ms"] = fast_result.inference_ms
            outcome["fallback_to_medium"] = False
            outcome["fast_assessment"] = fast_assessment

        # Phase 7.6.7 (section 3/4): Medium only for a plausible-but-
        # unconfirmed wake candidate. Empty/unrelated fast results rearm
        # immediately -- a real session showed tiny-miss -> Medium costing
        # ~9s only to end in wake_ignored anyway. Also never start the slow
        # pass once the capture was cancelled (STANDBY exited / app
        # shutdown): its result would be discarded as stale regardless,
        # and an uninterruptible Medium inference is exactly what would
        # otherwise hold up a clean shutdown.
        if fast_assessment == "wake":
            fallback_reason = "none"
        elif stop_event.is_set():
            fallback_reason = "cancelled"
        elif fast_assessment == "near_wake":
            fallback_reason = "near_wake"
        else:
            fallback_reason = f"skipped_{fast_assessment}"
        _log_wake_recognizer(
            recognizer=fast_result.model,
            transcript=fast_result.transcript,
            assessment=fast_assessment,
            no_speech_prob=fast_result.no_speech_prob,
            avg_logprob=fast_result.avg_logprob,
            audio_rms=audio_rms,
            inference_ms=fast_result.inference_ms,
            fallback=fallback_reason,
        )
        if fast_assessment != "wake":
            self._keep_wake_miss_audio()
        if fallback_reason != "near_wake":
            audio_path.unlink(missing_ok=True)
            return fast_result.transcript if fast_assessment == "wake" else ""

        medium_cold_load = self._model is None
        fallback_started_at = time.perf_counter()
        try:
            medium_text = self.transcribe(audio_path, input_language=self.input_language)
        except (VoiceNoSpeechDetected, VoiceInputError):
            medium_text = ""
        fallback_ms = (time.perf_counter() - fallback_started_at) * 1000.0
        if outcome is not None:
            outcome["recognizer"] = "medium_fallback"
            outcome["fallback_to_medium"] = True
            outcome["medium_cold_load"] = medium_cold_load
            outcome["stt_ms"] = fast_result.inference_ms + fallback_ms
        _log_wake_recognizer(
            recognizer=self.model_name,
            transcript=medium_text,
            assessment="wake" if classify_voice_wake_intent(medium_text) == "wake" else "none",
            audio_rms=audio_rms,
            inference_ms=fallback_ms,
            cold_load=medium_cold_load,
        )
        return medium_text

    def _keep_wake_miss_audio(self) -> None:
        """Phase 7.6.7 (section 5): the debug WAV is overwritten by the very
        next capture, which is how the real failed wake's audio was lost
        before anyone could replay it. Keep a copy of the most recent
        STANDBY capture the fast recognizer did NOT read as a wake, so a
        missed "Echo" can be re-run offline before any alias is added.
        Local-only, one file, overwritten each time; never fatal."""
        try:
            if self.debug_audio_path.exists():
                shutil.copyfile(self.debug_audio_path, self.debug_audio_path.with_name("last_wake_miss.wav"))
        except OSError:
            pass

    def _wait_for_ready_delay(self, stop_event: threading.Event) -> None:
        deadline = time.monotonic() + (self.ready_delay_ms / 1000)
        while time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(0.01)

    def _open_input_stream(self, sd: Any, selected_device: int | None, callback: Callable[..., None], blocksize: int):
        return sd.InputStream(
            device=selected_device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=blocksize,
            callback=callback,
        )

    def _log_start_diagnostics(self) -> None:
        diagnostics = self.last_start_diagnostics
        print(
            "[VOICE_START] "
            f"mic_click_to_stream_ms={_format_optional_ms(diagnostics.mic_click_to_stream_ms)} "
            f"stream_to_first_frame_ms={_format_optional_ms(diagnostics.stream_to_first_frame_ms)} "
            f"first_frame_to_listening_status_ms={_format_optional_ms(diagnostics.first_frame_to_listening_status_ms)} "
            f"preroll_frames={diagnostics.preroll_frames} "
            f"preroll_duration_seconds={diagnostics.preroll_duration_seconds:.3f} "
            f"device_index={diagnostics.selected_device_index} "
            f"device_name={diagnostics.selected_device_name}"
        )

    def _log_stt_decision(self, decision: VoiceSttDecision) -> None:
        diagnostics = self.last_audio_diagnostics
        print(
            "[VOICE_STT] "
            f"model={self.model_name} "
            f"audio_rms={_format_optional_float(diagnostics.rms if diagnostics else None)} "
            f"near_silence_ratio={_format_optional_float((diagnostics.near_silence_percent / 100.0) if diagnostics else None)} "
            f"no_speech_probability={_format_optional_float(decision.no_speech_probability)} "
            f"avg_logprob={_format_optional_float(decision.avg_logprob)} "
            f"compression_ratio={_format_optional_float(decision.compression_ratio)} "
            f"accepted={str(decision.accepted).lower()} "
            f"rejection_reason={decision.rejection_reason or 'none'}"
        )

    def _write_wav(self, output: Path, audio: Any) -> None:
        with wave.open(str(output), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())

    def transcribe(self, audio_path: Path, *, delete_after: bool = True, input_language: str | None = None) -> str:
        """`input_language=None` (the default) is the LEGACY behavior every
        existing caller (manual MIC's record_and_transcribe(), direct
        callers/tests) still gets unchanged: force self.language ("pt" by
        default) with the Portuguese initial_prompt. Passing "auto"/"pt"/
        "en" explicitly (as record_and_transcribe_vad() does) switches to
        the Phase 7.5 input-language resolution (section 10/11), entirely
        independent of response_language."""
        try:
            import whisper
        except ImportError as exc:
            raise VoiceInputError(
                "Voice Input precisa do Whisper local instalado: openai-whisper."
            ) from exc

        if input_language is None:
            whisper_language: str | None = self.language or "pt"
            prompt = self.initial_prompt
            logged_input_language = whisper_language
        else:
            whisper_language, prompt = _resolve_input_language(input_language, self.initial_prompt)
            logged_input_language = str(input_language or "auto").strip().lower()

        try:
            if self._model is None:
                self._model = whisper.load_model(self.model_name)
            result = self._model.transcribe(
                str(audio_path),
                language=whisper_language,
                initial_prompt=prompt,
                task="transcribe",
                temperature=0.0,
                condition_on_previous_text=False,
                fp16=False,
            )
        except Exception as exc:
            raise VoiceInputError(f"Nao consegui transcrever o audio com Whisper: {exc}") from exc
        finally:
            if delete_after:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception:
                    pass

        text = str(result.get("text", "")).strip()
        decision = decide_stt_acceptance(
            text,
            self.last_audio_diagnostics,
            result,
            silent_rms_threshold=self.silent_rms_threshold,
        )
        self._log_stt_decision(decision)
        _log_stt_language(input_language=logged_input_language, detected_language=str(result.get("language") or ""))
        if not decision.accepted:
            raise VoiceNoSpeechDetected("Nao detetei fala.")
        return text

    def record_and_transcribe(
        self,
        stop_event: threading.Event,
        status_callback: Callable[[str], None] | None = None,
    ) -> str:
        audio_path = self.record_until_stopped(stop_event, status_callback=status_callback)
        text = self.transcribe(audio_path)
        self._save_accepted_capture(text)
        return text

    def transcribe_existing_wav(
        self,
        audio_path: Path | str = DEFAULT_DEBUG_AUDIO_PATH,
        *,
        input_language: str | None = None,
    ) -> str:
        """Transcribe an existing WAV with the same Whisper configuration used by live STT."""

        path = Path(audio_path)
        if not path.exists():
            raise VoiceInputError(f"Nao encontrei o ficheiro de audio: {path}")
        self.last_audio_diagnostics = analyze_voice_wav(
            path,
            silent_rms_threshold=self.silent_rms_threshold,
        )
        self.last_audio_duration_seconds = self.last_audio_diagnostics.duration_seconds
        self.last_audio_rms = self.last_audio_diagnostics.rms
        self.last_audio_peak = self.last_audio_diagnostics.peak
        self.last_audio_near_silence_percent = self.last_audio_diagnostics.near_silence_percent
        self.last_audio_clipping_detected = self.last_audio_diagnostics.clipping_detected
        return self.transcribe(path, delete_after=False, input_language=input_language)


def _resolve_input_language(input_language: str, portuguese_prompt: str) -> tuple[str | None, str]:
    """Phase 7.5 (section 10/11): STT input language is independent of
    response_language -- never derive one from the other, and never call
    this with anything response_language-derived. "auto" passes
    language=None to Whisper (real language auto-detection) with a NEUTRAL
    (empty) initial_prompt -- the existing Portuguese-only prompt would
    bias detection/decoding toward Portuguese for every language, which
    must not happen in auto/en modes."""
    normalized = str(input_language or "auto").strip().lower()
    if normalized == "pt":
        return "pt", portuguese_prompt
    if normalized == "en":
        return "en", ""
    return None, ""


def _log_vad(event: str, **fields: object) -> None:
    # Section 15: bounded, one line per state transition -- never one line
    # per audio frame/block.
    parts = [f"[VAD] event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts))


_WAKE_TRANSCRIPT_LOG_MAX_CHARS = 48


def _bounded_wake_transcript(text: str) -> str:
    """Phase 7.6.7 (section 1): wake utterances are a word or two, so a
    short, single-line, quote-safe excerpt is enough to diagnose a missed
    wake -- and the cap keeps this from ever becoming a general transcript
    log if someone speaks a long sentence near the mic during STANDBY."""
    flat = " ".join(str(text or "").split()).replace('"', "'")
    if len(flat) > _WAKE_TRANSCRIPT_LOG_MAX_CHARS:
        flat = flat[: _WAKE_TRANSCRIPT_LOG_MAX_CHARS - 3] + "..."
    return flat


def _log_wake_recognizer(
    *,
    recognizer: str,
    transcript: str,
    assessment: str,
    audio_rms: float | None,
    inference_ms: float,
    no_speech_prob: float | None = None,
    avg_logprob: float | None = None,
    fallback: str | None = None,
    cold_load: bool | None = None,
) -> None:
    # Phase 7.6.7 (section 1/2): STANDBY-only, one line per recognizer pass.
    # assessment + no_speech_prob/avg_logprob + audio_rms together separate
    # "heard nothing" (empty), "misheard the wake word" (near_wake),
    # "hallucinated something unrelated" (unrelated, usually low
    # avg_logprob) and "the capture itself was poor" (low audio_rms).
    parts = [
        "[VOICE_WAKE_RECOGNIZER]",
        f"recognizer={recognizer}",
        f'transcript="{_bounded_wake_transcript(transcript)}"',
        f"assessment={assessment}",
        f"no_speech_prob={_format_optional_float(no_speech_prob)}",
        f"avg_logprob={_format_optional_float(avg_logprob)}",
        f"audio_rms={_format_optional_float(audio_rms)}",
        f"inference_ms={int(inference_ms)}",
    ]
    if fallback is not None:
        parts.append(f"fallback_reason={fallback}")
    if cold_load is not None:
        parts.append(f"cold_load={str(cold_load).lower()}")
    print(" ".join(parts))


def _log_stt_language(*, input_language: str, detected_language: str) -> None:
    print(f"[VOICE_STT] input_language={input_language} detected_language={detected_language or 'unknown'}")


def check_voice_runtime() -> list[str]:
    """Return missing local dependencies required by Voice Input."""

    missing: list[str] = []
    if find_spec("whisper") is None:
        missing.append("openai-whisper")
    if find_spec("sounddevice") is None:
        missing.append("sounddevice")
    if find_spec("numpy") is None:
        missing.append("numpy")
    if which("ffmpeg") is None:
        missing.append("ffmpeg")
    return missing


def voice_status_report(
    enabled: bool,
    missing_dependencies: list[str] | tuple[str, ...] | None = None,
    microphone_ok: bool = False,
    microphone_message: str = "",
    input_device: str | int | None = "default",
    auto_select_input: bool = True,
    sample_rate: int = 44100,
    channels: int = 1,
    model_name: str = "base",
    language: str = "pt",
    silent_rms_threshold: float = 0.001,
    probe_duration: float = 0.5,
) -> str:
    missing = list(missing_dependencies or [])
    chosen = None
    if enabled and "sounddevice" not in missing and "numpy" not in missing:
        try:
            chosen = choose_best_input_device(
                input_device=input_device,
                auto_select=auto_select_input,
                silent_rms_threshold=silent_rms_threshold,
                sample_rate=sample_rate,
                channels=channels,
                probe_duration=probe_duration,
            )
        except Exception:
            chosen = None

    lines = ["Estado da voz:"]
    lines.append(f"- configuracao: {'ativa' if enabled else 'desativada'}")
    lines.append(f"- voice.enabled: {enabled}")
    lines.append(f"- input_device: {input_device}")
    lines.append(f"- auto_select_input: {auto_select_input}")
    lines.append(f"- modelo whisper: {model_name}")
    lines.append(f"- idioma configurado: {language or 'pt'}")
    lines.append(f"- sample rate: {sample_rate} Hz")
    lines.append(f"- canais: {channels}")
    lines.append(f"- whisper: {'pronto' if 'openai-whisper' not in missing else 'em falta'}")
    lines.append(f"- ffmpeg: {'pronto' if 'ffmpeg' not in missing else 'em falta'}")
    lines.append(f"- sounddevice: {'pronto' if 'sounddevice' not in missing else 'em falta'}")
    lines.append(f"- numpy: {'pronto' if 'numpy' not in missing else 'em falta'}")
    if chosen is not None:
        lines.append(f"- microfone escolhido: {chosen.device_name}")
        lines.append(f"- indice escolhido: {chosen.device_index if chosen.device_index is not None else 'default'}")
        lines.append(f"- nivel RMS atual: {chosen.rms:.5f}")
        lines.append(f"- estado do sinal: {classify_rms_level(chosen.rms, silent_rms_threshold)}")
    if microphone_message:
        lines.append(f"- microfone: {microphone_message}")
    else:
        lines.append(f"- microfone: {'pronto' if microphone_ok else 'nao testado'}")
    if not enabled:
        lines.append("A voz esta desligada em config/settings.json.")
    elif missing:
        lines.append("A voz ainda nao esta pronta porque faltam dependencias locais.")
    elif not microphone_ok:
        lines.append("As dependencias estao prontas; falta testar o microfone.")
    else:
        lines.append("A voz esta pronta para uso.")
    return "\n".join(lines)


def decide_stt_acceptance(
    text: str,
    diagnostics: VoiceAudioDiagnostics | None,
    whisper_result: dict[str, Any] | None,
    silent_rms_threshold: float = 0.001,
) -> VoiceSttDecision:
    """Conservatively decide whether a Whisper transcription is safe to insert."""

    clean_text = " ".join(str(text or "").split())
    no_speech_probability, avg_logprob, compression_ratio = _whisper_confidence_signals(whisper_result or {})

    if diagnostics is not None and _audio_is_effective_silence(diagnostics, silent_rms_threshold):
        return VoiceSttDecision(
            accepted=False,
            rejection_reason="silence",
            no_speech_probability=no_speech_probability,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
        )
    if not clean_text:
        return VoiceSttDecision(
            accepted=False,
            rejection_reason="empty_transcription",
            no_speech_probability=no_speech_probability,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
        )
    if no_speech_probability is not None and no_speech_probability >= 0.75:
        return VoiceSttDecision(
            accepted=False,
            rejection_reason="silence",
            no_speech_probability=no_speech_probability,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
        )
    if avg_logprob is not None and avg_logprob <= -1.25:
        return VoiceSttDecision(
            accepted=False,
            rejection_reason="low_confidence",
            no_speech_probability=no_speech_probability,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
        )
    if compression_ratio is not None and compression_ratio >= 3.2:
        return VoiceSttDecision(
            accepted=False,
            rejection_reason="excessive_repetition",
            no_speech_probability=no_speech_probability,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
        )
    if _has_excessive_repetition(clean_text):
        return VoiceSttDecision(
            accepted=False,
            rejection_reason="excessive_repetition",
            no_speech_probability=no_speech_probability,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
        )
    return VoiceSttDecision(
        accepted=True,
        no_speech_probability=no_speech_probability,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
    )


def _rms_int16_audio(audio: Any) -> float:
    try:
        import numpy as np

        arr = np.asarray(audio, dtype=float)
        if arr.size == 0:
            return 0.0
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        arr = arr / 32768.0
        return float(np.sqrt(np.mean(arr**2)))
    except Exception:
        return 0.0


def _audio_is_effective_silence(diagnostics: VoiceAudioDiagnostics, silent_rms_threshold: float) -> bool:
    if diagnostics.duration_seconds <= 0:
        return True
    rms_limit = max(float(silent_rms_threshold) * 2.5, 0.0008)
    peak_limit = max(float(silent_rms_threshold) * 12.0, 0.008)
    return (
        diagnostics.rms < rms_limit
        and diagnostics.peak < peak_limit
        and diagnostics.near_silence_percent >= 98.0
    )


def _whisper_confidence_signals(result: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return (
            _optional_float(result.get("no_speech_prob")),
            _optional_float(result.get("avg_logprob")),
            _optional_float(result.get("compression_ratio")),
        )

    no_speech_values: list[float] = []
    avg_logprob_values: list[float] = []
    compression_values: list[float] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        no_speech = _optional_float(segment.get("no_speech_prob"))
        avg_logprob = _optional_float(segment.get("avg_logprob"))
        compression = _optional_float(segment.get("compression_ratio"))
        if no_speech is not None:
            no_speech_values.append(no_speech)
        if avg_logprob is not None:
            avg_logprob_values.append(avg_logprob)
        if compression is not None:
            compression_values.append(compression)

    no_speech_probability = max(no_speech_values) if no_speech_values else None
    avg_logprob = (
        sum(avg_logprob_values) / len(avg_logprob_values)
        if avg_logprob_values
        else None
    )
    compression_ratio = max(compression_values) if compression_values else None
    return no_speech_probability, avg_logprob, compression_ratio


def _has_excessive_repetition(text: str) -> bool:
    tokens = [_normalize_token(token) for token in text.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < 6:
        return False
    for size in range(1, min(5, len(tokens) // 2) + 1):
        windows = [tuple(tokens[index : index + size]) for index in range(0, len(tokens) - size + 1)]
        for window in set(windows):
            if not window:
                continue
            count = windows.count(window)
            if count >= 4 and count * size >= max(6, int(len(tokens) * 0.65)):
                return True
    unique_ratio = len(set(tokens)) / len(tokens)
    return len(tokens) >= 10 and unique_ratio <= 0.35


def _normalize_token(token: str) -> str:
    return "".join(ch.lower() for ch in token if ch.isalnum())


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _decode_pcm_samples(frames: bytes, sample_width: int) -> list[int]:
    if not frames:
        return []
    if sample_width == 2:
        values = array("h")
        values.frombytes(frames)
        return [int(value) for value in values]
    if sample_width == 1:
        return [int(byte) - 128 for byte in frames]
    if sample_width == 4:
        values = array("i")
        values.frombytes(frames)
        return [int(value) for value in values]
    return []


def _pcm_max_value(sample_width: int) -> int:
    if sample_width == 1:
        return 128
    if sample_width == 4:
        return 2147483648
    return 32768


def _input_device_name(index: int | None) -> str:
    if index is None:
        return "microfone predefinido"
    for device in list_input_devices():
        if int(device.get("index", -1)) == int(index):
            return str(device.get("name", f"microfone {index}"))
    return f"microfone {index}"


def _input_device_candidates(selected_device: int | None, auto_select: bool) -> list[int | None]:
    candidates: list[int | None] = [selected_device]
    if not auto_select:
        return candidates
    seen = {selected_device}
    for device in list_input_devices():
        try:
            index = int(device["index"])
        except Exception:
            continue
        if index in seen:
            continue
        candidates.append(index)
        seen.add(index)
    return candidates


def _elapsed_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, (end - start) * 1000.0)


def _format_optional_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.5f}"


def _frames_duration(frames: list[Any], sample_rate: int) -> float:
    total = 0
    for frame in frames:
        try:
            total += len(frame)
        except Exception:
            pass
    return float(total) / float(sample_rate) if sample_rate > 0 else 0.0
