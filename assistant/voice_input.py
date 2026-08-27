from __future__ import annotations

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
    ) -> None:
        self.model_name = model_name
        self.language = language
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

    def transcribe(self, audio_path: Path, *, delete_after: bool = True) -> str:
        try:
            import whisper
        except ImportError as exc:
            raise VoiceInputError(
                "Voice Input precisa do Whisper local instalado: openai-whisper."
            ) from exc

        try:
            if self._model is None:
                self._model = whisper.load_model(self.model_name)
            result = self._model.transcribe(
                str(audio_path),
                language=self.language or "pt",
                initial_prompt=self.initial_prompt,
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
        if not decision.accepted:
            raise VoiceNoSpeechDetected("Nao detetei fala.")
        return text

    def record_and_transcribe(
        self,
        stop_event: threading.Event,
        status_callback: Callable[[str], None] | None = None,
    ) -> str:
        audio_path = self.record_until_stopped(stop_event, status_callback=status_callback)
        return self.transcribe(audio_path)

    def transcribe_existing_wav(self, audio_path: Path | str = DEFAULT_DEBUG_AUDIO_PATH) -> str:
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
        return self.transcribe(path, delete_after=False)


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
