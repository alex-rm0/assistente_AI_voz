from __future__ import annotations

import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from shutil import which
from typing import Any

from assistant.audio_device import (
    choose_best_input_device,
    classify_rms_level,
    microphone_test_report,
)


DEFAULT_INITIAL_PROMPT = (
    "Transcricao em portugues de Portugal. "
    "O utilizador chama-se Alexandre e esta a testar o microfone do AssistenteIA."
)
DEFAULT_DEBUG_AUDIO_PATH = Path("data/debug/last_voice_input.wav")


class VoiceInputError(RuntimeError):
    """Raised when local voice input cannot run."""


class MicrophoneCheckError(RuntimeError):
    """Raised when the microphone cannot be opened."""


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
        selected = choose_best_input_device(
            input_device=self.input_device,
            auto_select=self.auto_select_input,
            silent_rms_threshold=self.silent_rms_threshold,
            sample_rate=self.sample_rate,
            channels=self.channels,
            probe_duration=self.probe_duration,
        )
        selected_device = selected.device_index
        self.last_device_index = selected.device_index
        self.last_device_name = selected.device_name

        def callback(indata, frame_count, time_info, status) -> None:
            if status:
                return
            chunk = indata.copy()
            with lock:
                if recording_active.is_set():
                    frames.append(chunk)
                else:
                    preroll_frames.append(chunk)

        try:
            if status_callback is not None:
                status_callback("Preparar...")
            with sd.InputStream(
                device=selected_device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=blocksize,
                callback=callback,
            ):
                self._wait_for_ready_delay(stop_event)
                with lock:
                    frames.extend(list(preroll_frames))
                    preroll_frames.clear()
                    recording_active.set()
                if status_callback is not None:
                    status_callback("A ouvir...")
                stop_event.wait()
        except Exception as exc:
            raise VoiceInputError(f"Nao consegui gravar audio do microfone: {exc}") from exc

        with lock:
            captured_frames = list(frames)

        if not captured_frames:
            raise VoiceInputError("Nao foi captado audio suficiente.")

        audio = np.concatenate(captured_frames, axis=0)
        self.last_audio_duration_seconds = float(len(audio)) / float(self.sample_rate)
        self.last_audio_rms = _rms_int16_audio(audio)
        temp_file = tempfile.NamedTemporaryFile(prefix="assistenteia_voice_", suffix=".wav", delete=False)
        temp_file.close()
        output = Path(temp_file.name)
        self._write_wav(output, audio)
        self.debug_audio_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_wav(self.debug_audio_path, audio)
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

    def _write_wav(self, output: Path, audio: Any) -> None:
        with wave.open(str(output), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())

    def transcribe(self, audio_path: Path) -> str:
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
                fp16=False,
            )
        except Exception as exc:
            raise VoiceInputError(f"Nao consegui transcrever o audio com Whisper: {exc}") from exc
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

        text = str(result.get("text", "")).strip()
        if not text:
            raise VoiceInputError("O Whisper nao devolveu texto.")
        return text

    def record_and_transcribe(
        self,
        stop_event: threading.Event,
        status_callback: Callable[[str], None] | None = None,
    ) -> str:
        audio_path = self.record_until_stopped(stop_event, status_callback=status_callback)
        return self.transcribe(audio_path)


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
