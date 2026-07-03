from __future__ import annotations

import tempfile
import threading
import wave
from importlib.util import find_spec
from pathlib import Path
from shutil import which
from typing import Any


class VoiceInputError(RuntimeError):
    """Raised when local voice input cannot run."""


class MicrophoneCheckError(RuntimeError):
    """Raised when the microphone cannot be opened."""


def check_microphone(sample_rate: int = 16000) -> str:
    """Open the default microphone briefly to confirm it is usable."""

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise MicrophoneCheckError("A biblioteca sounddevice nao esta instalada.") from exc

    try:
        devices = sd.query_devices(kind="input")
    except Exception as exc:
        raise MicrophoneCheckError(f"Nao encontrei um microfone de entrada: {exc}") from exc

    device_name = str(devices.get("name", "microfone predefinido")) if isinstance(devices, dict) else "microfone predefinido"
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16"):
            pass
    except Exception as exc:
        raise MicrophoneCheckError(f"Nao consegui abrir o microfone: {exc}") from exc

    return f"Microfone pronto: {device_name}."


class VoiceTranscriber:
    """Records microphone audio and transcribes it with a local Whisper model."""

    def __init__(
        self,
        model_name: str = "base",
        language: str = "pt",
        sample_rate: int = 16000,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate
        self._model: Any | None = None

    def record_until_stopped(self, stop_event: threading.Event) -> Path:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise VoiceInputError(
                "Voice Input precisa das dependencias sounddevice e numpy instaladas."
            ) from exc

        frames: list[Any] = []

        def callback(indata, frame_count, time_info, status) -> None:
            if status:
                return
            frames.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=callback,
            ):
                stop_event.wait()
        except Exception as exc:
            raise VoiceInputError(f"Nao consegui gravar audio do microfone: {exc}") from exc

        if not frames:
            raise VoiceInputError("Nao foi captado audio suficiente.")

        audio = np.concatenate(frames, axis=0)
        temp_file = tempfile.NamedTemporaryFile(prefix="assistenteia_voice_", suffix=".wav", delete=False)
        temp_file.close()
        output = Path(temp_file.name)
        with wave.open(str(output), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())
        return output

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
            result = self._model.transcribe(str(audio_path), language=self.language, fp16=False)
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

    def record_and_transcribe(self, stop_event: threading.Event) -> str:
        audio_path = self.record_until_stopped(stop_event)
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
) -> str:
    missing = list(missing_dependencies or [])
    lines = ["Estado da voz:"]
    lines.append(f"- configuracao: {'ativa' if enabled else 'desativada'}")
    lines.append(f"- whisper: {'pronto' if 'openai-whisper' not in missing else 'em falta'}")
    lines.append(f"- ffmpeg: {'pronto' if 'ffmpeg' not in missing else 'em falta'}")
    lines.append(f"- sounddevice: {'pronto' if 'sounddevice' not in missing else 'em falta'}")
    lines.append(f"- numpy: {'pronto' if 'numpy' not in missing else 'em falta'}")
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
