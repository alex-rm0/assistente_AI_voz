import io
import json
import queue
import wave
from pathlib import Path

import pytest

from assistant.stt_backend import FasterWhisperBackend
from assistant.voice_input import VoiceNoSpeechDetected, VoiceTranscriber


class Output:
    def __init__(self):
        self.items = queue.Queue()
    def __iter__(self):
        while (item := self.items.get()) is not None:
            yield json.dumps(item) + "\n"
    def close(self):
        pass


class Process:
    def __init__(self, device="cuda", silent=False):
        self.stdout = Output()
        self.stdout.items.put({"ok": True, "device": device, "model_load_ms": 1})
        self.requests = []
        self.done = False
        self.silent = silent
        owner = self
        class Input:
            def write(self, line):
                owner.requests.append(json.loads(line))
                if not owner.silent:
                    owner.stdout.items.put({"ok": True, "result": {"text": "texto", "segments": []}})
            def flush(self):
                pass
            def close(self):
                owner.done = True
                owner.stdout.items.put(None)
        self.stdin = Input()
    def poll(self):
        return 0 if self.done else None
    def wait(self, timeout=None):
        return 0


def test_worker_reused_and_text_not_put_on_command_line(tmp_path):
    python = tmp_path / "python.exe"
    python.touch()
    processes, commands = [], []
    def launch(command, **kwargs):
        commands.append(command)
        processes.append(Process())
        return processes[-1]
    backend = FasterWhisperBackend(python=python, process_factory=launch)
    try:
        assert backend.transcribe(tmp_path / "one.wav", language="pt", initial_prompt="private prompt")["text"] == "texto"
        backend.transcribe(tmp_path / "two.wav", language=None, initial_prompt="")
        assert len(processes) == 1
        assert len(processes[0].requests) == 2
        assert processes[0].requests[0]["initial_prompt"] == "private prompt"
        assert processes[0].requests[1]["language"] is None
        assert "private prompt" not in str(commands)
    finally:
        backend.close()
    with pytest.raises(RuntimeError, match="encerrado"):
        backend.transcribe(tmp_path / "x.wav", language="pt", initial_prompt="")


def test_missing_runtime_does_not_fallback_to_cpu(tmp_path):
    backend = FasterWhisperBackend(python=tmp_path / "missing.exe")
    try:
        with pytest.raises(RuntimeError, match="Runtime STT GPU"):
            backend.transcribe(tmp_path / "a.wav", language="pt", initial_prompt="")
    finally:
        backend.close()


@pytest.mark.parametrize("device,silent,expected", [("cpu", False, "CUDA"), ("cuda", True, "limite")])
def test_cpu_or_timeout_are_explicit_errors(tmp_path, device, silent, expected):
    python = tmp_path / "python.exe"
    python.touch()
    backend = FasterWhisperBackend(python=python, process_factory=lambda *a, **kw: Process(device, silent),
                                  transcription_timeout=.01)
    try:
        with pytest.raises(RuntimeError, match=expected):
            backend.transcribe(tmp_path / "a.wav", language="pt", initial_prompt="")
    finally:
        backend.close()


class Backend:
    def __init__(self, confidence=-.3):
        self.calls = []
        self.closed = False
        self.confidence = confidence
    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        return {"text": "Abro pixel e minimizo os outros.", "language": "pt",
                "segments": [{"no_speech_prob": 0, "avg_logprob": self.confidence, "compression_ratio": .8}]}
    def close(self):
        self.closed = True


def audio(tmp_path):
    path = tmp_path / "test.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(b"\x10\x10\xf0\xef" * 8000)
    return path


def test_gpu_uses_same_filters_language_and_does_not_rewrite_commands(tmp_path):
    backend = Backend()
    transcriber = VoiceTranscriber(stt_provider="faster-whisper", language="pt")
    transcriber._stt_backend = backend
    path = audio(tmp_path)
    assert transcriber.transcribe_existing_wav(path) == "Abro pixel e minimizo os outros."
    assert path.exists()
    assert backend.calls[0]["language"] == "pt"
    transcriber.close()
    assert backend.closed


def test_gpu_low_confidence_is_rejected_and_audio_preserved(tmp_path):
    transcriber = VoiceTranscriber(stt_provider="faster-whisper")
    transcriber._stt_backend = Backend(confidence=-2)
    path = audio(tmp_path)
    with pytest.raises(VoiceNoSpeechDetected):
        transcriber.transcribe_existing_wav(path)
    assert path.exists()


def test_default_and_opt_in_model_selection(monkeypatch):
    monkeypatch.delenv("ECHO_STT_PROVIDER", raising=False)
    assert VoiceTranscriber(model_name="medium").stt_provider == "openai-whisper"
    monkeypatch.setenv("ECHO_STT_PROVIDER", "faster-whisper")
    transcriber = VoiceTranscriber(model_name="medium")
    assert transcriber.model_name == "large-v3-turbo"
    assert transcriber._stt_backend is None
    assert VoiceTranscriber(stt_provider="openai-whisper", model_name="medium").model_name == "medium"
