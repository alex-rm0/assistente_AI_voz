"""R2: the GPU STT process (and its model) only started on the first
transcription: 4756 ms model load + 1346 ms first CUDA inference, against
~160 ms for later ones (11/11 log). warmup() starts the worker and runs one
short silent transcription so the first real turn pays neither.
"""
from __future__ import annotations

import pytest

from assistant.stt_backend import FasterWhisperBackend
from assistant.voice_input import VoiceTranscriber
from tests.test_stt_backend import Process


def _backend(tmp_path, processes, *, device="cuda"):
    python = tmp_path / "python.exe"
    python.touch()

    def launch(command, **kwargs):
        processes.append(Process(device=device))
        return processes[-1]

    return FasterWhisperBackend(python=python, process_factory=launch)


def test_warmup_starts_the_worker_and_runs_one_transcription(tmp_path) -> None:
    processes: list = []
    backend = _backend(tmp_path, processes)
    try:
        backend.warmup()

        assert len(processes) == 1
        assert len(processes[0].requests) == 1  # the silent warm-up clip
        backend.transcribe(tmp_path / "real.wav", language="pt", initial_prompt="")
        assert len(processes) == 1  # the real turn reuses the warm worker
    finally:
        backend.close()


def test_warmup_failure_is_reported_not_swallowed(tmp_path) -> None:
    processes: list = []
    backend = _backend(tmp_path, processes, device="cpu")  # e.g. no CUDA / out of memory
    try:
        with pytest.raises(RuntimeError):
            backend.warmup()
    finally:
        backend.close()


def test_transcriber_prewarm_is_a_noop_with_the_default_stt() -> None:
    transcriber = VoiceTranscriber(model_name="medium", stt_provider="openai-whisper")

    assert transcriber.prewarm_stt() is False
    assert transcriber._stt_backend is None


def test_transcriber_prewarm_warms_the_gpu_backend() -> None:
    transcriber = VoiceTranscriber(model_name="medium", stt_provider="faster-whisper")
    calls: list[str] = []

    class _Backend:
        def warmup(self) -> None:
            calls.append("warmup")

        def close(self) -> None:
            calls.append("close")

    transcriber._stt_backend = _Backend()

    assert transcriber.prewarm_stt() is True
    assert calls == ["warmup"]
