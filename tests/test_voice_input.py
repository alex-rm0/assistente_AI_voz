from __future__ import annotations

import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

from assistant.voice_input import (
    DEFAULT_INITIAL_PROMPT,
    VoiceAudioDiagnostics,
    VoiceNoSpeechDetected,
    VoiceTranscriber,
    analyze_voice_wav,
    decide_stt_acceptance,
    format_voice_audio_diagnostics,
    voice_status_report,
)


def test_voice_status_report_ready() -> None:
    result = voice_status_report(
        enabled=True,
        missing_dependencies=[],
        microphone_ok=True,
        microphone_message="Microfone pronto.",
        input_device="default",
        auto_select_input=True,
        model_name="base",
        language="pt",
    )

    assert "whisper: pronto" in result
    assert "ffmpeg: pronto" in result
    assert "voice.enabled: True" in result
    assert "input_device: default" in result
    assert "auto_select_input: True" in result
    assert "modelo whisper: base" in result
    assert "idioma configurado: pt" in result
    assert "A voz esta pronta para uso." in result


def test_voice_status_report_missing_dependencies() -> None:
    result = voice_status_report(
        enabled=True,
        missing_dependencies=["openai-whisper", "ffmpeg"],
    )

    assert "whisper: em falta" in result
    assert "ffmpeg: em falta" in result
    assert "faltam dependencias locais" in result


def test_analyze_voice_wav_reports_bounded_audio_diagnostics(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_test_wav(audio_path, samples=[0, 2000, -2000, 32767, -32768], sample_rate=16000)

    diagnostics = analyze_voice_wav(audio_path, silent_rms_threshold=0.001)
    report = format_voice_audio_diagnostics(diagnostics)

    assert diagnostics.sample_rate == 16000
    assert diagnostics.channels == 1
    assert diagnostics.sample_width_bytes == 2
    assert diagnostics.duration_seconds > 0
    assert diagnostics.rms > 0
    assert diagnostics.peak >= 0.999
    assert diagnostics.clipping_detected is True
    assert "sample rate: 16000 Hz" in report
    assert "clipping: sim" in report


def test_transcribe_passes_conservative_portuguese_whisper_options(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_test_wav(audio_path, samples=[0, 1000, -1000], sample_rate=44100)
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = VoiceTranscriber(model_name="base", language="pt", initial_prompt=DEFAULT_INITIAL_PROMPT)
    text = transcriber.transcribe(audio_path)

    assert text == "Vou a Estocolmo de 10 a 15 de outubro."
    assert fake_whisper.loaded_models == ["base"]
    assert fake_whisper.model.calls[0]["language"] == "pt"
    assert fake_whisper.model.calls[0]["initial_prompt"] == DEFAULT_INITIAL_PROMPT
    assert fake_whisper.model.calls[0]["task"] == "transcribe"
    assert fake_whisper.model.calls[0]["temperature"] == 0.0
    assert fake_whisper.model.calls[0]["condition_on_previous_text"] is False
    assert fake_whisper.model.calls[0]["fp16"] is False
    assert not audio_path.exists()


def test_transcribe_existing_wav_preserves_debug_file_and_supports_stronger_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "last_voice_input.wav"
    _write_test_wav(audio_path, samples=[0, 1200, -1200], sample_rate=44100)
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = VoiceTranscriber(model_name="small", language="pt")
    text = transcriber.transcribe_existing_wav(audio_path)

    assert text == "Vou a Estocolmo de 10 a 15 de outubro."
    assert audio_path.exists()
    assert fake_whisper.loaded_models == ["small"]
    assert transcriber.last_audio_diagnostics is not None
    assert transcriber.last_audio_diagnostics.sample_rate == 44100
    assert transcriber.last_audio_rms > 0


def test_record_start_resolves_device_without_pre_recording_probe(monkeypatch, tmp_path: Path) -> None:
    fake_sounddevice = _FakeCaptureSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    def fail_if_probe_runs(*args, **kwargs):
        raise AssertionError("choose_best_input_device must not run before capture starts")

    monkeypatch.setattr("assistant.voice_input.choose_best_input_device", fail_if_probe_runs)
    stop_event = threading.Event()
    threading.Timer(0.03, stop_event.set).start()
    transcriber = VoiceTranscriber(
        sample_rate=1000,
        ready_delay_ms=40,
        preroll_ms=120,
        min_record_seconds=0.0,
        debug_audio_path=tmp_path / "debug.wav",
    )

    audio_path = transcriber.record_until_stopped(stop_event, status_callback=fake_sounddevice.events.append)

    try:
        assert audio_path.exists()
        assert fake_sounddevice.stream_created is True
        assert fake_sounddevice.events[0] == "Preparar..."
        assert "A ouvir..." in fake_sounddevice.events
        assert transcriber.last_start_diagnostics.preroll_frames >= 1
        assert transcriber.last_start_diagnostics.mic_click_to_stream_ms is not None
    finally:
        audio_path.unlink(missing_ok=True)


def test_preroll_keeps_first_speech_frames_before_listening_status(monkeypatch, tmp_path: Path) -> None:
    fake_sounddevice = _FakeCaptureSoundDevice(first_sample=12000)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    stop_event = threading.Event()
    threading.Timer(0.03, stop_event.set).start()
    transcriber = VoiceTranscriber(
        sample_rate=1000,
        ready_delay_ms=40,
        preroll_ms=120,
        min_record_seconds=0.0,
        debug_audio_path=tmp_path / "debug.wav",
    )

    audio_path = transcriber.record_until_stopped(stop_event, status_callback=fake_sounddevice.events.append)

    try:
        assert fake_sounddevice.callback_events
        assert fake_sounddevice.events.index("A ouvir...") > fake_sounddevice.callback_events[0]
        assert _first_wav_sample(audio_path) == 12000
        assert _first_wav_sample(tmp_path / "debug.wav") == 12000
    finally:
        audio_path.unlink(missing_ok=True)


def test_ready_delay_does_not_delay_capture_start(monkeypatch, tmp_path: Path) -> None:
    fake_sounddevice = _FakeCaptureSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    stop_event = threading.Event()
    threading.Timer(0.02, stop_event.set).start()
    transcriber = VoiceTranscriber(
        sample_rate=1000,
        ready_delay_ms=80,
        preroll_ms=160,
        min_record_seconds=0.0,
        debug_audio_path=tmp_path / "debug.wav",
    )

    audio_path = transcriber.record_until_stopped(stop_event, status_callback=fake_sounddevice.events.append)

    try:
        diagnostics = transcriber.last_start_diagnostics
        assert diagnostics.stream_to_first_frame_ms is not None
        assert diagnostics.first_frame_to_listening_status_ms is not None
        assert diagnostics.first_frame_to_listening_status_ms > 0
        assert diagnostics.preroll_duration_seconds > 0
    finally:
        audio_path.unlink(missing_ok=True)


def test_stt_accepts_real_speech_like_audio_with_confident_result() -> None:
    decision = decide_stt_acceptance(
        "Vou a Estocolmo de 10 a 15 de outubro.",
        _diagnostics(rms=0.03, peak=0.35, near_silence_percent=40.0),
        {"segments": [{"no_speech_prob": 0.05, "avg_logprob": -0.2, "compression_ratio": 1.1}]},
    )

    assert decision.accepted is True
    assert decision.rejection_reason == ""


def test_stt_rejects_silence_audio_before_trusting_whisper_text() -> None:
    decision = decide_stt_acceptance(
        "de um avião de um avião",
        _diagnostics(rms=0.0001, peak=0.001, near_silence_percent=99.5),
        {"segments": [{"no_speech_prob": 0.2, "avg_logprob": -0.2, "compression_ratio": 1.2}]},
    )

    assert decision.accepted is False
    assert decision.rejection_reason == "silence"


def test_stt_rejects_high_no_speech_probability() -> None:
    decision = decide_stt_acceptance(
        "talvez qualquer coisa",
        _diagnostics(rms=0.02, peak=0.2, near_silence_percent=60.0),
        {"segments": [{"no_speech_prob": 0.91, "avg_logprob": -0.2, "compression_ratio": 1.2}]},
    )

    assert decision.accepted is False
    assert decision.rejection_reason == "silence"


def test_stt_rejects_repeated_hallucinated_phrase() -> None:
    decision = decide_stt_acceptance(
        "de um avião de um avião de um avião de um avião",
        _diagnostics(rms=0.02, peak=0.2, near_silence_percent=60.0),
        {"segments": [{"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 1.2}]},
    )

    assert decision.accepted is False
    assert decision.rejection_reason == "excessive_repetition"


def test_stt_rejects_low_confidence_transcription() -> None:
    decision = decide_stt_acceptance(
        "texto incerto",
        _diagnostics(rms=0.02, peak=0.2, near_silence_percent=60.0),
        {"segments": [{"no_speech_prob": 0.1, "avg_logprob": -1.8, "compression_ratio": 1.2}]},
    )

    assert decision.accepted is False
    assert decision.rejection_reason == "low_confidence"


def test_transcribe_rejects_unreliable_whisper_result(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_test_wav(audio_path, samples=[0, 1, 0, -1], sample_rate=44100)
    fake_whisper = _FakeWhisperModule(
        result={
            "text": "de um avião de um avião de um avião",
            "segments": [{"no_speech_prob": 0.96, "avg_logprob": -0.2, "compression_ratio": 1.1}],
        }
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = VoiceTranscriber(model_name="small", language="pt")
    transcriber.last_audio_diagnostics = _diagnostics(rms=0.0001, peak=0.001, near_silence_percent=99.8)

    try:
        transcriber.transcribe(audio_path)
    except VoiceNoSpeechDetected as exc:
        assert "Nao detetei fala" in str(exc)
    else:
        raise AssertionError("Expected VoiceNoSpeechDetected")


def _write_test_wav(path: Path, samples: list[int], sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))


def _first_wav_sample(path: Path) -> int:
    with wave.open(str(path), "rb") as wav_file:
        data = wav_file.readframes(1)
    return int.from_bytes(data[:2], "little", signed=True)


def _diagnostics(rms: float, peak: float, near_silence_percent: float) -> VoiceAudioDiagnostics:
    return VoiceAudioDiagnostics(
        path="debug.wav",
        sample_rate=44100,
        channels=1,
        sample_width_bytes=2,
        frame_count=44100,
        duration_seconds=1.0,
        rms=rms,
        peak=peak,
        near_silence_percent=near_silence_percent,
        clipping_detected=False,
        level="normal",
    )


class _FakeCaptureInputStream:
    def __init__(self, module, **kwargs) -> None:
        self.module = module
        self.device = kwargs.get("device")
        self.callback = kwargs.get("callback")

    def __enter__(self):
        import numpy as np

        self.module.stream_created = True
        if self.callback is not None:
            self.module.events.append("__frame__")
            self.module.callback_events.append(len(self.module.events))
            self.callback(np.full((4, 1), self.module.first_sample, dtype=np.int16), 4, None, None)
            self.module.events.append("__frame__")
            self.module.callback_events.append(len(self.module.events))
            self.callback(np.full((4, 1), 4000, dtype=np.int16), 4, None, None)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeCaptureSoundDevice:
    def __init__(self, first_sample: int = 8000) -> None:
        self.default = SimpleNamespace(device=[0, None])
        self.first_sample = first_sample
        self.stream_created = False
        self.events: list[str] = []
        self.callback_events: list[int] = []
        self.devices = [
            {"name": "Default Mic", "max_input_channels": 1, "default_samplerate": 44100},
        ]

    def query_devices(self, index=None, kind=None):
        if index is not None:
            return self.devices[int(index)]
        if kind == "input":
            return self.devices[0]
        return self.devices

    def InputStream(self, **kwargs):
        return _FakeCaptureInputStream(self, **kwargs)


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.result: dict[str, object] = {"text": "Vou a Estocolmo de 10 a 15 de outubro."}

    def transcribe(self, audio_path: str, **options):
        self.calls.append({"audio_path": audio_path, **options})
        return self.result


class _FakeWhisperModule:
    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.loaded_models: list[str] = []
        self.model = _FakeWhisperModel()
        if result is not None:
            self.model.result = result

    def load_model(self, model_name: str):
        self.loaded_models.append(model_name)
        return self.model
