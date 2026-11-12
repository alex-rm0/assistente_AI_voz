from __future__ import annotations

import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace
import pytest

from assistant.voice_input import (
    DEFAULT_INITIAL_PROMPT,
    VadConfig,
    VoiceAudioDiagnostics,
    VoiceInputError,
    VoiceNoSpeechDetected,
    VoiceTranscriber,
    analyze_voice_wav,
    decide_stt_acceptance,
    format_voice_audio_diagnostics,
    voice_status_report,
)
from assistant.voice_wake import classify_voice_wake_intent


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


# --- Phase 7.5: VAD / automatic end-of-speech --------------------------------
# Lightweight RMS-based end-of-speech detection for Voice Conversation Mode
# (record_until_speech_complete / record_and_transcribe_vad). Manual MIC keeps
# using record_until_stopped, completely untouched -- these tests exercise
# the NEW capture path only.
#
# The tracker's timing model is block-count based (block_ms derives from
# blocksize/sample_rate, not wall-clock time), so a fake sounddevice module
# that feeds a scripted sequence of frames synchronously inside __enter__
# drives every VAD state transition fully deterministically, with no real
# sleeps needed.

LOUD = 700  # rms ~= 700/32768 ~= 0.0214, above the default 0.01 speech threshold
QUIET = 0  # rms == 0.0


class _FakeVadInputStream:
    """Feeds the scripted plan from a background thread, with a short
    head-start delay -- record_until_speech_complete() only starts routing
    frames to the tracker (vs. the preroll buffer) once recording_active is
    set in the method BODY, which runs strictly after __enter__() returns.
    Delivering frames synchronously inside __enter__ itself (as the
    existing _FakeCaptureInputStream does for record_until_stopped, which
    has no such branch) would have every block land in preroll and never
    reach the tracker at all -- silently hanging the outer wait loop."""

    def __init__(self, module, **kwargs) -> None:
        self.module = module
        self.callback = kwargs.get("callback")
        self.blocksize = kwargs.get("blocksize", 1)
        self._thread = None

    def __enter__(self):
        import threading as _threading
        import time as _time

        def feed() -> None:
            import numpy as np

            _time.sleep(0.02)  # let recording_active.set() run first
            if self.callback is None:
                return
            for amplitude, n_blocks in self.module.plan:
                chunk = np.full((self.blocksize, 1), amplitude, dtype=np.int16)
                for _ in range(n_blocks):
                    self.callback(chunk, self.blocksize, None, None)
                    _time.sleep(0.005)  # real pacing -- lets a Timer-driven STOP land mid-delivery

        self._thread = _threading.Thread(target=feed, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeVadSoundDevice:
    def __init__(self, plan: list[tuple[int, int]]) -> None:
        self.default = SimpleNamespace(device=[0, None])
        self.plan = plan
        self.devices = [{"name": "Fake Mic", "max_input_channels": 1, "default_samplerate": 44100}]

    def query_devices(self, index=None, kind=None):
        if index is not None:
            return self.devices[int(index)]
        if kind == "input":
            return self.devices[0]
        return self.devices

    def InputStream(self, **kwargs):
        return _FakeVadInputStream(self, **kwargs)


class _NeverCalledWhisperModule:
    """Fails loudly if Whisper is ever invoked -- proves the VAD path skips
    transcription entirely for pure silence / insufficient-speech captures
    (section 2/8/16) instead of silently succeeding by coincidence."""

    def load_model(self, model_name: str):
        raise AssertionError("Whisper must not be invoked when no valid utterance was captured")


def _vad_transcriber(monkeypatch, tmp_path: Path, plan: list[tuple[int, int]], **overrides) -> VoiceTranscriber:
    fake_sounddevice = _FakeVadSoundDevice(plan)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    kwargs: dict[str, object] = dict(
        sample_rate=1000,
        ready_delay_ms=0,
        preroll_ms=0,
        min_record_seconds=0.0,
        debug_audio_path=tmp_path / "debug.wav",
        vad_config=VadConfig(
            speech_rms_threshold=0.01,
            speech_start_min_ms=40,
            trailing_silence_ms=120,
            min_speech_ms=80,
            max_utterance_ms=2000,
        ),
    )
    kwargs.update(overrides)
    return VoiceTranscriber(**kwargs)


def test_vad_a_silence_before_speech_does_not_trigger_whisper(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(QUIET, 50)])  # full max_utterance_ms of silence

    try:
        transcriber.record_and_transcribe_vad(threading.Event())
        raise AssertionError("Expected VoiceNoSpeechDetected")
    except VoiceNoSpeechDetected:
        pass
    assert transcriber.last_vad_outcome["speech_ms"] == 0
    assert transcriber.last_vad_outcome["reason"] == "max_duration"


def test_vad_b_detected_speech_starts_utterance(monkeypatch, tmp_path: Path) -> None:
    # 2 loud blocks (crosses speech_start_min_ms=40ms after block 1) then
    # enough silence (3 blocks = 120ms) to end the utterance naturally.
    plan = [(LOUD, 2), (QUIET, 3)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)

    audio_path = transcriber.record_until_speech_complete(threading.Event())

    try:
        assert transcriber.last_vad_outcome["reason"] == "end_of_speech"
        assert transcriber.last_vad_outcome["speech_ms"] >= 80  # 2 blocks of 40ms
        assert audio_path.exists()
    finally:
        audio_path.unlink(missing_ok=True)


def test_vad_c_short_internal_pause_does_not_end_utterance(monkeypatch, tmp_path: Path) -> None:
    # loud -> a brief 1-block (40ms) pause, well under the 120ms trailing-
    # silence threshold -> loud again -> real trailing silence. Must be
    # treated as ONE utterance, and Whisper called exactly once.
    plan = [(LOUD, 2), (QUIET, 1), (LOUD, 2), (QUIET, 3)]
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)

    transcriber.record_and_transcribe_vad(threading.Event())

    assert len(fake_whisper.model.calls) == 1
    assert transcriber.last_vad_outcome["reason"] == "end_of_speech"
    # total speech spans both loud bursts -- the brief pause did not reset it
    assert transcriber.last_vad_outcome["speech_ms"] >= 160


def test_vad_d_trailing_silence_threshold_ends_utterance(monkeypatch, tmp_path: Path) -> None:
    plan = [(LOUD, 3), (QUIET, 3)]  # exactly 120ms of trailing silence
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)

    audio_path = transcriber.record_until_speech_complete(threading.Event())

    try:
        assert transcriber.last_vad_outcome["reason"] == "end_of_speech"
        assert transcriber.last_vad_outcome["trailing_silence_ms"] >= 120
    finally:
        audio_path.unlink(missing_ok=True)


def test_vad_e_automatic_end_of_turn_invokes_transcription_once(monkeypatch, tmp_path: Path) -> None:
    plan = [(LOUD, 3), (QUIET, 3)]
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)

    text = transcriber.record_and_transcribe_vad(threading.Event())

    assert text == "Vou a Estocolmo de 10 a 15 de outubro."
    assert len(fake_whisper.model.calls) == 1


def test_vad_f_no_manual_stop_required_for_valid_voice_turn(monkeypatch, tmp_path: Path) -> None:
    """F: the stop_event passed in is never .set() by the test itself --
    only the VAD tracker's own natural completion ends the capture."""
    plan = [(LOUD, 3), (QUIET, 3)]
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)
    stop_event = threading.Event()

    text = transcriber.record_and_transcribe_vad(stop_event)

    assert text
    assert stop_event.is_set() is False
    assert len(fake_whisper.model.calls) == 1


def test_vad_g_ambient_noise_spike_does_not_become_a_turn(monkeypatch, tmp_path: Path) -> None:
    # 1 loud block (40ms) crosses speech_start_min_ms but is well below
    # min_speech_ms=80ms -- a single pop/click, not a real utterance.
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    plan = [(LOUD, 1), (QUIET, 3)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)

    try:
        transcriber.record_and_transcribe_vad(threading.Event())
        raise AssertionError("Expected VoiceInputError for an insufficient noise burst")
    except VoiceInputError as exc:
        assert "curta" in str(exc).lower() or "insuficiente" in str(exc).lower()
    assert 0 < transcriber.last_vad_outcome["speech_ms"] < 80


def test_vad_h_max_utterance_duration_safely_finishes_capture(monkeypatch, tmp_path: Path) -> None:
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    # Continuous speech, never silent -- only the safety ceiling ends it.
    plan = [(LOUD, 60)]  # 60 * 40ms = 2400ms > max_utterance_ms=2000ms
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)

    text = transcriber.record_and_transcribe_vad(threading.Event())

    assert text
    assert transcriber.last_vad_outcome["reason"] == "max_duration"
    assert transcriber.last_vad_outcome["total_capture_ms"] >= 2000
    assert len(fake_whisper.model.calls) == 1


def test_vad_i_manual_stop_still_works(monkeypatch, tmp_path: Path) -> None:
    """I: pressing STOP mid-utterance (continuous speech, never silent, so
    only an external stop or the safety ceiling would end it) terminates
    capture safely -- no crash, no hang -- and is attributed to
    "manual_stop" rather than a VAD-detected end of speech."""
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    plan = [(LOUD, 30)]  # continuous speech; would not end on its own within this plan
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)
    stop_event = threading.Event()
    threading.Timer(0.05, stop_event.set).start()  # fires mid-delivery (feed() paces at 5ms/block)

    text_or_error = None
    try:
        text_or_error = transcriber.record_and_transcribe_vad(stop_event)
    except (VoiceInputError, VoiceNoSpeechDetected):
        pass

    assert transcriber.last_vad_outcome["reason"] == "manual_stop"
    assert text_or_error is None or isinstance(text_or_error, str)


def test_vad_k_input_language_auto_passes_language_none(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_test_wav(audio_path, samples=[0, 1000, -1000], sample_rate=44100)
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = VoiceTranscriber(model_name="medium", language="pt")
    transcriber.transcribe(audio_path, input_language="auto")

    assert fake_whisper.model.calls[0]["language"] is None


def test_vad_l_input_language_pt_forces_portuguese(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_test_wav(audio_path, samples=[0, 1000, -1000], sample_rate=44100)
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = VoiceTranscriber(model_name="medium", initial_prompt=DEFAULT_INITIAL_PROMPT)
    transcriber.transcribe(audio_path, input_language="pt")

    assert fake_whisper.model.calls[0]["language"] == "pt"
    assert fake_whisper.model.calls[0]["initial_prompt"] == DEFAULT_INITIAL_PROMPT


def test_vad_m_input_language_en_forces_english(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_test_wav(audio_path, samples=[0, 1000, -1000], sample_rate=44100)
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = VoiceTranscriber(model_name="medium")
    transcriber.transcribe(audio_path, input_language="en")

    assert fake_whisper.model.calls[0]["language"] == "en"


def test_vad_n_auto_mode_does_not_use_portuguese_only_initial_prompt(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_test_wav(audio_path, samples=[0, 1000, -1000], sample_rate=44100)
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = VoiceTranscriber(model_name="medium", initial_prompt=DEFAULT_INITIAL_PROMPT)
    transcriber.transcribe(audio_path, input_language="auto")

    assert fake_whisper.model.calls[0]["initial_prompt"] != DEFAULT_INITIAL_PROMPT
    assert fake_whisper.model.calls[0]["initial_prompt"] == ""


def test_vad_o_response_language_never_passed_to_input_language_resolution() -> None:
    """O: input_language resolution must never see response_language --
    architecturally enforced by the helper's own signature (no such
    parameter exists to accidentally wire one through)."""
    import inspect

    from assistant.voice_input import _resolve_input_language

    signature = inspect.signature(_resolve_input_language)
    assert "response_language" not in signature.parameters


def test_vad_legacy_transcribe_call_unaffected_by_input_language_feature(monkeypatch, tmp_path: Path) -> None:
    """Existing callers that never pass input_language (manual MIC's
    record_and_transcribe(), any direct .transcribe(path) call) keep the
    exact pre-Phase-7.5 behavior: force self.language, Portuguese prompt."""
    audio_path = tmp_path / "sample.wav"
    _write_test_wav(audio_path, samples=[0, 1000, -1000], sample_rate=44100)
    fake_whisper = _FakeWhisperModule()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = VoiceTranscriber(model_name="base", language="pt", initial_prompt=DEFAULT_INITIAL_PROMPT)
    transcriber.transcribe(audio_path)

    assert fake_whisper.model.calls[0]["language"] == "pt"
    assert fake_whisper.model.calls[0]["initial_prompt"] == DEFAULT_INITIAL_PROMPT


# --- Phase 7.6: barge-in VAD (section 8-11) ----------------------------------
# A conservative, separate VadConfig -- higher RMS threshold, longer
# sustained-activity requirement -- plus require_confirmation_before_recording
# (discard-until-confirmed) so ordinary TTS output bleeding into the mic
# while only monitoring never ends up in what gets sent to Whisper.

BARGE_IN_LOUD = 1200  # rms ~= 1200/32768 ~= 0.0366, above a 0.03 barge-in threshold
TTS_BLEED = 500  # rms ~= 500/32768 ~= 0.01526 -- above the NORMAL 0.01 threshold, below barge-in's

_BARGE_IN_TEST_CONFIG = VadConfig(
    speech_rms_threshold=0.03,
    speech_start_min_ms=80,
    trailing_silence_ms=120,
    min_speech_ms=80,
    max_utterance_ms=2000,
)


def test_barge_in_l_tts_output_alone_does_not_trigger_interruption(monkeypatch, tmp_path: Path) -> None:
    """L: audio at ordinary TTS-bleed level (below the barge-in threshold)
    never confirms, however long it continues -- Whisper is never invoked."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    plan = [(TTS_BLEED, 60)]  # sustained "TTS bleed" for 2.4s, never louder
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)
    confirmed: list[bool] = []

    try:
        transcriber.record_until_speech_complete(
            threading.Event(),
            vad_config=_BARGE_IN_TEST_CONFIG,
            on_speech_confirmed=lambda: confirmed.append(True),
            require_confirmation_before_recording=True,
        )
        raise AssertionError("Expected a benign no-content error -- nothing should have confirmed")
    except (VoiceNoSpeechDetected, VoiceInputError):
        # Since nothing was ever confirmed, no audio was ever committed to
        # `frames` at all (discard-until-confirmed mode) -- this correctly
        # surfaces as the earlier "no audio captured" VoiceInputError
        # rather than VoiceNoSpeechDetected, which assumes SOME (silent)
        # audio was captured. Both are benign/no-Whisper outcomes.
        pass
    assert confirmed == []
    assert transcriber.last_vad_outcome["speech_ms"] == 0


def test_barge_in_m_short_noise_spike_does_not_confirm(monkeypatch, tmp_path: Path) -> None:
    """M: a brief loud spike crossing the barge-in threshold but shorter
    than speech_start_min_ms (a cough, a click) does not confirm an
    interruption."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    plan = [(BARGE_IN_LOUD, 1), (QUIET, 50)]  # one 40ms spike, then silence past max_utterance_ms
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)
    confirmed: list[bool] = []

    try:
        transcriber.record_until_speech_complete(
            threading.Event(),
            vad_config=_BARGE_IN_TEST_CONFIG,
            on_speech_confirmed=lambda: confirmed.append(True),
            require_confirmation_before_recording=True,
        )
        raise AssertionError("Expected a benign no-content error -- the spike alone must not confirm")
    except (VoiceNoSpeechDetected, VoiceInputError):
        pass
    assert confirmed == []
    assert transcriber.last_vad_outcome["speech_ms"] == 0


def test_barge_in_n_sustained_speech_confirms_and_discards_leading_bleed(monkeypatch, tmp_path: Path) -> None:
    """N (and section 9's core protection): sustained genuinely-loud speech
    confirms exactly once, and the committed WAV excludes the earlier
    TTS-bleed-level audio that preceded it -- proving Echo's own output
    never gets baked into what Whisper eventually sees."""
    plan = [(TTS_BLEED, 5), (BARGE_IN_LOUD, 2), (QUIET, 3)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan, preroll_ms=0)
    confirmed_count = 0

    def on_confirmed() -> None:
        nonlocal confirmed_count
        confirmed_count += 1

    audio_path = transcriber.record_until_speech_complete(
        threading.Event(),
        vad_config=_BARGE_IN_TEST_CONFIG,
        on_speech_confirmed=on_confirmed,
        require_confirmation_before_recording=True,
    )

    try:
        assert confirmed_count == 1
        assert transcriber.last_vad_outcome["reason"] == "end_of_speech"
        # Only the post-confirmation audio (2 loud + 3 quiet blocks = 200ms)
        # was committed -- the 5 earlier TTS-bleed blocks (200ms) were
        # discarded, not baked into the WAV.
        assert transcriber.last_audio_duration_seconds < 0.25
    finally:
        audio_path.unlink(missing_ok=True)


# --- Phase 7.6.1: standby VAD config (section 2/3/6) -------------------------
# Real-mic validation showed standby was reusing the ACTIVE conversational
# vad_config (min_speech_ms~300ms) -- a short wake word like "Echo" was
# routinely measured at 200-280ms and rejected as a benign retry BEFORE
# Whisper ever ran. standby_vad_config is a separate, more permissive
# config specifically for wake/sleep-word capture; ACTIVE's vad_config is
# untouched (asserted explicitly in test E).

import dataclasses  # noqa: E402


def _fast_config(config: VadConfig, *, max_utterance_ms: int = 2000) -> VadConfig:
    """A test-speed copy of a real production VadConfig -- same
    thresholds/timing, just a much shorter safety ceiling so a "never
    confirms" test doesn't have to simulate a full 60-second window."""
    return dataclasses.replace(config, max_utterance_ms=max_utterance_ms)


def test_standby_a_40ms_spike_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(LOUD, 1), (QUIET, 50)])
    standby_config = _fast_config(transcriber.standby_vad_config)

    try:
        transcriber.record_and_transcribe_vad(threading.Event(), vad_config=standby_config)
        raise AssertionError("Expected a benign no-content outcome for a 40ms spike")
    except (VoiceNoSpeechDetected, VoiceInputError):
        pass
    assert transcriber.last_vad_outcome["speech_ms"] == 0


def test_standby_b_80ms_accidental_burst_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(LOUD, 2), (QUIET, 50)])
    standby_config = _fast_config(transcriber.standby_vad_config)
    # 80ms of loud audio is still short of standby_vad_config's own
    # speech_start_min_ms (100ms) -- never confirmed as real speech.
    assert 80 < standby_config.speech_start_min_ms

    try:
        transcriber.record_and_transcribe_vad(threading.Event(), vad_config=standby_config)
        raise AssertionError("Expected a benign no-content outcome for an 80ms burst")
    except (VoiceNoSpeechDetected, VoiceInputError):
        pass
    assert transcriber.last_vad_outcome["speech_ms"] == 0


def test_standby_c_short_wake_like_speech_is_accepted(monkeypatch, tmp_path: Path) -> None:
    """C: ~200ms of genuine speech -- comfortably in the 150-250ms range a
    short wake word like "Echo" actually measured at -- is accepted under
    standby_vad_config, and Whisper DOES run (section 5)."""
    fake_whisper = _FakeWhisperModule(result={"text": "Echo"})
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)])
    standby_config = _fast_config(transcriber.standby_vad_config)

    text = transcriber.record_and_transcribe_vad(threading.Event(), vad_config=standby_config)

    assert text == "Echo"
    assert len(fake_whisper.model.calls) == 1
    assert transcriber.last_vad_outcome["reason"] == "end_of_speech"
    assert 150 <= transcriber.last_vad_outcome["speech_ms"] <= 250


def test_standby_e_active_minimum_duration_behavior_unchanged(monkeypatch, tmp_path: Path) -> None:
    """E: the SAME ~200ms speech that standby accepts (test C) is still
    correctly rejected as too short under the normal (ACTIVE) vad_config --
    proving section 9's requirement that ACTIVE's threshold was never
    lowered globally to make wake work."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    # ACTIVE's default trailing_silence_ms is 900ms (23 blocks of 40ms) --
    # needs enough quiet blocks in the plan to actually reach it, unlike
    # standby's much shorter 650ms. _vad_transcriber's OWN default
    # vad_config override is a fast test-tuned one (Phase 7.5), not the
    # real production default -- pass VadConfig() explicitly so
    # transcriber.vad_config reflects the actual unmodified ACTIVE
    # defaults this test needs to verify were never touched.
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 25)], vad_config=VadConfig())
    active_config = _fast_config(transcriber.vad_config)
    assert active_config.min_speech_ms >= 300  # unchanged default

    try:
        transcriber.record_and_transcribe_vad(threading.Event(), vad_config=active_config)
        raise AssertionError("Expected ~200ms of speech to still be rejected as too short under ACTIVE's config")
    except VoiceInputError as exc:
        assert "curta" in str(exc).lower()
    assert 150 <= transcriber.last_vad_outcome["speech_ms"] <= 250


# --- Phase 7.6.5: reduce wake latency (section 9/17.H/17.N) -----------------
# Real validation showed wake ("Echo") felt slow even though it worked --
# standby_vad_config's 650ms trailing_silence_ms was the single largest
# fixed contributor: a short wake word doesn't need the same post-speech
# confirmation margin a full conversational turn does. Lowered to 450ms
# (within the suggested 350-500ms range); ACTIVE's own vad_config is
# completely untouched, asserted explicitly here (as it already was in
# test_standby_e above) so a future regression can't silently creep in.


def test_standby_h_trailing_silence_faster_than_active() -> None:
    transcriber = VoiceTranscriber()

    assert transcriber.vad_config.trailing_silence_ms == 900  # ACTIVE unchanged
    assert transcriber.standby_vad_config.trailing_silence_ms == 450
    assert 350 <= transcriber.standby_vad_config.trailing_silence_ms <= 500
    assert transcriber.standby_vad_config.trailing_silence_ms < transcriber.vad_config.trailing_silence_ms


def test_standby_n_barge_in_and_active_vad_configs_unaffected_by_standby_change() -> None:
    """N: lowering standby_vad_config must not have touched ANY other VAD
    config -- barge_in_vad_config's own trailing_silence_ms is explicitly
    derived from vad_config (see VoiceTranscriber.__init__), so this
    confirms it still reflects ACTIVE's 900ms, not standby's new 450ms."""
    transcriber = VoiceTranscriber()

    assert transcriber.barge_in_vad_config.trailing_silence_ms == transcriber.vad_config.trailing_silence_ms
    assert transcriber.barge_in_vad_config.trailing_silence_ms == 900
    assert transcriber.barge_in_vad_config.speech_rms_threshold == 0.0045
    assert transcriber.barge_in_vad_config.speech_start_min_ms == 300


def test_standby_wake_timing_records_stt_ms_on_last_vad_outcome(monkeypatch, tmp_path: Path) -> None:
    """Phase 7.6.5 (section 8): record_and_transcribe_vad must record how
    long Whisper itself took into last_vad_outcome["stt_ms"], so the
    controller's [VOICE_WAKE_TIMING] diagnostic can report a real
    stt_ms instead of guessing."""
    fake_whisper = _FakeWhisperModule(result={"text": "Echo"})
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)])
    standby_config = _fast_config(transcriber.standby_vad_config)

    transcriber.record_and_transcribe_vad(threading.Event(), vad_config=standby_config)

    assert "stt_ms" in transcriber.last_vad_outcome
    assert transcriber.last_vad_outcome["stt_ms"] >= 0.0


def test_standby_wake_timing_records_stt_ms_even_when_stt_rejects(monkeypatch, tmp_path: Path) -> None:
    """Phase 7.6.5 (section 8): stt_ms must still be recorded for an
    IGNORED wake attempt (STT ran but the result was rejected/empty) --
    diagnosing why a false wake attempt felt slow matters just as much as
    diagnosing a successful one."""
    fake_whisper = _FakeWhisperModule(result={"text": ""})
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)])
    standby_config = _fast_config(transcriber.standby_vad_config)

    try:
        transcriber.record_and_transcribe_vad(threading.Event(), vad_config=standby_config)
    except (VoiceNoSpeechDetected, VoiceInputError):
        pass

    assert "stt_ms" in transcriber.last_vad_outcome


# --- Phase 7.6.2: re-tuned barge-in thresholds + aggregate diagnostics -------
# Real-mic validation showed the Phase 7.6 barge-in detector never
# confirmed even when the user clearly spoke over Anna: [BARGE_IN]
# event=detector_started with nothing after it, and no diagnostic to
# explain why. Two changes: (1) the default barge_in_vad_config threshold/
# duration were moderately relaxed (0.035->0.02 RMS, 250ms->180ms) --
# still well above normal listening's own 0.01/120ms, so self-TTS
# protection remains real; (2) record_until_speech_complete now tracks
# peak_rms/max_sustained_ms/candidate_entered so a future real session's
# "never confirmed" can actually be diagnosed.

PLAUSIBLE_SPEECH = 750  # rms ~= 750/32768 ~= 0.0229 -- below Phase 7.6's
# original 0.035 barge-in threshold, above the Phase 7.6.2-adjusted 0.02.


def test_barge_in_h_lower_volume_real_user_speech_can_confirm(monkeypatch, tmp_path: Path) -> None:
    """H: audio at ~0.023 RMS -- plausible for a user speaking normally
    from typical mic distance, but below Phase 7.6's original 0.035
    threshold -- now confirms under the adjusted default barge_in_vad_config.

    A brief quiet lead-in precedes the speech: realistically, monitoring
    always starts with Anna already mid-sentence and the user silent for
    at least a little while before they decide to interrupt, so the
    settle window (section 10) learns a near-silent baseline first -- the
    same way it would in real use -- rather than the degenerate case of
    "user speech present from the very first monitored instant"."""
    plan = [(QUIET, 5), (PLAUSIBLE_SPEECH, 8), (QUIET, 3)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)
    assert transcriber.barge_in_vad_config.speech_rms_threshold < 0.035  # confirms the adjustment took effect
    confirmed: list[bool] = []

    audio_path = transcriber.record_until_speech_complete(
        threading.Event(),
        vad_config=transcriber.barge_in_vad_config,
        on_speech_confirmed=lambda: confirmed.append(True),
        require_confirmation_before_recording=True,
    )

    try:
        assert confirmed == [True]
    finally:
        audio_path.unlink(missing_ok=True)


QUIET_AMBIENT_BLEED = 100  # rms ~= 100/32768 ~= 0.00305 -- below the recalibrated 0.0045 floor


def test_barge_in_i_quiet_ambient_bleed_does_not_confirm_after_adjustment(monkeypatch, tmp_path: Path) -> None:
    """I: sustained low-level ambient/TTS-bleed audio -- below the
    recalibrated 0.0045 floor -- still never confirms. Self-TTS protection
    remains intact for the realistic case the floor was actually chosen
    for (comfortably above pure silence, below the one real observed
    interruption peak of 0.00665)."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    plan = [(QUIET_AMBIENT_BLEED, 60)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)
    confirmed: list[bool] = []

    try:
        transcriber.record_and_transcribe_vad(
            threading.Event(),
            vad_config=transcriber.barge_in_vad_config,
            on_speech_confirmed=lambda: confirmed.append(True),
            require_confirmation_before_recording=True,
        )
        raise AssertionError("Expected a benign no-content outcome for quiet ambient bleed")
    except (VoiceNoSpeechDetected, VoiceInputError):
        pass
    assert confirmed == []


def test_barge_in_adaptive_settle_window_learns_bleed_louder_than_the_floor(monkeypatch, tmp_path: Path) -> None:
    """Without the settle window, bleed louder than the 0.0045 absolute
    floor from block 1 would false-confirm: there's never a "quiet" block
    to teach the baseline, since every block looks loud relative to the
    still-unlearned floor-only threshold. The settle window (section 10)
    closes this gap by unconditionally feeding the baseline for the first
    ~150ms BEFORE candidate detection starts, so it learns the historical
    TTS_BLEED level (0.01526, well above the floor) immediately -- the
    effective threshold then rises to ~3x that, and sustained bleed at the
    SAME level never confirms afterward."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    plan = [(TTS_BLEED, 60)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)

    try:
        transcriber.record_and_transcribe_vad(
            threading.Event(),
            vad_config=transcriber.barge_in_vad_config,
            require_confirmation_before_recording=True,
        )
        raise AssertionError("Expected sustained bleed to be learned and rejected, not confirmed")
    except (VoiceNoSpeechDetected, VoiceInputError):
        pass
    outcome = transcriber.last_vad_outcome
    assert outcome["confirmed"] is False
    assert outcome["baseline_rms"] > 0.0  # the settle window DID learn it
    assert outcome["effective_threshold"] > outcome["threshold"]  # rose above the floor


def test_barge_in_j_candidate_duration_accumulation_tracked_correctly(monkeypatch, tmp_path: Path) -> None:
    """J: a loud burst that resets before crossing the confirmation
    threshold, followed by a second burst that does cross it -- the new
    monitor_summary diagnostics (candidate_entered/max_sustained_ms/
    peak_rms/confirmed) should reflect this correctly. A quiet lead-in
    comes first so the settle window (section 10) learns a near-silent
    baseline, not the speech level itself -- see test H's docstring for
    why "speech present from the very first monitored instant" is an
    unrealistic scenario that the settle window is not meant to handle."""
    plan = [(QUIET, 5), (PLAUSIBLE_SPEECH, 3), (QUIET, 2), (PLAUSIBLE_SPEECH, 8), (QUIET, 3)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)

    audio_path = transcriber.record_until_speech_complete(
        threading.Event(),
        vad_config=transcriber.barge_in_vad_config,
        require_confirmation_before_recording=True,
    )

    try:
        outcome = transcriber.last_vad_outcome
        assert outcome["candidate_entered"] is True
        assert outcome["confirmed"] is True
        assert outcome["max_sustained_ms"] >= transcriber.barge_in_vad_config.speech_start_min_ms
        assert outcome["peak_rms"] > 0
        assert outcome["threshold"] == transcriber.barge_in_vad_config.speech_rms_threshold
        assert outcome["required_ms"] == transcriber.barge_in_vad_config.speech_start_min_ms
    finally:
        audio_path.unlink(missing_ok=True)


def test_barge_in_l_confirmed_interruption_reaches_whisper_once(monkeypatch, tmp_path: Path) -> None:
    """L: once confirmed, the SAME single record_and_transcribe_vad call
    invokes Whisper exactly once on the completed utterance."""
    fake_whisper = _FakeWhisperModule(result={"text": "wait that's not what I meant"})
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    plan = [(QUIET, 5), (PLAUSIBLE_SPEECH, 8), (QUIET, 3)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)

    text = transcriber.record_and_transcribe_vad(
        threading.Event(),
        vad_config=transcriber.barge_in_vad_config,
        require_confirmation_before_recording=True,
    )

    assert text == "wait that's not what I meant"
    assert len(fake_whisper.model.calls) == 1


REAL_OBSERVED_PEAK = 218  # rms ~= 218/32768 ~= 0.00665 -- the EXACT real monitor_summary evidence


def test_barge_in_i_exact_real_observed_peak_rms_can_confirm(monkeypatch, tmp_path: Path) -> None:
    """I (section 16): the EXACT real-world evidence value from the task
    (peak_rms=0.00665) -- sustained realistically (with a quiet lead-in) --
    now confirms under the recalibrated default barge_in_vad_config, where
    it previously (0.02 floor) could not."""
    plan = [(QUIET, 5), (REAL_OBSERVED_PEAK, 8), (QUIET, 3)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)
    confirmed: list[bool] = []

    audio_path = transcriber.record_until_speech_complete(
        threading.Event(),
        vad_config=transcriber.barge_in_vad_config,
        on_speech_confirmed=lambda: confirmed.append(True),
        require_confirmation_before_recording=True,
    )

    try:
        assert confirmed == [True]
        assert abs(transcriber.last_vad_outcome["peak_rms"] - 0.00665) < 0.0005
    finally:
        audio_path.unlink(missing_ok=True)


def test_barge_in_k_short_noise_spike_still_rejected_under_new_default(monkeypatch, tmp_path: Path) -> None:
    """K: a brief spike that crosses the new, lower 0.0045 floor but lasts
    well under speech_start_min_ms (180ms) still does not confirm -- the
    recalibration lowered the RMS bar, not the sustained-duration
    requirement (a keyboard click/cough must still not trigger)."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    short_spike = 250  # rms ~= 0.00763 -- above the 0.0045 floor, for just one 40ms block
    plan = [(QUIET, 5), (short_spike, 1), (QUIET, 50)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)
    confirmed: list[bool] = []

    try:
        transcriber.record_and_transcribe_vad(
            threading.Event(),
            vad_config=transcriber.barge_in_vad_config,
            on_speech_confirmed=lambda: confirmed.append(True),
            require_confirmation_before_recording=True,
        )
        raise AssertionError("Expected a benign no-content outcome for a brief spike")
    except (VoiceNoSpeechDetected, VoiceInputError):
        pass


@pytest.mark.parametrize("burst_blocks", [5, 6, 7])
def test_barge_in_does_not_stop_tts_for_burst_shorter_than_usable_speech(monkeypatch, tmp_path, burst_blocks):
    """The real log confirmed at 200ms, then threw away a 240ms capture."""
    plan = [(QUIET, 5), (10000, burst_blocks), (QUIET, 50)]
    transcriber = _vad_transcriber(monkeypatch, tmp_path, plan)
    confirmed = []
    with pytest.raises((VoiceNoSpeechDetected, VoiceInputError)):
        transcriber.record_until_speech_complete(
            threading.Event(), vad_config=transcriber.barge_in_vad_config,
            on_speech_confirmed=lambda: confirmed.append(True),
            require_confirmation_before_recording=True)
    assert confirmed == []
    assert transcriber.last_vad_outcome["confirmed"] is False
    assert transcriber._barge_in_carryover is None


def test_barge_in_default_confirmation_waits_for_usable_speech():
    transcriber = VoiceTranscriber()
    assert transcriber.barge_in_vad_config.speech_start_min_ms >= transcriber.barge_in_vad_config.min_speech_ms
    assert transcriber.vad_config.speech_start_min_ms == 120


# --- Phase 7.6.6: STANDBY wake fast path (section 3/9/16 D-I) ---------------
# Real logs isolated Whisper Medium as the dominant wake-latency
# contributor (stt_ms 2.8-7s). record_and_transcribe_wake() reuses the SAME
# standby_vad_config capture but transcribes with a small, cheap
# WakeRecognizer ("tiny" by default) first, falling back to the SAME cached
# Medium instance (self.transcribe()) exactly once only when the fast
# result does not read as a clear wake command. classify_voice_wake_intent
# (the SAME deterministic parser the controller uses) is the only authority
# for that decision -- these tests use a fake WakeRecognizer double so the
# fast stage is fully deterministic without needing a real model.


class _FakeWakeRecognizerForVoiceInput:
    """A minimal double matching WakeRecognizer's public interface --
    records every recognize() call so tests can assert the fast path ran
    exactly once (and never twice, section 16.I)."""

    def __init__(self, transcript: str, *, model: str = "tiny") -> None:
        self.transcript = transcript
        self.model_name = model
        self.calls: list[Path] = []

    def recognize(self, audio_path: Path, *, input_language: str = "pt"):
        from assistant.wake_recognizer import WakeRecognitionResult

        self.calls.append(audio_path)
        return WakeRecognitionResult(
            transcript=self.transcript, model=self.model_name, model_load_ms=12.0, inference_ms=180.0
        )

    def prewarm(self) -> None:
        pass


def test_wake_path_d_fast_clear_echo_activates_without_medium_fallback(monkeypatch, tmp_path: Path) -> None:
    """D: a clear "Echo" from the fast recognizer must be trusted
    immediately -- Medium (self._model/whisper) is never invoked at all."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("Echo")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    text = transcriber.record_and_transcribe_wake(threading.Event())

    assert text == "Echo"
    assert len(fake_recognizer.calls) == 1
    assert transcriber.last_vad_outcome["recognizer"] == "wake_fast"
    assert transcriber.last_vad_outcome["wake_model"] == "tiny"
    assert transcriber.last_vad_outcome["fallback_to_medium"] is False


def test_wake_path_e_fast_clear_acorda_activates_without_medium_fallback(monkeypatch, tmp_path: Path) -> None:
    """E: the Portuguese wake word must work identically."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("acorda")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    text = transcriber.record_and_transcribe_wake(threading.Event())

    assert text == "acorda"
    assert transcriber.last_vad_outcome["fallback_to_medium"] is False


def test_wake_path_f_english_wake_phrase_activates_without_medium_fallback(monkeypatch, tmp_path: Path) -> None:
    """F: the English wake phrase must work identically."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("wake up")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    text = transcriber.record_and_transcribe_wake(threading.Event())

    assert text == "wake up"
    assert transcriber.last_vad_outcome["fallback_to_medium"] is False


def test_wake_path_g_unrelated_transcript_rearms_without_medium_fallback(monkeypatch, tmp_path: Path) -> None:
    """G (Phase 7.6.7, test B): a clearly unrelated fast transcript must NOT
    pay for a Medium pass any more -- it returns "" straight away so STANDBY
    rearms quickly (the real session paid stt_ms=9184 for a Medium fallback
    that ended in wake_ignored anyway)."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("ola")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    text = transcriber.record_and_transcribe_wake(threading.Event())

    assert text == ""
    assert transcriber.last_vad_outcome["fallback_to_medium"] is False
    assert transcriber.last_vad_outcome["fast_assessment"] == "unrelated"
    assert transcriber.last_vad_outcome["recognizer"] == "wake_fast"


def test_wake_path_h_uncertain_fast_result_falls_back_to_medium_once(monkeypatch, tmp_path: Path) -> None:
    """H (Phase 7.6.7, test D): when the fast transcript is a plausible
    near-miss of the wake word ("Eco") and Medium's reads as a real wake,
    Medium's result is trusted -- the fallback still recovers a wake the
    small model misheard."""
    fake_whisper = _FakeWhisperModule(result={"text": "Echo"})
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("Eco.")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    text = transcriber.record_and_transcribe_wake(threading.Event())

    assert text == "Echo"
    assert len(fake_whisper.model.calls) == 1
    assert transcriber.last_vad_outcome["recognizer"] == "medium_fallback"
    assert transcriber.last_vad_outcome["fallback_to_medium"] is True
    assert transcriber.last_vad_outcome["fast_assessment"] == "near_wake"
    assert transcriber.last_vad_outcome["medium_cold_load"] is True


def test_wake_path_i_no_duplicate_stt_across_fast_and_fallback(monkeypatch, tmp_path: Path) -> None:
    """I: the fast recognizer and Medium must each run at most ONCE per
    utterance -- never duplicated STT work -- even when Medium ALSO fails
    to confirm the near-wake candidate."""
    fake_whisper = _FakeWhisperModule(result={"text": "ego"})
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("Ekko")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    text = transcriber.record_and_transcribe_wake(threading.Event())

    assert classify_voice_wake_intent(text) == "none"
    assert len(fake_recognizer.calls) == 1
    assert len(fake_whisper.model.calls) == 1


# --- Phase 7.6.7: fast-wake fallback policy + recognizer diagnostics --------


def test_wake_767_empty_fast_result_rearms_without_medium(monkeypatch, tmp_path: Path) -> None:
    """C: tiny producing nothing usable (empty / punctuation only) must
    rearm quickly -- no Medium pass."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    fake_recognizer = _FakeWakeRecognizerForVoiceInput(" ... ")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    assert transcriber.record_and_transcribe_wake(threading.Event()) == ""
    assert transcriber.last_vad_outcome["fast_assessment"] == "empty"
    assert transcriber.last_vad_outcome["fallback_to_medium"] is False


def test_wake_767_real_tiny_miss_transcript_rearms_without_medium(monkeypatch, tmp_path: Path) -> None:
    """B/E: "Ainda, a" is what real tiny produced (offline replay) on the
    session's real low-level (rms 0.0026) STANDBY capture -- an unrelated
    hallucination, so it must rearm, not spend ~9s in Medium."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("Ainda, a")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    assert transcriber.record_and_transcribe_wake(threading.Event()) == ""
    assert transcriber.last_vad_outcome["fast_assessment"] == "unrelated"


def test_wake_767_cancelled_capture_never_starts_medium(monkeypatch, tmp_path: Path) -> None:
    """H/I (shutdown during the fast recognizer): once stop_event is set --
    STANDBY exited or app close began -- a near-wake result must NOT start
    the slow, uninterruptible Medium pass that would hold shutdown up."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    stop_event = threading.Event()

    class _StoppingRecognizer(_FakeWakeRecognizerForVoiceInput):
        def recognize(self, audio_path: Path, *, input_language: str = "pt"):
            stop_event.set()  # shutdown arrives while tiny is running
            return super().recognize(audio_path, input_language=input_language)

    fake_recognizer = _StoppingRecognizer("Eco")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    assert transcriber.record_and_transcribe_wake(stop_event) == ""
    assert transcriber.last_vad_outcome["fallback_to_medium"] is False


def test_wake_767_recognizer_diagnostic_line_is_bounded(monkeypatch, tmp_path: Path, capsys) -> None:
    """Section 1/2: one [VOICE_WAKE_RECOGNIZER] line per pass, with a short
    single-line transcript, assessment, confidence fields and the reason the
    Medium fallback did or didn't run."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    long_text = 'uma frase "longa" ' * 10
    fake_recognizer = _FakeWakeRecognizerForVoiceInput(long_text)
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    transcriber.record_and_transcribe_wake(threading.Event())

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[VOICE_WAKE_RECOGNIZER]")]
    assert len(lines) == 1
    line = lines[0]
    assert "recognizer=tiny" in line
    assert "assessment=unrelated" in line
    assert "fallback_reason=skipped_unrelated" in line
    assert "no_speech_prob=" in line and "avg_logprob=" in line and "audio_rms=" in line
    transcript = line.split('transcript="', 1)[1].split('"', 1)[0]
    assert len(transcript) <= 48
    assert transcript.endswith("...")


def test_wake_767_missed_wake_audio_is_kept_for_offline_replay(monkeypatch, tmp_path: Path) -> None:
    """Section 5: the debug WAV is overwritten by the next capture; a
    non-wake STANDBY capture is copied aside so alias decisions can be
    based on replaying REAL missed audio."""
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("ola")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    transcriber.record_and_transcribe_wake(threading.Event())

    assert (tmp_path / "last_wake_miss.wav").exists()


def test_wake_767_clear_wake_does_not_keep_miss_audio(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "whisper", _NeverCalledWhisperModule())
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("Echo")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 17)], wake_recognizer=fake_recognizer
    )

    assert transcriber.record_and_transcribe_wake(threading.Event()) == "Echo"
    assert not (tmp_path / "last_wake_miss.wav").exists()


def test_wake_path_active_mode_still_uses_medium_transcriber_directly(monkeypatch, tmp_path: Path) -> None:
    """B: ACTIVE conversation (record_and_transcribe_vad, no wake_recognizer
    involved at all) must keep going straight to Medium -- proving the fast
    path is wired ONLY into the new record_and_transcribe_wake method, never
    into ACTIVE's own path."""
    fake_whisper = _FakeWhisperModule(result={"text": "uma pergunta real"})
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    fake_recognizer = _FakeWakeRecognizerForVoiceInput("should never be used")
    transcriber = _vad_transcriber(
        monkeypatch, tmp_path, [(LOUD, 5), (QUIET, 25)], wake_recognizer=fake_recognizer
    )

    text = transcriber.record_and_transcribe_vad(threading.Event())

    assert text == "uma pergunta real"
    assert fake_recognizer.calls == []  # the fast recognizer was never touched
