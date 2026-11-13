from __future__ import annotations

import threading

import numpy as np
import pytest

from assistant.voice_input import VoiceNoSpeechDetected
from tests.test_voice_input import QUIET, _vad_transcriber


class FixedGate:
    def __init__(self, speech):
        self.speech = speech
        self.max_probability = 0.95 if speech else 0.13

    def is_speech(self, chunk):
        return self.speech


def test_long_loud_blow_does_not_confirm_or_stop_tts(monkeypatch, tmp_path):
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(QUIET, 5), (10000, 25), (QUIET, 30)])
    transcriber.barge_in_speech_gate_factory = lambda **_: FixedGate(False)
    confirmed = []
    with pytest.raises(VoiceNoSpeechDetected):
        transcriber.record_until_speech_complete(
            threading.Event(), vad_config=transcriber.barge_in_vad_config,
            on_speech_confirmed=lambda: confirmed.append(True), require_confirmation_before_recording=True)
    assert confirmed == []
    assert transcriber.last_vad_outcome["confirmed"] is False


def test_real_speech_gate_can_confirm_without_changing_audio(monkeypatch, tmp_path):
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(QUIET, 5), (700, 10), (QUIET, 30)])
    transcriber.barge_in_speech_gate_factory = lambda **_: FixedGate(True)
    confirmed = []
    path = transcriber.record_until_speech_complete(
        threading.Event(), vad_config=transcriber.barge_in_vad_config,
        on_speech_confirmed=lambda: confirmed.append(True), require_confirmation_before_recording=True)
    try:
        assert confirmed == [True]
        assert transcriber.last_audio_rms > 0
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("during_capture", [False, True])
def test_vad_failure_never_falls_back_to_volume_only(monkeypatch, tmp_path, during_capture):
    def broken(**kwargs):
        if not during_capture:
            raise ImportError("onnxruntime missing")
        class BrokenGate:
            def is_speech(self, chunk):
                raise RuntimeError("inference failed")
        return BrokenGate()
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(QUIET, 5), (10000, 30), (QUIET, 30)])
    transcriber.barge_in_speech_gate_factory = broken
    confirmed = []
    with pytest.raises(VoiceNoSpeechDetected):
        transcriber.record_until_speech_complete(
            threading.Event(), vad_config=transcriber.barge_in_vad_config,
            on_speech_confirmed=lambda: confirmed.append(True), require_confirmation_before_recording=True)
    assert confirmed == []
    assert transcriber.last_vad_outcome["reason"] == "speech_gate_unavailable"


class FakeSession:
    def __init__(self, probability=.95):
        self.probability = probability
        self.calls = []

    def run(self, outputs, inputs):
        self.calls.append({key: value.copy() for key, value in inputs.items()})
        return np.array([[self.probability]], dtype=np.float32), inputs["h"] + 1, inputs["c"] + 1


def test_streaming_gate_keeps_state_per_capture_and_does_not_change_input():
    from assistant.barge_in_speech import StreamingSpeechGate
    session = FakeSession()
    gate = StreamingSpeechGate(16000, session=session)
    original = np.full((640, 1), 1000, dtype=np.int16)
    assert gate.is_speech(original)
    assert gate.is_speech(original)
    assert np.all(original == 1000)
    assert session.calls[0]["input"].shape == (1, 576)
    assert not session.calls[0]["h"].any()
    assert session.calls[1]["h"].all()
    assert gate.pending.size < 512
    fresh = StreamingSpeechGate(16000, session=session)
    fresh.is_speech(original)
    assert not session.calls[-1]["h"].any()


def test_streaming_gate_resamples_44100_and_bounds_its_buffer():
    from assistant.barge_in_speech import StreamingSpeechGate
    session = FakeSession(.13)
    gate = StreamingSpeechGate(44100, session=session)
    for _ in range(50):
        assert not gate.is_speech(np.full((1764, 1), 1000, dtype=np.int16))
        assert gate.pending.size < 512
    assert 61 <= len(session.calls) <= 63  # 2 seconds / 32ms
    assert gate.max_probability == pytest.approx(.13)


def test_nonfinite_model_output_is_not_treated_as_speech():
    from assistant.barge_in_speech import StreamingSpeechGate
    gate = StreamingSpeechGate(16000, session=FakeSession(float("nan")))
    with pytest.raises(ValueError):
        gate.is_speech(np.zeros((640, 1), dtype=np.int16))


def test_missing_local_model_fails_without_downloading(tmp_path):
    from assistant.barge_in_speech import create_speech_gate
    with pytest.raises(FileNotFoundError):
        create_speech_gate(sample_rate=44100, runtime_python=tmp_path / "missing/Scripts/python.exe")


def test_normal_capture_never_uses_barge_in_speech_gate(monkeypatch, tmp_path):
    transcriber = _vad_transcriber(monkeypatch, tmp_path, [(QUIET, 5), (700, 10), (QUIET, 30)])
    def forbidden(**kwargs):
        raise AssertionError("The speech gate is for barge-in only")
    transcriber.barge_in_speech_gate_factory = forbidden
    path = transcriber.record_until_speech_complete(threading.Event())
    assert path.is_file()
    path.unlink()
