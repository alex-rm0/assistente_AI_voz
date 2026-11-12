from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import threading

import pytest

from assistant.turn_input import TurnInput, capture_transcription, current_turn_input, record_stt_evidence


def test_capture_is_immutable_and_typed_has_no_acoustic_confidence():
    def record():
        record_stt_evidence(model="turbo", avg_logprob=-.7, no_speech_probability=.1, compression_ratio=1.2)
        return "Francisca e Karina"
    voice = capture_transcription(record, session_id="s1", mode="active")
    assert voice.source == "voice" and voice.avg_logprob == -.7
    assert voice.stt_model == "turbo" and voice.capture_id and voice.session_id == "s1"
    with pytest.raises(FrozenInstanceError):
        voice.text = "changed"
    typed = TurnInput.typed(voice.text)
    assert typed.source == "typed" and typed.avg_logprob is None
    assert typed.capture_id is None and typed.no_speech_probability is None
    with pytest.raises(ValueError):
        TurnInput(text="typed", avg_logprob=-1)


def test_captures_are_local_across_wake_barge_in_and_next_capture():
    barrier = threading.Barrier(2)
    def capture(mode, confidence):
        def record():
            record_stt_evidence(model=mode, avg_logprob=confidence)
            barrier.wait(timeout=3)
            return mode
        return capture_transcription(record, session_id="same-session", mode=mode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        wake = pool.submit(capture, "standby", -.2)
        barge = pool.submit(capture, "barge_in", -.8)
        first, second = wake.result(), barge.result()
    assert (first.avg_logprob, second.avg_logprob) == (-.2, -.8)
    assert first.capture_id != second.capture_id
    third = capture_transcription(lambda: "next", session_id="new-session", mode="active")
    assert third.avg_logprob is None and third.stt_model is None
    assert third.capture_id not in {first.capture_id, second.capture_id}


def test_engine_receives_the_exact_envelope_and_resets_afterwards(tmp_path, monkeypatch):
    from tests.test_automatic_routing_regressions import make_engine
    engine, *_ = make_engine(tmp_path, mode="local")
    received = []
    monkeypatch.setattr(engine, "_respond_inner", lambda text: received.append((text, current_turn_input())) or "ok")
    capture = capture_transcription(lambda: "hello", session_id="s", mode="active")
    assert engine.respond_turn(capture) == "ok"
    engine.respond("typed")
    assert received[0] == ("hello", capture)
    assert received[1][1].source == "typed" and received[1][1].avg_logprob is None
    assert current_turn_input() is None


def test_real_transcriber_reports_wake_then_active_evidence_without_reuse(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from assistant.voice_input import VoiceTranscriber
    from tests.test_stt_backend import Backend, audio
    transcriber = VoiceTranscriber(stt_provider="faster-whisper")
    wake_audio = tmp_path / "wake.wav"
    wake_audio.write_bytes(b"temporary capture")
    monkeypatch.setattr(transcriber, "record_until_speech_complete", lambda *a, **kw: wake_audio)
    transcriber.wake_recognizer = SimpleNamespace(recognize=lambda *a, **kw: SimpleNamespace(
        transcript="Acorda, Echo.", model="tiny", no_speech_prob=.02, avg_logprob=-.7,
        model_load_ms=0, inference_ms=1))
    wake = capture_transcription(lambda: transcriber.record_and_transcribe_wake(threading.Event()), mode="standby", session_id="1")
    assert wake.stt_model == "tiny" and wake.avg_logprob == -.7
    transcriber._stt_backend = Backend(confidence=-.3)
    path = audio(tmp_path)
    active = capture_transcription(lambda: transcriber.transcribe_existing_wav(path), mode="barge_in", session_id="2")
    assert active.stt_model == "large-v3-turbo" and active.avg_logprob == -.3
    assert active.no_speech_probability == 0 and active.compression_ratio == .8
    assert wake.stt_model == "tiny" and wake.avg_logprob == -.7
    next_capture = capture_transcription(lambda: "next", mode="active", session_id="2")
    assert next_capture.avg_logprob is None and next_capture.stt_model is None
    transcriber.close()


def test_background_warmup_cannot_supply_capture_confidence():
    def record():
        thread = threading.Thread(target=lambda: record_stt_evidence(model="warmup", avg_logprob=-2))
        thread.start()
        thread.join()
        return "real text"
    assert capture_transcription(record).avg_logprob is None


def test_failed_capture_does_not_leave_evidence_for_a_later_capture():
    def fail():
        record_stt_evidence(model="failed", avg_logprob=-2)
        raise RuntimeError("capture cancelled")
    with pytest.raises(RuntimeError):
        capture_transcription(fail)
    next_turn = capture_transcription(lambda: "next")
    assert next_turn.avg_logprob is None and next_turn.stt_model is None
