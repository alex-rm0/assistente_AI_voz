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
