"""Diagnostics only (no behaviour change): one [VOICE_TURN_TIMING] line per
voice turn, and [VOICE_LOST_SPEECH] when a VAD/barge-in saw speech that
nothing will transcribe. Drives the real EchoUIController voice loop with
scripted transcriber/TTS doubles (no audio hardware).
"""
from __future__ import annotations

import re
import time

from assistant.voice_input import VoiceInputError, VoiceNoSpeechDetected
from tests.test_web_ui_prototype import (
    _FakeEngine,
    _FakeTTSManager,
    _qt_app,
    _run_until,
    _stop_and_drain,
)

TIMING_FIELDS = (
    "session_id",
    "outcome",
    "stt_model",
    "speech_end_to_stt_start_ms",
    "stt_ms",
    "engine_ms",
    "response_to_tts_request_ms",
    "tts_to_first_audio_ms",
    "provider_first_audio_ms",
    "provider_synthesis_ms",
    "speech_end_to_first_sound_ms",
    "mic_closed_ms",
)


def _fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", line))


def _lines(output: str, tag: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith(tag)]


class _TimedVadTranscriber:
    """First ACTIVE capture returns a transcript with the capture/STT marks
    record_until_speech_complete/record_and_transcribe_vad now record; later
    ACTIVE captures wait for a stop. No barge-in capability (the controller
    duck-types it), so no monitor runs while speaking."""

    model_name = "medium"

    def __init__(self) -> None:
        self.calls = 0
        self.last_vad_outcome: dict | None = None

    def record_and_transcribe_vad(self, stop_event, status_callback=None, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            now = time.perf_counter()
            self.last_vad_outcome = {
                "speech_ms": 900.0,
                "trailing_silence_ms": 300.0,
                "capture_ended_at": now - 0.05,
                "stt_started_at": now,
                "stt_ms": 120.0,
            }
            return "ola echo"
        while not stop_event.is_set():
            time.sleep(0.01)
        raise VoiceNoSpeechDetected("Nao detetei fala.")

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        raise AssertionError("manual path not used")


def _controller(transcriber, tts):
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    return EchoUIController(
        _FakeEngine(reply="Spoken reply.").respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.01,
        thinking_min_ms=0,
    )


def test_spoken_voice_turn_logs_one_timing_line_with_all_fields(capsys) -> None:
    transcriber = _TimedVadTranscriber()
    tts = _FakeTTSManager()
    controller = _controller(transcriber, tts)

    controller.startVoiceConversation()
    assert _run_until(lambda: tts.calls and transcriber.calls >= 2, timeout_ms=5000)
    _stop_and_drain(controller)

    lines = _lines(capsys.readouterr().out, "[VOICE_TURN_TIMING]")
    assert len(lines) == 1, lines
    fields = _fields(lines[0])
    assert tuple(fields) == TIMING_FIELDS
    assert fields["outcome"] == "spoken"
    assert fields["stt_model"] == "medium"
    assert fields["stt_ms"] == "120"
    # capture ended 50 ms before STT; speech itself ended 300 ms (trailing
    # silence) before the capture ended.
    assert 330 <= int(fields["speech_end_to_stt_start_ms"]) <= 450
    for key in ("engine_ms", "response_to_tts_request_ms", "tts_to_first_audio_ms", "mic_closed_ms"):
        assert int(fields[key]) >= 0, (key, fields[key])
    assert int(fields["speech_end_to_first_sound_ms"]) >= int(fields["mic_closed_ms"]) >= int(fields["tts_to_first_audio_ms"])
    assert fields["provider_synthesis_ms"] == "4"


def test_tts_failure_still_logs_the_turn_timing(capsys) -> None:
    transcriber = _TimedVadTranscriber()
    tts = _FakeTTSManager(error=RuntimeError("tts down"))
    controller = _controller(transcriber, tts)

    controller.startVoiceConversation()
    assert _run_until(lambda: transcriber.calls >= 2, timeout_ms=5000)
    _stop_and_drain(controller)

    lines = _lines(capsys.readouterr().out, "[VOICE_TURN_TIMING]")
    assert len(lines) == 1, lines
    fields = _fields(lines[0])
    assert fields["outcome"] == "tts_error"
    assert fields["stt_ms"] == "120"
    assert fields["tts_to_first_audio_ms"] == "n/a"
    assert fields["speech_end_to_first_sound_ms"] == "n/a"


class _BargeInTranscriber:
    """ACTIVE: one transcript, then waits. BARGE-IN: scripted per mode --
    "candidate" (loud enough but never confirmed, cancelled when TTS ends)
    or "short" (confirms, then the capture is too short to transcribe)."""

    model_name = "medium"

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.active_calls = 0
        self.barge_in_calls = 0
        self.last_vad_outcome: dict | None = None
        self.vad_config = type("Cfg", (), {"min_speech_ms": 300})()

    def record_and_transcribe_vad(
        self,
        stop_event,
        status_callback=None,
        *,
        input_language=None,
        vad_config=None,
        on_speech_confirmed=None,
        require_confirmation_before_recording=False,
    ) -> str:
        if require_confirmation_before_recording:
            self.barge_in_calls += 1
            if self.barge_in_calls > 1:
                while not stop_event.is_set():
                    time.sleep(0.01)
                raise VoiceNoSpeechDetected("Nao detetei fala.")
            if self.mode == "short":
                if on_speech_confirmed is not None:
                    on_speech_confirmed()
                time.sleep(0.05)
                self.last_vad_outcome = {"speech_ms": 140.0, "candidate_entered": True, "confirmed": True}
                raise VoiceInputError("A fala detetada foi demasiado curta. Fala detetada: 140ms; o minimo e 300ms.")
            while not stop_event.is_set():
                time.sleep(0.01)
            self.last_vad_outcome = {
                "speech_ms": 0.0,
                "candidate_entered": True,
                "confirmed": False,
                "max_sustained_ms": 180,
                "required_ms": 300,
                "peak_rms": 0.02,
                "effective_threshold": 0.01,
                "baseline_rms": 0.004,
                "threshold": 0.01,
            }
            raise VoiceNoSpeechDetected("Nao detetei fala.")
        self.active_calls += 1
        if self.active_calls == 1:
            return "ola echo"
        while not stop_event.is_set():
            time.sleep(0.01)
        raise VoiceNoSpeechDetected("Nao detetei fala.")

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        raise AssertionError("manual path not used")


def _short_player(_path, stop_checker=None, **_kwargs) -> float:
    for _ in range(15):
        if stop_checker and stop_checker():
            return 0.0
        time.sleep(0.01)
    return 0.0


def _barge_in_controller(transcriber):
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    return EchoUIController(
        _FakeEngine(reply="Spoken reply.").respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=_FakeTTSManager(),
        tts_player=_short_player,
        thinking_min_ms=0,
    )


def test_barge_in_candidate_dropped_when_tts_ends_is_logged_as_lost_speech(capsys) -> None:
    transcriber = _BargeInTranscriber("candidate")
    controller = _barge_in_controller(transcriber)

    controller.startVoiceConversation()
    assert _run_until(lambda: transcriber.barge_in_calls >= 1 and transcriber.active_calls >= 2, timeout_ms=6000)
    _stop_and_drain(controller)

    lines = _lines(capsys.readouterr().out, "[VOICE_LOST_SPEECH]")
    assert len(lines) == 1, lines
    fields = _fields(lines[0])
    assert fields["reason"] == "barge_in_candidate_dropped_at_tts_end"
    assert fields["approx_ms"] == "180"
    assert fields["required_ms"] == "300"
    assert "session_id" in fields and "peak_rms" in fields and "effective_threshold" in fields


def test_short_capture_after_barge_in_is_logged_as_lost_speech(capsys) -> None:
    transcriber = _BargeInTranscriber("short")
    controller = _barge_in_controller(transcriber)

    controller.startVoiceConversation()
    assert _run_until(lambda: transcriber.barge_in_calls >= 1 and transcriber.active_calls >= 2, timeout_ms=6000)
    _stop_and_drain(controller)

    lines = _lines(capsys.readouterr().out, "[VOICE_LOST_SPEECH]")
    assert len(lines) == 1, lines
    fields = _fields(lines[0])
    assert fields["reason"] == "short_capture_after_barge_in"
    assert fields["approx_ms"] == "140"
    assert fields["min_speech_ms"] == "300"


def test_plain_no_speech_capture_is_not_lost_speech(capsys) -> None:
    transcriber = _TimedVadTranscriber()
    controller = _controller(transcriber, _FakeTTSManager())

    controller.startVoiceConversation()
    assert _run_until(lambda: transcriber.calls >= 2, timeout_ms=5000)
    _stop_and_drain(controller)

    assert _lines(capsys.readouterr().out, "[VOICE_LOST_SPEECH]") == []
