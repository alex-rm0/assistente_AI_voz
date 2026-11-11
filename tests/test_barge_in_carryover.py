"""Voice test of 11/11: a confirmed barge-in whose capture ended short
(speech_ms=240 < min_speech_ms=300) was discarded as "too short", and the
next capture started from scratch -- the start of the user's sentence was
lost ([VOICE_LOST_SPEECH] reason=short_capture_after_barge_in). The audio
already captured by the barge-in now goes at the start of the next normal
capture (pre-roll), and its speech counts toward that capture's minimum.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from assistant.voice_input import VoiceInputError
from tests.test_voice_input import LOUD, QUIET, _vad_transcriber

# blocks are 40 ms (sample_rate=1000, blocksize=40); confirm after 40 ms,
# end after 120 ms of silence, minimum speech 80 ms.
SHORT_UTTERANCE = [(QUIET, 2), (LOUD, 1), (QUIET, 4)]  # 40 ms of speech


def _capture(transcriber, plan, *, barge_in: bool):
    sys.modules["sounddevice"].plan = plan
    return transcriber.record_until_speech_complete(
        threading.Event(), require_confirmation_before_recording=barge_in
    )


def test_short_barge_in_audio_is_carried_into_the_next_capture(monkeypatch, tmp_path: Path) -> None:
    transcriber = _vad_transcriber(monkeypatch, tmp_path, SHORT_UTTERANCE)
    with pytest.raises(VoiceInputError, match="curta"):
        _capture(transcriber, SHORT_UTTERANCE, barge_in=True)
    barge_in_seconds = transcriber.last_audio_duration_seconds

    # Alone this would also be too short (40 ms < 80 ms); with the barge-in
    # speech carried over it reaches the minimum and is kept.
    path = _capture(transcriber, SHORT_UTTERANCE, barge_in=False)

    assert Path(path).exists()
    assert transcriber.last_vad_outcome["carried_over_speech_ms"] == pytest.approx(40, abs=1)
    assert transcriber.last_vad_outcome["speech_ms"] == pytest.approx(80, abs=1)
    assert transcriber.last_audio_duration_seconds > barge_in_seconds  # the WAV starts with the barge-in audio


def test_carryover_is_used_once(monkeypatch, tmp_path: Path) -> None:
    transcriber = _vad_transcriber(monkeypatch, tmp_path, SHORT_UTTERANCE)
    with pytest.raises(VoiceInputError):
        _capture(transcriber, SHORT_UTTERANCE, barge_in=True)
    _capture(transcriber, SHORT_UTTERANCE, barge_in=False)

    with pytest.raises(VoiceInputError, match="curta"):
        _capture(transcriber, SHORT_UTTERANCE, barge_in=False)


def test_stale_carryover_is_dropped(monkeypatch, tmp_path: Path) -> None:
    transcriber = _vad_transcriber(monkeypatch, tmp_path, SHORT_UTTERANCE)
    transcriber.BARGE_IN_CARRYOVER_TTL_S = 0.0
    with pytest.raises(VoiceInputError):
        _capture(transcriber, SHORT_UTTERANCE, barge_in=True)

    with pytest.raises(VoiceInputError, match="curta"):
        _capture(transcriber, SHORT_UTTERANCE, barge_in=False)


def test_a_normal_short_capture_is_not_carried_over(monkeypatch, tmp_path: Path) -> None:
    transcriber = _vad_transcriber(monkeypatch, tmp_path, SHORT_UTTERANCE)
    with pytest.raises(VoiceInputError):
        _capture(transcriber, SHORT_UTTERANCE, barge_in=False)

    with pytest.raises(VoiceInputError, match="curta"):
        _capture(transcriber, SHORT_UTTERANCE, barge_in=False)
