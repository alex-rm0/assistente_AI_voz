"""Debug option voice.debug_save_captures (off by default): accepted voice
captures are kept in data/debug/voice_captures/ as <stamp>.wav plus the
transcript in <stamp>.txt, for scripts/benchmark_stt.py.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from assistant.voice_input import VoiceNoSpeechDetected, VoiceTranscriber

ROOT = Path(__file__).resolve().parents[1]


def _transcriber(tmp_path: Path, result: str | Exception) -> VoiceTranscriber:
    debug_wav = tmp_path / "last_voice_input.wav"
    debug_wav.write_bytes(b"RIFF-fake-wav")
    transcriber = VoiceTranscriber(model_name="medium", debug_audio_path=debug_wav)

    def record(stop_event, **kwargs):
        temp = tmp_path / "turn.wav"
        temp.write_bytes(b"RIFF-fake-wav")
        return temp

    def transcribe(audio_path, *, delete_after=True, input_language=None):
        if isinstance(result, Exception):
            raise result
        return result

    transcriber.record_until_speech_complete = record
    transcriber.record_until_stopped = lambda stop_event, status_callback=None: record(stop_event)
    transcriber.transcribe = transcribe
    return transcriber


def test_capture_saving_is_off_by_default(tmp_path: Path) -> None:
    transcriber = _transcriber(tmp_path, "olá echo")

    assert transcriber.capture_dir is None
    assert transcriber.record_and_transcribe_vad(threading.Event()) == "olá echo"
    assert not (tmp_path / "voice_captures").exists()


@pytest.mark.parametrize("path", ["vad", "manual"])
def test_accepted_capture_is_saved_with_its_transcript(tmp_path: Path, path: str) -> None:
    transcriber = _transcriber(tmp_path, "qual é a capital de França?")
    transcriber.capture_dir = tmp_path / "voice_captures"

    if path == "vad":
        text = transcriber.record_and_transcribe_vad(threading.Event())
    else:
        text = transcriber.record_and_transcribe(threading.Event())

    assert text == "qual é a capital de França?"
    wavs = sorted(transcriber.capture_dir.glob("*.wav"))
    txts = sorted(transcriber.capture_dir.glob("*.txt"))
    assert len(wavs) == 1 and len(txts) == 1
    assert wavs[0].stem == txts[0].stem
    assert wavs[0].read_bytes() == b"RIFF-fake-wav"
    assert txts[0].read_text(encoding="utf-8").strip() == "qual é a capital de França?"


def test_rejected_capture_is_not_saved(tmp_path: Path) -> None:
    transcriber = _transcriber(tmp_path, VoiceNoSpeechDetected("Nao detetei fala."))
    transcriber.capture_dir = tmp_path / "voice_captures"

    with pytest.raises(VoiceNoSpeechDetected):
        transcriber.record_and_transcribe_vad(threading.Event())
    assert not transcriber.capture_dir.exists() or not any(transcriber.capture_dir.iterdir())


def test_setting_defaults_to_off_and_captures_are_git_ignored() -> None:
    settings = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
    assert settings["voice"]["debug_save_captures"] is False
    assert "data/debug/voice_captures/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
