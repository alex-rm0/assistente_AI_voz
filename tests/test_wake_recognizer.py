from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from assistant.wake_recognizer import WakeRecognitionResult, WakeRecognizer


class _FakeWakeWhisperModel:
    def __init__(self, text: str = "Echo") -> None:
        self.calls: list[dict[str, object]] = []
        self.text = text

    def transcribe(self, audio_path: str, **options: object) -> dict[str, object]:
        self.calls.append({"audio_path": audio_path, **options})
        return {"text": self.text}


class _FakeWakeWhisperModule:
    def __init__(self, text: str = "Echo", load_delay_seconds: float = 0.0) -> None:
        self.loaded_models: list[str] = []
        self.model = _FakeWakeWhisperModel(text=text)
        self.load_delay_seconds = load_delay_seconds

    def load_model(self, model_name: str) -> _FakeWakeWhisperModel:
        self.loaded_models.append(model_name)
        if self.load_delay_seconds:
            time.sleep(self.load_delay_seconds)
        return self.model


def test_wake_recognizer_c_model_loads_once_and_is_reused(monkeypatch, tmp_path: Path) -> None:
    """C (Phase 7.6.6 section 5/16): the wake model must be loaded once and
    reused across calls -- never reloaded per utterance."""
    fake_whisper = _FakeWakeWhisperModule(text="Echo")
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    recognizer = WakeRecognizer(model_name="tiny")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    recognizer.recognize(audio_path, input_language="en")
    recognizer.recognize(audio_path, input_language="en")
    recognizer.recognize(audio_path, input_language="en")

    assert fake_whisper.loaded_models == ["tiny"]
    assert len(fake_whisper.model.calls) == 3


def test_wake_recognizer_k_warm_call_reports_zero_model_load_ms(monkeypatch, tmp_path: Path) -> None:
    """K: a warm (already-loaded) call must report model_load_ms=0 -- only
    the call that actually triggers the load reports a real duration."""
    fake_whisper = _FakeWakeWhisperModule(text="Echo")
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    recognizer = WakeRecognizer(model_name="tiny")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    first = recognizer.recognize(audio_path, input_language="en")
    second = recognizer.recognize(audio_path, input_language="en")

    assert isinstance(first, WakeRecognitionResult)
    assert first.model_load_ms > 0.0
    assert second.model_load_ms == 0.0
    assert first.transcript == second.transcript == "Echo"
    assert first.model == "tiny"


def test_wake_recognizer_l_prewarm_absorbs_load_cost_before_first_real_call(monkeypatch, tmp_path: Path) -> None:
    """L: prewarm() (called when STANDBY is entered, section 6) must load
    the model ahead of time -- the first real recognize() call afterward
    reports zero load cost of its own."""
    fake_whisper = _FakeWakeWhisperModule(text="Echo")
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    recognizer = WakeRecognizer(model_name="tiny")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    assert recognizer.is_loaded() is False
    recognizer.prewarm()
    assert recognizer.is_loaded() is True
    assert recognizer.last_load_ms is not None

    result = recognizer.recognize(audio_path, input_language="en")

    assert result.model_load_ms == 0.0
    assert fake_whisper.loaded_models == ["tiny"]


def test_wake_recognizer_prewarm_never_loads_twice_when_racing_a_real_call(monkeypatch, tmp_path: Path) -> None:
    """The lock inside WakeRecognizer must prevent a concurrent prewarm()
    (background thread, section 6) and a real recognize() call from both
    triggering whisper.load_model -- whichever arrives second reuses the
    model the other one just finished loading (section 5)."""
    fake_whisper = _FakeWakeWhisperModule(text="Echo", load_delay_seconds=0.05)
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    recognizer = WakeRecognizer(model_name="tiny")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")
    results: list[WakeRecognitionResult] = []

    def _recognize() -> None:
        results.append(recognizer.recognize(audio_path, input_language="en"))

    prewarm_thread = threading.Thread(target=recognizer.prewarm)
    recognize_thread = threading.Thread(target=_recognize)
    prewarm_thread.start()
    time.sleep(0.01)
    recognize_thread.start()
    prewarm_thread.join(timeout=2)
    recognize_thread.join(timeout=2)

    assert fake_whisper.loaded_models == ["tiny"]
    assert len(results) == 1


def test_wake_recognizer_uses_wake_specific_prompt_not_conversation_prompt(monkeypatch, tmp_path: Path) -> None:
    """The wake recognizer must bias decoding with a short, wake-vocabulary
    prompt -- not the main conversational DEFAULT_INITIAL_PROMPT (travel/
    task vocabulary unrelated to a one-word wake utterance)."""
    fake_whisper = _FakeWakeWhisperModule(text="Echo")
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    recognizer = WakeRecognizer(model_name="tiny")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    recognizer.recognize(audio_path, input_language="pt")

    prompt = fake_whisper.model.calls[0]["initial_prompt"]
    assert "echo" in prompt.lower() or "acorda" in prompt.lower()
    assert "viagens" not in prompt.lower()  # not the conversational prompt
