"""Phase 7.6.6: a dedicated, lightweight speech-recognition path for
STANDBY's wake/sleep-word capture ONLY -- never used for ACTIVE conversation
transcription, which keeps using VoiceTranscriber's own Whisper Medium model
exactly as before.

Real logs isolated Whisper Medium inference as the dominant remaining
wake-latency contributor (speech_start_to_end_ms/end_of_speech_to_stt_
start_ms had already been reduced to a few hundred ms by Phase 7.6.5, but
stt_ms itself was routinely 2.8-7s). The target wake vocabulary is
extremely narrow ("Echo", "acorda", "wake up", and short combinations of
those) -- nothing close to the open-vocabulary, multi-sentence quality
Medium's ACTIVE conversation transcription needs -- so a much smaller model
is a reasonable trade for this ONE path, as long as it never blocks/starves
Medium and is never asked to do ACTIVE's real transcription job.

A local benchmark (see the Phase 7.6.6 report) measured cold load and warm
inference across tiny/base/small/medium on real synthesized audio (Pocket
TTS for Portuguese phrases, Windows SAPI for English phrases): tiny loaded
in ~220ms and transcribed every sample in ~175-360ms with perfect accuracy
on every clean English wake phrase, while "base" occasionally fell into a
multi-second Portuguese hallucination-repetition loop (slower than warm
Medium in that failure mode) and "small" was consistently ~1-1.2s -- over
budget for the <500ms warm-inference target. "tiny" is the only candidate
that reliably met both the <500ms warm and <1s total targets with no
observed catastrophic failure mode, so it is the default here -- backed by
a two-stage fallback to Medium (see VoiceTranscriber.record_and_transcribe_
wake) for whatever real-world accuracy gap it has against Medium, since a
synthetic voice-cloned benchmark cannot reliably validate real human
Portuguese speech recognition accuracy.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_WAKE_MODEL_NAME = "tiny"

# Phase 7.6.6 (section 12): a short, wake-vocabulary-only prompt -- NOT the
# main DEFAULT_INITIAL_PROMPT (assistant/voice_input.py), which biases
# decoding toward unrelated travel/task/date vocabulary that has nothing to
# do with a one-or-two-word wake utterance and would only cost accuracy here.
WAKE_INITIAL_PROMPT_PT = "Echo. Acorda, Echo."
WAKE_INITIAL_PROMPT_EN = "Echo. Wake up, Echo."


@dataclass
class WakeRecognitionResult:
    """What a single recognize() call produced, plus enough timing detail
    for the controller's [VOICE_WAKE_TIMING] diagnostic (section 13) to
    report model/model_load_ms/inference_ms without guessing."""

    transcript: str
    model: str
    model_load_ms: float
    inference_ms: float
    # Phase 7.6.7 (section 1): Whisper's own per-segment confidence signals,
    # surfaced so [VOICE_WAKE_RECOGNIZER] can tell "tiny heard nothing"
    # apart from "tiny confidently heard something else". None when the
    # model returned no segments (or a test double doesn't provide them).
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    language: str = ""


class WakeRecognizer:
    """Loads ONE small Whisper model, once, and reuses it for every
    recognize() call -- section 5's "do NOT load the wake model per
    utterance" requirement. Thread-safe: prewarm() (called once when
    STANDBY is entered, from a background thread -- section 6) and the
    first real recognize() call racing each other only ever load the model
    once, guarded by a lock; whichever call arrives second simply reuses
    the model the other one just finished loading."""

    def __init__(self, model_name: str = DEFAULT_WAKE_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._lock = threading.Lock()
        self.last_load_ms: float | None = None

    def prewarm(self) -> None:
        """Loads the model now if it isn't already loaded. Safe to call
        from a background thread (section 6: never blocks UI entering
        STANDBY) and safe to call more than once (a no-op once loaded)."""
        self._ensure_model()

    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> tuple[Any, float]:
        """Returns (model, load_ms_this_call) -- load_ms_this_call is 0.0
        whenever the model was already cached (the common case, including
        every call after prewarm() or a previous recognize() finished),
        and only non-zero on the ONE call that actually triggers the load.
        This is what lets section 13's model_load_ms= field distinguish a
        genuine cold-load from ordinary warm reuse, instead of repeating
        the original load time on every subsequent call."""
        if self._model is not None:
            return self._model, 0.0
        with self._lock:
            if self._model is None:
                import whisper

                started_at = time.perf_counter()
                self._model = whisper.load_model(self.model_name)
                load_ms = (time.perf_counter() - started_at) * 1000.0
                self.last_load_ms = load_ms
                return self._model, load_ms
        return self._model, 0.0

    def recognize(self, audio_path: Path, *, input_language: str = "pt") -> WakeRecognitionResult:
        """Transcribes audio_path with the small wake model. Does NOT
        delete audio_path (unlike VoiceTranscriber.transcribe) -- the
        caller owns the temp file's lifecycle since it may still need it
        for a Medium fallback pass (section 9)."""
        model, model_load_ms = self._ensure_model()
        normalized = str(input_language or "pt").strip().lower()
        if normalized == "pt":
            whisper_language, prompt = "pt", WAKE_INITIAL_PROMPT_PT
        elif normalized == "en":
            whisper_language, prompt = "en", WAKE_INITIAL_PROMPT_EN
        else:
            whisper_language, prompt = None, ""

        started_at = time.perf_counter()
        result = model.transcribe(
            str(audio_path),
            language=whisper_language,
            initial_prompt=prompt,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            fp16=False,
        )
        inference_ms = (time.perf_counter() - started_at) * 1000.0
        text = str(result.get("text", "")).strip()
        segments = [segment for segment in result.get("segments") or [] if isinstance(segment, dict)]
        no_speech_values = [float(s["no_speech_prob"]) for s in segments if "no_speech_prob" in s]
        logprob_values = [float(s["avg_logprob"]) for s in segments if "avg_logprob" in s]
        return WakeRecognitionResult(
            transcript=text,
            model=self.model_name,
            model_load_ms=model_load_ms,
            inference_ms=inference_ms,
            no_speech_prob=max(no_speech_values) if no_speech_values else None,
            avg_logprob=(sum(logprob_values) / len(logprob_values)) if logprob_values else None,
            language=str(result.get("language") or ""),
        )
