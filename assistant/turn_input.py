"""Immutable per-turn input, not the future IncomingMessage contract.

message_id/text/source_channel/received_at can map to that contract later;
capture identity, modality and acoustic evidence belong to its metadata.
ContextVars are scoped call contexts, never a transcriber's latest result.
"""
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from uuid import uuid4


@dataclass(frozen=True)
class TurnInput:
    text: str
    source: str = "typed"
    avg_logprob: float | None = None
    no_speech_probability: float | None = None
    compression_ratio: float | None = None
    stt_model: str | None = None
    capture_id: str | None = None
    session_id: str | None = None
    capture_mode: str | None = None
    message_id: str = field(default_factory=lambda: uuid4().hex)
    received_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_channel: str = "echo_os"

    def __post_init__(self):
        if self.source not in {"typed", "voice"}:
            raise ValueError("Unknown input source")
        if self.source == "typed" and any(value is not None for value in (
                self.avg_logprob, self.no_speech_probability, self.compression_ratio,
                self.stt_model, self.capture_id, self.capture_mode)):
            raise ValueError("Typed input cannot carry acoustic evidence")

    @classmethod
    def typed(cls, text):
        return cls(text=str(text or ""))


_capture = ContextVar("echo_capture", default=None)
_turn = ContextVar("echo_turn_input", default=None)


class CaptureScope:
    def __init__(self, *, session_id=None, mode="manual", capture_id=None):
        self.session_id = str(session_id) if session_id is not None else None
        self.mode = mode
        self.capture_id = capture_id or uuid4().hex
        self.evidence = {}

    def __enter__(self):
        self.token = _capture.set(self)
        return self

    def __exit__(self, *args):
        _capture.reset(self.token)

    def finish(self, text):
        return TurnInput(text=str(text or "").strip(), source="voice", capture_id=self.capture_id,
                         session_id=self.session_id, capture_mode=self.mode, **self.evidence)


def record_stt_evidence(*, model, avg_logprob=None, no_speech_probability=None, compression_ratio=None):
    scope = _capture.get()
    if scope is not None:
        def finite(value):
            return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
        scope.evidence = dict(stt_model=str(model), avg_logprob=finite(avg_logprob),
                              no_speech_probability=finite(no_speech_probability), compression_ratio=finite(compression_ratio))


def capture_transcription(record, *, session_id=None, mode="manual"):
    with CaptureScope(session_id=session_id, mode=mode) as scope:
        return scope.finish(record())


def current_turn_input():
    return _turn.get()


def respond_to_turn(engine, value):
    turn = value if isinstance(value, TurnInput) else TurnInput.typed(value)
    engine.begin_request()
    token = _turn.set(turn)
    try:
        if turn.source == "voice":
            from assistant.speech_clarity import observe_speech_clarity, repeat_response
            enforce = getattr(engine, "speech_clarity_enforce", False) is True
            decision = observe_speech_clarity(turn, engine, enforce=enforce)
            if enforce and decision.would_ask_repeat:
                engine._begin_turn_trace(turn.text)
                return engine._complete_turn(turn.text, repeat_response(engine.response_language), "SPEECH_CLARITY",
                                             selected_path="SPEECH_CLARITY", technical=True, remember=False)
        return engine._respond_inner(turn.text)
    except Exception as exc:
        return engine._handle_unexpected_error(turn.text, exc)
    finally:
        _turn.reset(token)
