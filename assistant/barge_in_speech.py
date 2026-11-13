"""CPU Silero speech gate for barge-in; no downloads or shared capture state."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def model_path(runtime_python):
    root = Path(runtime_python).resolve().parent.parent
    candidates = [root / "Lib/site-packages/faster_whisper/assets/silero_vad_v6.onnx"]
    candidates.extend(root.glob("lib/python*/site-packages/faster_whisper/assets/silero_vad_v6.onnx"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Local Silero VAD model not found in the configured STT runtime")


@lru_cache(maxsize=2)
def _cpu_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.inter_op_num_threads = options.intra_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 4
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=options)


def create_speech_gate(*, sample_rate, runtime_python):
    return StreamingSpeechGate(sample_rate, session=_cpu_session(str(model_path(runtime_python))))


class StreamingSpeechGate:
    """Silero v6: 512 samples at 16kHz plus 64 context samples per step.

    Only the stateless ONNX session is cached. Recurrent state, resampler and
    pending samples belong to this capture, never the previous voice turn.
    """
    def __init__(self, sample_rate, *, session):
        import numpy as np
        if sample_rate <= 0:
            raise ValueError("Invalid microphone sample rate")
        self.sample_rate, self.session = sample_rate, session
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros_like(self.h)
        self.context = np.zeros(64, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.resampler_state = None
        self.max_probability = 0.0

    def is_speech(self, chunk):
        import numpy as np
        mono = np.asarray(chunk)
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        pcm = mono.astype("<i2").tobytes()
        if self.sample_rate != 16000:
            # The project runtime is Python 3.11. Keep a streaming resampler
            # so 44.1kHz block boundaries never duplicate/drop sample ranges.
            import audioop
            pcm, self.resampler_state = audioop.ratecv(pcm, 2, 1, self.sample_rate, 16000, self.resampler_state)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.pending = np.concatenate((self.pending, samples))
        speech = False
        while self.pending.size >= 512:
            frame, self.pending = self.pending[:512], self.pending[512:]
            inputs = np.concatenate((self.context, frame))[None, :]
            probability, self.h, self.c = self.session.run(None, {"input": inputs, "h": self.h, "c": self.c})
            probability = float(np.asarray(probability).reshape(-1)[0])
            if not np.isfinite(probability):
                raise ValueError("Non-finite Silero speech probability")
            self.context = frame[-64:].copy()
            self.max_probability = max(self.max_probability, probability)
            speech = speech or probability >= 0.5
        return speech
