"""GPU STT worker. Executed only by the isolated CUDA runtime."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != script_dir]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3-turbo")
    args = parser.parse_args()
    handles = []
    for entry in sys.path:
        for directory in (Path(entry or ".") / "nvidia").glob("*/bin"):
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                handles.append(os.add_dll_directory(str(directory)))
    started = time.perf_counter()
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="cuda", compute_type="float16")
        if str(model.model.device) != "cuda":
            raise RuntimeError("CUDA não confirmado.")
    except Exception as exc:
        detail = f": {exc.name}" if isinstance(exc, ModuleNotFoundError) else ""
        emit({"ok": False, "error": f"Não consegui iniciar o STT CUDA ({type(exc).__name__}{detail}). Verifica runtime, GPU e DLLs CUDA."})
        return 1
    emit({"ok": True, "device": "cuda", "model_load_ms": (time.perf_counter()-started)*1000})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            segments, info = model.transcribe(
                request["path"], language=request["language"], initial_prompt=request["initial_prompt"],
                task="transcribe", temperature=0., condition_on_previous_text=False,
                beam_size=1, vad_filter=False,
            )
            segments = list(segments)
            emit({"ok": True, "result": {
                "text": " ".join(segment.text.strip() for segment in segments).strip(),
                "language": info.language,
                "segments": [{"no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob,
                              "compression_ratio": s.compression_ratio} for s in segments],
            }})
        except Exception as exc:
            emit({"ok": False, "error": f"Falha na transcrição CUDA ({type(exc).__name__})."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
