from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


MODEL_NAME = "ResembleAI/chatterbox"


def main() -> int:
    _isolate_import_path()
    parser = argparse.ArgumentParser(description="Isolated Chatterbox TTS runner for Echo.")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--synthesize", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--text", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--options-json", default="{}")
    args = parser.parse_args()

    if args.probe:
        return _emit(_probe(args.device))
    if args.server:
        return _server(args.device)
    if args.synthesize:
        return _emit(_synthesize(args))
    return _emit({"ok": False, "error": "missing_mode"}, code=2)


def _probe(device: str) -> dict[str, Any]:
    missing = [
        name
        for name in ("torch", "torchaudio", "chatterbox")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return {"ok": False, "error": "missing_dependencies:" + ",".join(missing)}
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": _resolved_device(device),
    }


def _synthesize(args: argparse.Namespace) -> dict[str, Any]:
    text = str(args.text or "").strip()
    reference = Path(args.reference)
    output = Path(args.output)
    if not text:
        return {"ok": False, "error": "empty_text"}
    if not reference.exists():
        return {"ok": False, "error": "reference_audio_missing"}

    try:
        import torchaudio
        from chatterbox.tts import ChatterboxTTS
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc)}

    try:
        options = json.loads(args.options_json or "{}")
    except json.JSONDecodeError:
        options = {}
    if not isinstance(options, dict):
        options = {}

    device = _resolved_device(args.device)
    load_started = time.perf_counter()
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc)}
    model_load_ms = (time.perf_counter() - load_started) * 1000.0

    synth_started = time.perf_counter()
    try:
        generate_kwargs = _supported_generate_kwargs(model, options)
        wav = model.generate(text, audio_prompt_path=str(reference), **generate_kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output), wav.cpu() if callable(getattr(wav, "cpu", None)) else wav, int(getattr(model, "sr", 24000)))
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc)}
    synthesis_ms = (time.perf_counter() - synth_started) * 1000.0
    return {
        "ok": True,
        "provider": "chatterbox",
        "model": MODEL_NAME,
        "device": device,
        "model_load_ms": model_load_ms,
        "synthesis_ms": synthesis_ms,
        "sample_rate": int(getattr(model, "sr", 24000)),
        "output": str(output),
    }


def _server(device: str) -> int:
    _configure_cpu_runtime()
    try:
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS
    except Exception as exc:
        _emit_line({"type": "ready", "ok": False, "error": _safe_error(exc)})
        return 1

    resolved_device = _resolved_device(device)
    load_started = time.perf_counter()
    try:
        model = ChatterboxTTS.from_pretrained(device=resolved_device)
    except Exception as exc:
        _emit_line({"type": "ready", "ok": False, "error": _safe_error(exc), "device": resolved_device})
        return 1
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    _emit_line(
        {
            "type": "ready",
            "ok": True,
            "model": MODEL_NAME,
            "device": resolved_device,
            "model_load_ms": model_load_ms,
            "torch_num_threads": int(getattr(torch, "get_num_threads", lambda: 0)() or 0),
        }
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _emit_line({"type": "result", "ok": False, "error": "invalid_json"})
            continue
        if not isinstance(request, dict):
            _emit_line({"type": "result", "ok": False, "error": "invalid_request"})
            continue
        request_id = str(request.get("id") or "")
        if request.get("type") == "shutdown":
            _emit_line({"type": "shutdown", "ok": True, "id": request_id})
            return 0
        if request.get("type") != "synthesize":
            _emit_line({"type": "result", "ok": False, "id": request_id, "error": "unknown_request_type"})
            continue
        payload = _synthesize_with_loaded_model(model, request, resolved_device, model_load_ms)
        payload["id"] = request_id
        _emit_line(payload)
    return 0


def _synthesize_with_loaded_model(model: Any, request: dict[str, Any], device: str, model_load_ms: float) -> dict[str, Any]:
    text = str(request.get("text") or "").strip()
    reference = Path(str(request.get("reference") or ""))
    output = Path(str(request.get("output") or ""))
    options = request.get("options") if isinstance(request.get("options"), dict) else {}
    if not text:
        return {"type": "result", "ok": False, "error": "empty_text"}
    if not reference.exists():
        return {"type": "result", "ok": False, "error": "reference_audio_missing"}
    synth_started = time.perf_counter()
    try:
        import torchaudio

        generate_kwargs = _supported_generate_kwargs(model, options)
        wav = model.generate(text, audio_prompt_path=str(reference), **generate_kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output), wav.cpu() if callable(getattr(wav, "cpu", None)) else wav, int(getattr(model, "sr", 24000)))
    except Exception as exc:
        return {"type": "result", "ok": False, "error": _safe_error(exc)}
    return {
        "type": "result",
        "ok": True,
        "provider": "chatterbox",
        "model": MODEL_NAME,
        "device": device,
        "model_load_ms": model_load_ms,
        "synthesis_ms": (time.perf_counter() - synth_started) * 1000.0,
        "sample_rate": int(getattr(model, "sr", 24000)),
        "output": str(output),
    }


def _supported_generate_kwargs(model: Any, options: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.generate)
    except (TypeError, ValueError):
        return options
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return options
    return {key: value for key, value in options.items() if key in signature.parameters}


def _resolved_device(requested: str) -> str:
    clean = str(requested or "cpu").strip().lower()
    if clean in {"cpu", "cuda", "mps"}:
        return clean
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _safe_error(exc: Exception) -> str:
    return (str(exc or "").strip() or exc.__class__.__name__)[:240]


def _emit(payload: dict[str, Any], *, code: int | None = None) -> int:
    print(json.dumps(payload, ensure_ascii=False))
    if code is not None:
        return code
    return 0 if payload.get("ok") is True else 1


def _emit_line(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _configure_cpu_runtime() -> None:
    thread_limit = os.environ.get("ECHO_TTS_TORCH_THREADS") or os.environ.get("ECHO_TTS_CPU_THREADS") or "2"
    try:
        import torch

        torch.set_num_threads(max(1, int(str(thread_limit).strip())))
    except Exception:
        pass


def _isolate_import_path() -> None:
    script_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != script_dir]


if __name__ == "__main__":
    raise SystemExit(main())
