from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


MODEL_NAME = "kyutai/pocket-tts"


def main() -> int:
    _isolate_import_path()
    parser = argparse.ArgumentParser(description="Isolated Pocket TTS runner for Echo.")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--synthesize", action="store_true")
    parser.add_argument("--text", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--predefined-voice", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--language", default="english")
    parser.add_argument("--options-json", default="{}")
    args = parser.parse_args()

    if args.probe:
        return _emit(_probe())
    if args.server:
        return _server(args.language)
    if args.synthesize:
        return _emit(_synthesize_once(args))
    return _emit({"ok": False, "error": "missing_mode"}, code=2)


def _probe() -> dict[str, Any]:
    missing = [
        name
        for name in ("torch", "scipy", "pocket_tts")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return {"ok": False, "error": "missing_dependencies:" + ",".join(missing)}
    return {"ok": True, "model": MODEL_NAME, "device": "cpu"}


def _synthesize_once(args: argparse.Namespace) -> dict[str, Any]:
    _configure_cpu_runtime()
    try:
        from pocket_tts import TTSModel
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc)}
    load_started = time.perf_counter()
    try:
        model = TTSModel.load_model(language=_language_name(args.language))
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc)}
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    # Section 3 (Phase 7.3.3): a predefined/catalog voice (e.g. "anna") is a
    # precomputed embedding, not raw-audio cloning -- it works fine even
    # when has_voice_cloning is False (see pocket_tts's own
    # _ORIGINS_OF_PREDEFINED_VOICES branch). Only a RAW REFERENCE-AUDIO
    # request genuinely needs cloning weights, so this blanket rejection
    # must not fire for a predefined-voice request.
    if not str(getattr(args, "predefined_voice", "") or "").strip() and getattr(model, "has_voice_cloning", True) is False:
        return {
            "ok": False,
            "error": "voice_cloning_weights_unavailable",
            "model": MODEL_NAME,
            "device": "cpu",
            "model_load_ms": model_load_ms,
            "has_voice_cloning": False,
        }
    return _synthesize_with_loaded_model(model, {}, args, model_load_ms)


def _server(language: str) -> int:
    _configure_cpu_runtime()
    try:
        import torch
        from pocket_tts import TTSModel
    except Exception as exc:
        _emit_line({"type": "ready", "ok": False, "error": _safe_error(exc)})
        return 1

    load_started = time.perf_counter()
    try:
        model = TTSModel.load_model(language=_language_name(language))
    except Exception as exc:
        _emit_line({"type": "ready", "ok": False, "error": _safe_error(exc), "device": "cpu"})
        return 1
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    has_voice_cloning = getattr(model, "has_voice_cloning", True) is not False
    voice_cache: dict[str, dict[str, Any]] = {}
    _emit_line(
        {
            "type": "ready",
            "ok": True,
            "model": MODEL_NAME,
            "device": "cpu",
            "model_load_ms": model_load_ms,
            "has_voice_cloning": has_voice_cloning,
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
        payload = _synthesize_with_loaded_model(model, voice_cache, _request_namespace(request, language), model_load_ms)
        payload["id"] = request_id
        _emit_line(payload)
    return 0


def _synthesize_with_loaded_model(
    model: Any,
    voice_cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    model_load_ms: float,
) -> dict[str, Any]:
    text = str(args.text or "").strip()
    predefined_voice = str(getattr(args, "predefined_voice", "") or "").strip()
    output = Path(args.output)
    if not text:
        return {"type": "result", "ok": False, "error": "empty_text"}

    if predefined_voice:
        # A predefined/catalog voice (e.g. "anna") is a precomputed
        # embedding -- it needs no reference audio file and works even when
        # has_voice_cloning is False (see main()'s docstring note above).
        cache_key = f"predefined:{predefined_voice}"
        reference_started = time.perf_counter()
        try:
            if cache_key not in voice_cache:
                voice_cache[cache_key] = model.get_state_for_audio_prompt(predefined_voice)
            voice_state = voice_cache[cache_key]
        except Exception as exc:
            return {"type": "result", "ok": False, "error": _safe_error(exc)}
        reference_prepare_ms = (time.perf_counter() - reference_started) * 1000.0
    else:
        reference = Path(args.reference)
        if not reference.exists():
            return {"type": "result", "ok": False, "error": "reference_audio_missing"}
        if getattr(model, "has_voice_cloning", True) is False:
            return {"type": "result", "ok": False, "error": "voice_cloning_weights_unavailable"}

        reference_started = time.perf_counter()
        try:
            cache_key = str(reference.resolve())
            if cache_key not in voice_cache:
                voice_cache[cache_key] = model.get_state_for_audio_prompt(reference)
            voice_state = voice_cache[cache_key]
        except Exception as exc:
            return {"type": "result", "ok": False, "error": _safe_error(exc)}
        reference_prepare_ms = (time.perf_counter() - reference_started) * 1000.0

    synth_started = time.perf_counter()
    try:
        import scipy.io.wavfile

        max_tokens = int(getattr(args, "max_tokens", 80) or 80)
        audio = model.generate_audio(voice_state, text, max_tokens=max_tokens)
        sample_rate = int(getattr(model, "sample_rate", 24000))
        data = audio.detach().cpu().numpy() if callable(getattr(audio, "detach", None)) else audio
        output.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(str(output), sample_rate, data)
    except Exception as exc:
        return {"type": "result", "ok": False, "error": _safe_error(exc)}
    synthesis_ms = (time.perf_counter() - synth_started) * 1000.0
    audio_duration_ms = _audio_duration_ms(data, sample_rate)
    return {
        "type": "result",
        "ok": True,
        "provider": "pocket_tts",
        "model": MODEL_NAME,
        "device": "cpu",
        "model_load_ms": model_load_ms,
        "reference_prepare_ms": reference_prepare_ms,
        "first_audio_ms": synthesis_ms,
        "synthesis_ms": synthesis_ms,
        "audio_duration_ms": audio_duration_ms,
        "sample_rate": sample_rate,
        "reference_cache_hit": reference_prepare_ms < 50.0,
        "predefined_voice": predefined_voice,
        "output": str(output),
    }


def _request_namespace(request: dict[str, Any], language: str) -> argparse.Namespace:
    options = request.get("options") if isinstance(request.get("options"), dict) else {}
    return argparse.Namespace(
        text=str(request.get("text") or ""),
        reference=str(request.get("reference") or ""),
        predefined_voice=str(request.get("predefined_voice") or ""),
        output=str(request.get("output") or ""),
        language=str(request.get("language") or language or "english"),
        max_tokens=int(options.get("max_tokens") or 80),
    )


def _language_name(value: str) -> str:
    clean = str(value or "").strip().lower()
    if clean in {"en", "en-gb", "english", "eng"}:
        return "english"
    return clean or "english"


def _audio_duration_ms(data: Any, sample_rate: int) -> int:
    try:
        length = int(getattr(data, "shape", [len(data)])[0])
        return int((length / max(1, sample_rate)) * 1000)
    except Exception:
        return 0


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
    thread_limit = os.environ.get("ECHO_TTS_POCKET_THREADS") or os.environ.get("ECHO_TTS_CPU_THREADS") or "2"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(thread_limit))
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
