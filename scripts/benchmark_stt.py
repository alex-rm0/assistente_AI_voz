"""STT benchmark over saved voice captures (diagnostics only -- the app keeps
using openai-whisper medium; nothing here changes it).

Compares, on every data/debug/voice_captures/<stamp>.wav (with the app's own
transcript in <stamp>.txt as the reference):

  1. openai-whisper medium            (current app STT, CPU)
  2. faster-whisper medium int8       (CPU)
  3. faster-whisper large-v3-turbo    (CUDA float16 if CTranslate2 sees a GPU,
                                       otherwise CPU int8)

with the app's decoding settings: language "pt", the app's Portuguese
initial_prompt, temperature 0, greedy decoding, no condition_on_previous_text.

Captures: set "voice": {"debug_save_captures": true} in config/settings.json,
use voice mode for a while, then set it back to false.

faster-whisper is NOT an app dependency. Install it in its own venv once:

    .venv\\Scripts\\python.exe -m venv .venv_stt_bench
    .venv_stt_bench\\Scripts\\python.exe -m pip install faster-whisper

(GPU for faster-whisper also needs the CUDA 12 cuBLAS/cuDNN 9 DLLs, e.g.
`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` in that venv; without
them the large-v3-turbo run falls back to CPU int8 and says so.)

Run (from the repo root):

    .venv\\Scripts\\python.exe scripts\\benchmark_stt.py
    .venv\\Scripts\\python.exe scripts\\benchmark_stt.py --limit 10 --out stt_benchmark.json
    .venv\\Scripts\\python.exe scripts\\benchmark_stt.py --engines 1,2   (skip large-v3-turbo)

Models are downloaded on first use into .hf_cache/ (faster-whisper) and the
usual Whisper cache (openai-whisper). Each engine runs in a subprocess with
its own interpreter, loads its model once, and reports load time and
per-file transcription time.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import statistics
import subprocess
import sys
import time
import unicodedata
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Copy of assistant.voice_input.DEFAULT_INITIAL_PROMPT (kept in sync by hand;
# the faster-whisper venv cannot import the app).
FALLBACK_PROMPT = (
    "Transcricao em portugues de Portugal europeu. "
    "Contexto possivel: viagens, datas, tarefas, pesquisa, orcamento, alojamento, "
    "Portugal, Estocolmo, projetos pessoais e conversa com o Echo."
)


# --- worker side (runs inside the engine's own interpreter) -----------------


def _worker(args: argparse.Namespace) -> int:
    files = [line for line in Path(args.files).read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    started = time.perf_counter()
    if args.engine == "openai-whisper":
        import whisper

        model = whisper.load_model(args.model, device="cpu")
        device, compute_type = "cpu", "float32"

        def run(path: str) -> str:
            result = model.transcribe(
                path, language=args.language, initial_prompt=prompt, task="transcribe",
                temperature=0.0, condition_on_previous_text=False, fp16=False,
            )
            return str(result.get("text", "")).strip()
    else:
        os.environ.setdefault("HF_HOME", str(REPO / ".hf_cache"))
        from faster_whisper import WhisperModel

        device, compute_type = args.device, args.compute_type
        fallback = ""
        if device == "auto":
            try:
                import ctranslate2

                has_cuda = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                has_cuda = False
            device, compute_type = ("cuda", "float16") if has_cuda else ("cpu", "int8")
        try:
            model = WhisperModel(args.model, device=device, compute_type=compute_type)
        except Exception as exc:  # e.g. CUDA libraries missing
            if device == "cpu":
                raise
            fallback = f"{device}/{compute_type} failed: {type(exc).__name__}: {str(exc)[:160]}"
            device, compute_type = "cpu", "int8"
            model = WhisperModel(args.model, device=device, compute_type=compute_type)
        if fallback:
            print("FALLBACK " + json.dumps({"reason": fallback}), flush=True)

        def run(path: str) -> str:
            segments, _info = model.transcribe(
                path, language=args.language, initial_prompt=prompt, task="transcribe",
                temperature=0.0, condition_on_previous_text=False, beam_size=1, vad_filter=False,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

    load_ms = (time.perf_counter() - started) * 1000.0
    print("LOADED " + json.dumps({"load_ms": load_ms, "device": device, "compute_type": compute_type}), flush=True)
    for path in files:
        t = time.perf_counter()
        try:
            text, error = run(path), ""
        except Exception as exc:
            text, error = "", f"{type(exc).__name__}: {str(exc)[:160]}"
        print("RESULT " + json.dumps({"file": path, "ms": (time.perf_counter() - t) * 1000.0, "text": text, "error": error}, ensure_ascii=False), flush=True)
    return 0


# --- orchestrator ------------------------------------------------------------


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize(a).split(), _normalize(b).split()).ratio()


def _duration_s(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:
        return 0.0


def _app_prompt() -> str:
    try:
        sys.path.insert(0, str(REPO))
        from assistant.voice_input import DEFAULT_INITIAL_PROMPT

        return DEFAULT_INITIAL_PROMPT
    except Exception:
        return FALLBACK_PROMPT


def _run_engine(python: Path, engine: dict, files: list[Path], language: str, work: Path) -> dict:
    files_list = work / "files.txt"
    files_list.write_text("\n".join(str(path) for path in files), encoding="utf-8")
    prompt_file = work / "prompt.txt"
    prompt_file.write_text(_app_prompt(), encoding="utf-8")
    command = [
        str(python), str(Path(__file__).resolve()), "--worker",
        "--engine", engine["engine"], "--model", engine["model"],
        "--device", engine["device"], "--compute-type", engine["compute_type"],
        "--language", language, "--files", str(files_list), "--prompt-file", str(prompt_file),
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    report = {"name": engine["name"], "python": str(python), "results": {}, "load_ms": None, "error": "", "fallback": ""}
    if not python.exists():
        report["error"] = f"interpreter not found: {python}"
        return report
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(REPO))
    for line in completed.stdout.splitlines():
        tag, _, payload = line.partition(" ")
        if tag == "LOADED":
            data = json.loads(payload)
            report.update(load_ms=data["load_ms"], device=data["device"], compute_type=data["compute_type"])
        elif tag == "FALLBACK":
            report["fallback"] = json.loads(payload)["reason"]
        elif tag == "RESULT":
            data = json.loads(payload)
            report["results"][data["file"]] = data
    if completed.returncode != 0 and report["load_ms"] is None:
        report["error"] = (completed.stderr.strip().splitlines() or ["failed"])[-1][:300]
    return report


def _engines() -> list[dict]:
    return [
        {"name": "openai-whisper medium", "engine": "openai-whisper", "model": "medium", "device": "cpu", "compute_type": "float32", "venv": "app"},
        {"name": "faster-whisper medium int8", "engine": "faster-whisper", "model": "medium", "device": "cpu", "compute_type": "int8", "venv": "bench"},
        {"name": "faster-whisper large-v3-turbo", "engine": "faster-whisper", "model": "large-v3-turbo", "device": "auto", "compute_type": "auto", "venv": "bench"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--compute-type", default="int8", help=argparse.SUPPRESS)
    parser.add_argument("--files", help=argparse.SUPPRESS)
    parser.add_argument("--prompt-file", help=argparse.SUPPRESS)
    parser.add_argument("--captures", type=Path, default=REPO / "data" / "debug" / "voice_captures")
    parser.add_argument("--app-python", type=Path, default=REPO / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--bench-python", type=Path, default=REPO / ".venv_stt_bench" / "Scripts" / "python.exe")
    parser.add_argument("--language", default="pt")
    parser.add_argument("--limit", type=int, default=0, help="only the N most recent captures (0 = all)")
    parser.add_argument("--out", type=Path, help="also write the full report as JSON")
    parser.add_argument("--engines", default="1,2,3", help="which engines to run, by number (default: 1,2,3)")
    args = parser.parse_args()
    if args.worker:
        return _worker(args)

    wavs = sorted(path for path in args.captures.glob("*.wav") if path.with_suffix(".txt").exists())
    if args.limit:
        wavs = wavs[-args.limit:]
    if not wavs:
        print(f"Sem capturas em {args.captures}. Liga voice.debug_save_captures e usa o modo de voz.")
        return 1
    references = {str(path): path.with_suffix(".txt").read_text(encoding="utf-8").strip() for path in wavs}
    audio_s = sum(_duration_s(path) for path in wavs)
    work = args.captures / ".benchmark_tmp"
    work.mkdir(exist_ok=True)

    selected = {int(item) for item in args.engines.split(",") if item.strip()}
    reports = []
    for number, engine in enumerate(_engines(), start=1):
        if number not in selected:
            continue
        python = args.app_python if engine["venv"] == "app" else args.bench_python
        print(f"== {engine['name']} ({python})", flush=True)
        reports.append(_run_engine(python, engine, wavs, args.language, work))

    print(f"\ncapturas={len(wavs)} audio_total_s={audio_s:.1f} (referência = transcrição guardada pela app)\n")
    print(f"{'motor':34} {'dispositivo':14} {'load_ms':>8} {'média_ms':>9} {'mediana_ms':>10} {'RTF':>6} {'semelhança':>10}")
    for report in reports:
        if report["error"]:
            print(f"{report['name']:34} ERRO: {report['error']}")
            continue
        times = [r["ms"] for r in report["results"].values() if not r["error"]]
        sims = [_similarity(references[f], r["text"]) for f, r in report["results"].items() if not r["error"]]
        rtf = (sum(times) / 1000.0) / audio_s if audio_s and times else 0.0
        device = f"{report.get('device')}/{report.get('compute_type')}"
        print(
            f"{report['name']:34} {device:14} {report['load_ms']:>8.0f} "
            f"{statistics.mean(times) if times else 0:>9.0f} {statistics.median(times) if times else 0:>10.0f} "
            f"{rtf:>6.2f} {statistics.mean(sims) if sims else 0:>10.2f}"
        )
        if report["fallback"]:
            print(f"{'':34} (fallback para CPU: {report['fallback']})")

    print("\nPor captura:")
    for path in wavs:
        key = str(path)
        print(f"\n{path.name}  ({_duration_s(path):.1f}s)\n  referência: {references[key]}")
        for report in reports:
            result = report["results"].get(key)
            if result:
                label = result["error"] or result["text"]
                print(f"  {report['name']:30} {result['ms']:>7.0f} ms  {label}")

    if args.out:
        args.out.write_text(json.dumps({"references": references, "audio_s": audio_s, "reports": reports}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nrelatório: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
