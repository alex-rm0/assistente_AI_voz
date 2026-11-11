"""Human-reviewed STT comparison; never changes the application's STT settings."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import unicodedata
import wave
from pathlib import Path

import benchmark_stt as bench


CASES = [
    ("short", "Olá, Echo. Estou a testar o microfone."),
    ("short", "Hoje estou a trabalhar em casa."),
    ("names", "Preciso de responder ao João e à Francisca."),
    ("names", "Vamos comparar Samsung, Apple e Google Pixel."),
    ("places", "Gostava de visitar Estocolmo e depois regressar a Coimbra."),
    ("dates", "A reunião é no dia dezanove de junho, às quinze e trinta."),
    ("numbers", "O orçamento máximo é de mil duzentos e cinquenta euros."),
    ("correction", "É para o João. Não, desculpa, é para a Joana."),
    ("correction", "Não quero cancelar a tarefa, quero adiá-la para amanhã."),
    ("command", "Abre o Pixel e minimiza os outros."),
    ("command", "Volta ao email que estávamos a escrever."),
    ("technical", "O Context Observer e o Task Intelligence pertencem ao projeto Echo."),
    ("long", "Quero preparar duas mensagens em paralelo: uma para confirmar a reunião e outra para pedir uma revisão do relatório antes de sexta-feira."),
    ("long", "Para as férias, preferimos natureza e tranquilidade, mas ainda não decidimos o destino nem quanto queremos gastar no alojamento."),
    ("pauses", "Preciso de... deixa-me pensar... rever o documento amanhã."),
    ("quiet", "Estou a falar mais baixo, mas quero que percebas esta frase."),
    ("noise", "Quero marcar uma reunião com o João na próxima quinta-feira."),
    ("noise", "Resume o projeto Echo em quatro pontos, sem inventar informação."),
    ("silence", ""),
    ("silence", ""),
]


def normalized(text: str) -> str:
    # Preserve accents: differences in Portuguese names should remain visible.
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def distance(a, b) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def score(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalized(reference), normalized(hypothesis)
    if not ref:
        return {"wer": None, "cer": None, "silence_hallucination": bool(hypothesis.strip())}
    words = ref.split()
    return {"wer": distance(words, hyp.split()) / len(words),
            "cer": distance(ref, hyp) / len(ref), "silence_hallucination": False}


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(path: Path) -> None:
    if path.exists():
        raise ValueError("Dataset already exists; refusing to overwrite recordings or review.")
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, {"cases": [{"id": f"stt_{i:02}", "category": category,
        "prompt": text, "reference": text, "reviewed": False,
        "audio": f"stt_{i:02}.wav", "seconds": 6 if category == "silence" else 14 if category == "long" else 9}
        for i, (category, text) in enumerate(CASES, 1)]})


def record(path: Path, case_id: str | None) -> None:
    import sounddevice as sd
    import numpy as np

    dataset = json.loads(path.read_text(encoding="utf-8"))
    cases = [c for c in dataset["cases"] if c["id"] == case_id] if case_id else [c for c in dataset["cases"] if not c["reviewed"]]
    if not cases:
        raise ValueError("No matching/unreviewed cases.")
    for case in cases:
        audio = path.parent / case["audio"]
        print(f"\n{case['id']} [{case['category']}] {case['prompt'] or 'Mantém silêncio; no segundo caso podes deixar ruído ambiente.'}")
        if audio.exists():
            print("Áudio existente preservado. Usa outro dataset para voltar a gravar.")
            continue
        input(f"Enter para gravar durante {case['seconds']} segundos; espera por 'A ouvir'.")
        frames = []
        def capture(data, count, info, status):
            frames.append(data.copy())
        with sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=capture):
            print("A ouvir.", flush=True)
            time.sleep(case["seconds"])
        pcm = np.concatenate(frames).tobytes()
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm)
        print("Gravado. Ouve o WAV e revê o campo reference no dataset; só depois marca reviewed=true.")
        save(path, dataset)


def reviewed_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Empty dataset or duplicate case IDs.")
    for case in cases:
        audio = path.parent / case["audio"]
        if case.get("reviewed") is not True or not audio.is_file():
            raise ValueError(f"{case['id']}: audio missing or reference not reviewed.")
        if case["category"] != "silence" and not normalized(case["reference"]):
            raise ValueError(f"{case['id']}: empty speech reference.")
        if case["category"] == "silence" and case["reference"].strip():
            raise ValueError(f"{case['id']}: silence reference must be empty.")
    return cases


def compare(args) -> None:
    cases = reviewed_cases(args.dataset)
    out = args.dataset.parent / time.strftime("comparison_%Y%m%d_%H%M%S")
    out.mkdir(exist_ok=False)
    files = [(args.dataset.parent / c["audio"]).resolve() for c in cases]
    reports = []
    for engine in [bench._engines()[0], bench._engines()[4]]:
        python = args.app_python if engine["venv"] == "app" else args.bench_python
        print(f"A avaliar {engine['name']}...", flush=True)
        report = bench._run_engine(python, engine, files, "pt", out)
        reports.append(report)
    rows = []
    for case, audio in zip(cases, files):
        row = {"id": case["id"], "category": case["category"], "reference": case["reference"],
               "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "models": []}
        for report in reports:
            result = report["results"].get(str(audio))
            error = report["error"] or (result.get("error", "") if result else "missing_result")
            row["models"].append({"name": report["name"], "error": error,
                "text": result["text"] if result else "", "ms": result["ms"] if result else None,
                "scores": score(case["reference"], result["text"]) if result and not error else None})
        rows.append(row)
    summaries = []
    for index, report in enumerate(reports):
        valid = [row for row in rows if row["models"][index]["scores"] is not None]
        speech = [row for row in valid if normalized(row["reference"])]
        words = sum(len(normalized(row["reference"]).split()) for row in speech)
        errors = sum(distance(normalized(row["reference"]).split(),
                              normalized(row["models"][index]["text"]).split()) for row in speech)
        summaries.append({"name": report["name"], "scored_cases": len(valid),
                          "failed_cases": len(rows)-len(valid), "corpus_wer": errors/words if words else None,
                          "silence_hallucinations": sum(row["models"][index]["scores"]["silence_hallucination"] for row in valid)})
    save(out / "report.json", {"cases": rows, "reports": reports, "summary": summaries,
         "note": "Raw STT comparison; app silence/repetition filters are not applied. Keep Ollama loaded externally."})
    lines = ["# STT quality comparison", "", "Referências revistas; resultados brutos, sem filtros da app.",
             "A coluna de revisão humana fica por preencher.", ""]
    for summary in summaries:
        lines.append(f"- Qualidade {summary['name']}: WER={summary['corpus_wer']}; "
                     f"falhas={summary['failed_cases']}; alucinações em silêncio={summary['silence_hallucinations']}")
    for report in reports:
        times = [r["ms"] for r in report["results"].values() if not r["error"]]
        lines.append(f"- {report['name']}: load={report['load_ms']} ms; warmup={report.get('warmup_ms')} ms; "
                     f"median={statistics.median(times) if times else None} ms; device={report.get('confirmed_device')}; "
                     f"GPU before/load/peak={report.get('gpu_before_mb')}/{report.get('gpu_after_load_mb')}/{report.get('gpu_peak_mb')} MiB")
    for row in rows:
        lines.extend(["", f"## {row['id']} ({row['category']})", "", f"Referência: {row['reference'] or '[silêncio]'}"])
        for model in row["models"]:
            lines.extend(["", f"{model['name']} — {model['ms']} ms; {model['scores']}; erro={model['error'] or 'nenhum'}",
                          "", model["text"].replace("<", "&lt;").replace(">", "&gt;")])
        lines.extend(["", "Revisão humana (nomes, números, negações, omissões): ______"])
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "comparison.md")
    if any(m["error"] for row in rows for m in row["models"]):
        raise ValueError("Some model runs failed; see report. Failed runs have no quality score.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "record", "compare"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--app-python", type=Path, default=bench.REPO / ".venv/Scripts/python.exe")
    parser.add_argument("--bench-python", type=Path, default=bench.REPO / ".venv_stt_bench/Scripts/python.exe")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare(args.dataset)
        elif args.action == "record":
            record(args.dataset, args.case_id)
        else:
            compare(args)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
