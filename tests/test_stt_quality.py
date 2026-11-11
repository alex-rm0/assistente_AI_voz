from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def quality(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("stt_quality", scripts / "stt_quality.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_word_errors_measure_substitution_deletion_and_insertion(quality):
    assert quality.score("o João vem amanhã", "o João chega amanhã")["wer"] == .25
    assert quality.score("o João vem amanhã", "o João amanhã")["wer"] == .25
    assert quality.score("o João vem amanhã", "o João vem só amanhã")["wer"] == .25
    assert quality.score("Olá, João!", "olá joão")["wer"] == 0
    assert quality.score("João", "Joana")["wer"] == 1


def test_silence_is_not_scored_as_perfect_speech(quality):
    assert quality.score("", "Obrigado por assistir.") == {
        "wer": None, "cer": None, "silence_hallucination": True}
    assert not quality.score("", "")["silence_hallucination"]
    assert quality.score("não cancela", "cancela")["wer"] == .5


def test_prepare_never_overwrites_review_and_requires_real_audio(quality, tmp_path):
    path = tmp_path / "dataset.json"
    quality.prepare(path)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    assert len(dataset["cases"]) == 20
    assert all(c["reviewed"] is False for c in dataset["cases"])
    with pytest.raises(ValueError, match="overwrite"):
        quality.prepare(path)
    with pytest.raises(ValueError, match="not reviewed"):
        quality.reviewed_cases(path)


def test_comparison_uses_identical_audio_and_reviewed_reference(quality, tmp_path, monkeypatch):
    path = tmp_path / "dataset.json"
    (tmp_path / "sample.wav").write_bytes(b"audio")
    quality.save(path, {"cases": [{"id": "sample", "category": "names", "reference": "Olá João",
                                "audio": "sample.wav", "reviewed": True}]})
    calls = []
    def run(python, engine, files, language, work):
        calls.append((files, language))
        return {"name": engine["name"], "error": "", "load_ms": 100,
                "results": {str(files[0]): {"text": "Olá João", "ms": 50, "error": ""}}}
    monkeypatch.setattr(quality.bench, "_run_engine", run)
    from argparse import Namespace
    quality.compare(Namespace(dataset=path, app_python=Path("app"), bench_python=Path("bench")))
    assert len(calls) == 2 and calls[0] == calls[1]
    result = json.loads(next(tmp_path.glob("comparison_*/report.json")).read_text(encoding="utf-8"))
    assert all(m["scores"]["wer"] == 0 for m in result["cases"][0]["models"])


def test_failed_transcription_never_gets_quality_score(quality, tmp_path, monkeypatch):
    path = tmp_path / "dataset.json"
    (tmp_path / "sample.wav").write_bytes(b"audio")
    quality.save(path, {"cases": [{"id": "sample", "category": "names", "reference": "João",
                                "audio": "sample.wav", "reviewed": True}]})
    monkeypatch.setattr(quality.bench, "_run_engine", lambda *a, **kw: {
        "name": "failed", "load_ms": None, "error": "CUDA unavailable", "results": {}})
    from argparse import Namespace
    with pytest.raises(ValueError, match="failed"):
        quality.compare(Namespace(dataset=path, app_python=Path("app"), bench_python=Path("bench")))
    result = json.loads(next(tmp_path.glob("comparison_*/report.json")).read_text(encoding="utf-8"))
    assert all(m["scores"] is None for m in result["cases"][0]["models"])
