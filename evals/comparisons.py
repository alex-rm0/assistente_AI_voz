"""Aggregate comparison across repeated runs (Part 1.3) with explicit
stability classification (Part 7) — never collapse N repetitions into "it
passed", because a model that passes 2/3 times is a different, more useful
signal than one that always passes or always fails.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals import results_store
from evals.assertions import detect_brazilian_portuguese_markers
from evals.results_store import safe_token
from evals.schemas import CaseEvaluation

PASS_STABLE = "PASS_STABLE"
FAIL_STABLE = "FAIL_STABLE"
FLAKY = "FLAKY"
NOT_RUN = "NOT_RUN"


def _case_passed(case_eval: CaseEvaluation) -> bool:
    return all(t.passed for t in case_eval.turn_evaluations)


def _response_signature(case_eval: CaseEvaluation) -> tuple[str, ...]:
    return tuple(t.result.final_response if t.result else "" for t in case_eval.turn_evaluations)


def _path_signature(case_eval: CaseEvaluation) -> tuple[str, ...]:
    return tuple(t.result.selected_path if t.result else "" for t in case_eval.turn_evaluations)


def _forbidden_vocab_per_repetition(case_eval: CaseEvaluation) -> set[str]:
    markers: set[str] = set()
    for turn in case_eval.turn_evaluations:
        if turn.result:
            markers.update(detect_brazilian_portuguese_markers(turn.result.final_response))
    return markers


def compare_repetitions(repetitions: list[list[CaseEvaluation]]) -> dict:
    """repetitions[i] = the full list of CaseEvaluations from repetition i.

    Returns {case_id: {pass_rate, status, responses_differ, paths_differ,
    inconsistent_forbidden_vocabulary}}.
    """
    by_case: dict[str, list[CaseEvaluation]] = {}
    for repetition in repetitions:
        for case_eval in repetition:
            by_case.setdefault(case_eval.case.id, []).append(case_eval)

    per_case: dict[str, dict] = {}
    for case_id, evaluations in by_case.items():
        if not evaluations:
            per_case[case_id] = {
                "case_id": case_id,
                "pass_rate": "0/0",
                "status": NOT_RUN,
                "responses_differ": False,
                "paths_differ": False,
                "inconsistent_forbidden_vocabulary": [],
            }
            continue

        passes = [1 if _case_passed(ce) else 0 for ce in evaluations]
        total = len(passes)
        passed_count = sum(passes)

        if passed_count == total:
            status = PASS_STABLE
        elif passed_count == 0:
            status = FAIL_STABLE
        else:
            status = FLAKY

        responses = {_response_signature(ce) for ce in evaluations}
        paths = {_path_signature(ce) for ce in evaluations}

        vocab_sets = [_forbidden_vocab_per_repetition(ce) for ce in evaluations]
        vocab_union = set().union(*vocab_sets) if vocab_sets else set()
        vocab_intersection = set.intersection(*vocab_sets) if vocab_sets else set()
        inconsistent_vocab = sorted(vocab_union - vocab_intersection)

        per_case[case_id] = {
            "case_id": case_id,
            "pass_rate": f"{passed_count}/{total}",
            "status": status,
            "responses_differ": len(responses) > 1,
            "paths_differ": len(paths) > 1,
            "inconsistent_forbidden_vocabulary": inconsistent_vocab,
        }

    return per_case


def status_counts(per_case: dict) -> dict:
    counts = {PASS_STABLE: 0, FAIL_STABLE: 0, FLAKY: 0, NOT_RUN: 0}
    for entry in per_case.values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return counts


def render_comparison_markdown(provider: str, model: str, per_case: dict, run_dirs: list[str] | None = None) -> str:
    counts = status_counts(per_case)
    lines = [f"# Comparação de repetições — {provider} / {model}", ""]
    lines.append(
        f"- PASS_STABLE: {counts[PASS_STABLE]} | FAIL_STABLE: {counts[FAIL_STABLE]} | "
        f"FLAKY: {counts[FLAKY]} | NOT_RUN: {counts[NOT_RUN]}"
    )
    if counts[FLAKY]:
        lines.append("")
        lines.append(
            "**Atenção**: casos FLAKY não devem ser lidos como \"passou\" só porque uma "
            "repetição teve sucesso — ver secção abaixo."
        )
    lines.append("")
    lines.append("| Caso | Taxa de sucesso | Estado | Respostas diferem | Paths diferem | Vocabulário inconsistente |")
    lines.append("|---|---:|---|:---:|:---:|---|")
    for case_id in sorted(per_case):
        entry = per_case[case_id]
        vocab = ", ".join(entry["inconsistent_forbidden_vocabulary"]) or "-"
        lines.append(
            f"| {case_id} | {entry['pass_rate']} | {entry['status']} | "
            f"{'sim' if entry['responses_differ'] else 'não'} | {'sim' if entry['paths_differ'] else 'não'} | {vocab} |"
        )
    lines.append("")
    if run_dirs:
        lines.append("## Execuções individuais (detalhe preservado, não apagado)")
        lines.append("")
        for run_dir in run_dirs:
            lines.append(f"- `{run_dir}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_comparison(
    provider: str,
    model: str,
    comparison_id: str,
    per_case: dict,
    *,
    suite: str = "",
    repeat: int = 0,
    run_dirs: list[Path] | None = None,
) -> Path:
    directory = results_store.COMPARISONS_DIR / f"{safe_token(provider)}__{safe_token(model)}" / comparison_id
    directory.mkdir(parents=True, exist_ok=True)
    run_dir_strs = [str(p) for p in (run_dirs or [])]
    payload = {
        "comparison_id": comparison_id,
        "provider": provider,
        "model": model,
        "suite": suite,
        "repeat": repeat,
        "run_dirs": run_dir_strs,
        "counts": status_counts(per_case),
        "cases": per_case,
    }
    (directory / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "comparison.md").write_text(
        render_comparison_markdown(provider, model, per_case, run_dir_strs), encoding="utf-8"
    )
    return directory
