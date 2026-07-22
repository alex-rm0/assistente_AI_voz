"""Report CONTENT generation (section 2.5) — pure builders, no file I/O.

Where reports get written (evals/results/runs/<date>/<run_name>/, latest/,
comparisons/) is evals/results_store.py's job. This module only turns a list
of CaseEvaluation into the JSON payload / CSV rows / Markdown text.
"""

from __future__ import annotations

import csv
import io
import json

from evals.schemas import CaseEvaluation


def _case_passed(case_eval: CaseEvaluation) -> bool:
    return all(turn.passed for turn in case_eval.turn_evaluations)


def summarize(case_evaluations: list[CaseEvaluation], provider: str, model: str) -> dict:
    total_turns = sum(len(c.turn_evaluations) for c in case_evaluations)
    failed_turns = [t for c in case_evaluations for t in c.turn_evaluations if not t.passed]
    exceptions = [t for c in case_evaluations for t in c.turn_evaluations if t.result and t.result.exception_type]
    latencies = [t.result.latency_ms for c in case_evaluations for t in c.turn_evaluations if t.result]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    by_category: dict[str, dict[str, int]] = {}
    for case_eval in case_evaluations:
        bucket = by_category.setdefault(case_eval.case.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        if _case_passed(case_eval):
            bucket["passed"] += 1

    by_classification: dict[str, int] = {}
    for turn in failed_turns:
        if turn.failure_classification:
            by_classification[turn.failure_classification] = by_classification.get(turn.failure_classification, 0) + 1

    human_review_turns = [
        t for c in case_evaluations for t in c.turn_evaluations if t.human_review_required
    ]

    total_cases = len(case_evaluations)
    passed_cases = sum(1 for c in case_evaluations if _case_passed(c))
    return {
        "provider": provider,
        "model": model,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "total_turns": total_turns,
        "failed_turns": len(failed_turns),
        "exceptions": len(exceptions),
        "average_latency_ms": avg_latency,
        "human_review_required_turns": len(human_review_turns),
        "by_category": by_category,
        "by_classification": by_classification,
    }


def render_json(run_id: str, summary: dict, case_evaluations: list[CaseEvaluation]) -> str:
    payload = {
        "run_id": run_id,
        "summary": summary,
        "cases": [c.to_dict() for c in case_evaluations],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


_CSV_FIELDNAMES = [
    "case_id",
    "category",
    "turn_index",
    "passed",
    "selected_path",
    "response_source",
    "llm_calls",
    "latency_ms",
    "exception_type",
    "failure_classification",
    "human_review_required",
    "review_reasons",
    "user_message",
    "final_response",
]


def render_csv(case_evaluations: list[CaseEvaluation]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDNAMES)
    writer.writeheader()
    for case_eval in case_evaluations:
        for turn in case_eval.turn_evaluations:
            result = turn.result
            writer.writerow(
                {
                    "case_id": case_eval.case.id,
                    "category": case_eval.case.category,
                    "turn_index": turn.turn_index,
                    "passed": turn.passed,
                    "selected_path": result.selected_path if result else "",
                    "response_source": result.response_source if result else "",
                    "llm_calls": result.llm_calls if result else "",
                    "latency_ms": f"{result.latency_ms:.1f}" if result else "",
                    "exception_type": result.exception_type if result else "",
                    "failure_classification": turn.failure_classification,
                    "human_review_required": turn.human_review_required,
                    "review_reasons": ";".join(turn.review_reasons),
                    "user_message": turn.user_message,
                    "final_response": (result.final_response if result else "")[:300],
                }
            )
    return buffer.getvalue()


def _pct(passed: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{(passed / total) * 100:.1f}%"


def render_markdown(run_id: str, summary: dict, case_evaluations: list[CaseEvaluation]) -> str:
    lines: list[str] = []
    lines.append("# Echo Evaluation Report")
    lines.append("")
    lines.append(f"- Run: {run_id}")
    lines.append(f"- Provider: {summary['provider']}")
    lines.append(f"- Modelo: {summary['model']}")
    lines.append(f"- Casos: {summary['total_cases']}")
    lines.append(f"- Casos que passaram: {summary['passed_cases']}")
    lines.append(f"- Casos que falharam: {summary['failed_cases']}")
    lines.append(f"- Turnos totais: {summary['total_turns']}")
    lines.append(f"- Turnos falhados: {summary['failed_turns']}")
    lines.append(f"- Exceções: {summary['exceptions']}")
    lines.append(f"- Latência média: {summary['average_latency_ms']:.0f} ms")
    if summary.get("human_review_required_turns"):
        lines.append(f"- Turnos sinalizados para revisão humana: {summary['human_review_required_turns']}")
    lines.append("")
    lines.append("## Por categoria")
    lines.append("")
    lines.append("| Categoria | Passaram | Total | Taxa |")
    lines.append("|---|---:|---:|---:|")
    for category, counts in sorted(summary["by_category"].items()):
        lines.append(f"| {category} | {counts['passed']} | {counts['total']} | {_pct(counts['passed'], counts['total'])} |")
    lines.append("")

    if summary["by_classification"]:
        lines.append("## Falhas por classificação")
        lines.append("")
        lines.append("| Classificação | Ocorrências |")
        lines.append("|---|---:|")
        for classification, count in sorted(summary["by_classification"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {classification} | {count} |")
        lines.append("")

    failing_cases = [c for c in case_evaluations if not _case_passed(c)]
    if failing_cases:
        lines.append("## Falhas")
        lines.append("")
        for case_eval in failing_cases:
            lines.append(f"### {case_eval.case.id} ({case_eval.case.category})")
            lines.append("")
            lines.append(f"_{case_eval.case.description}_")
            lines.append("")
            for turn in case_eval.turn_evaluations:
                if turn.passed:
                    continue
                result = turn.result
                lines.append(f"**Turno {turn.turn_index}** — falha: `{turn.failure_classification or 'UNKNOWN'}`")
                lines.append("")
                lines.append(f"- input: {turn.user_message!r}")
                lines.append(f"- obtido (selected_path): {result.selected_path if result else 'n/a'}")
                lines.append(f"- resposta: {result.final_response if result else 'n/a'!r}")
                if result and result.exception_type:
                    lines.append(f"- exceção: {result.exception_type}: {result.exception_message}")
                lines.append("- telemetria: " + json.dumps(result.to_dict() if result else {}, ensure_ascii=False))
                lines.append("- assertions falhadas:")
                for assertion in turn.assertions:
                    if not assertion.passed:
                        lines.append(f"  - {assertion.name}: {assertion.detail}")
                lines.append("")

    review_cases = [
        (case_eval, turn)
        for case_eval in case_evaluations
        for turn in case_eval.turn_evaluations
        if turn.human_review_required
    ]
    if review_cases:
        lines.append("## Sinalizados para revisão humana")
        lines.append("")
        lines.append(
            "Estes turnos passaram nas assertions automáticas mas têm sinais que só uma "
            "pessoa consegue avaliar bem (ver Parte 6). Não foram bloqueados."
        )
        lines.append("")
        for case_eval, turn in review_cases:
            result = turn.result
            lines.append(f"- **{case_eval.case.id}** turno {turn.turn_index} — razões: {', '.join(turn.review_reasons)}")
            lines.append(f"  - resposta: {result.final_response if result else 'n/a'!r}")
        lines.append("")

    content = "\n".join(lines) + "\n"
    return content
