"""Build a side-by-side markdown report from two eval report.json files.

This script does not call any model. It only compares existing eval outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


DEFAULT_SUBSET = [
    "model_quality_email_001",
    "model_quality_rewrite_ptpt_002",
    "model_quality_technical_explanation_003",
    "model_quality_emotional_support_004",
    "model_quality_planning_005",
    "model_quality_comparison_006",
    "model_quality_summary_007",
    "model_quality_useful_ambiguity_008",
    "model_quality_multiturn_continuity_009",
    "model_quality_honesty_limits_010",
]

PTPT_VIOLATION_MARKERS = (
    "aplicativos",
    "aplicativo",
    "tela",
    "arquivos",
    "arquivo",
    "acessar",
    "aprendizado",
    "voce",
    "revisar",
    "okay",
)
ASSUMPTION_MARKERS = ("parece que", "provavelmente", "assumo", "deduzo", "imagino")
PIPELINE_ASSERTIONS = {"memory_write_action", "forbid_memory_write"}
HUMAN_REVIEW_METRICS = (
    "naturalidade",
    "utilidade",
    "compreensao do contexto",
    "portugues de Portugal",
    "concisao",
    "preferencia final",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two Echo model eval reports side by side.")
    parser.add_argument("--left", required=True, help="Path to the first report.json")
    parser.add_argument("--right", required=True, help="Path to the second report.json")
    parser.add_argument("--output", default="", help="Optional markdown output path")
    parser.add_argument("--subset-file", default="", help="Optional file with one case id per line")
    args = parser.parse_args()

    subset = _load_subset(args.subset_file)
    left = _load_report(Path(args.left))
    right = _load_report(Path(args.right))
    markdown = render_side_by_side(left, right, subset)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


def _load_subset(path: str) -> list[str]:
    if not path:
        return list(DEFAULT_SUBSET)
    values = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return values or list(DEFAULT_SUBSET)


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_side_by_side(left: dict[str, Any], right: dict[str, Any], subset: list[str]) -> str:
    left_cases = _index_cases(left)
    right_cases = _index_cases(right)
    _validate_same_cases(left_cases, right_cases, subset)
    _validate_same_inputs(left, right, subset)
    _validate_model_quality_llm_calls(left_cases, right_cases, subset)
    left_label = _label(left)
    right_label = _label(right)
    lines = [
        "# Echo Model Comparison",
        "",
        f"- Esquerda: {left_label}",
        f"- Direita: {right_label}",
        "",
        f"- Casos comparados: {len(subset)}",
        "",
        "## Resumo automatico",
        "",
        "| Metrica | Llama | Claude | Empates |",
        "|---|---:|---:|---:|",
    ]
    automatic = _automatic_summary(left, right, subset)
    for metric, values in automatic.items():
        lines.append(f"| {metric} | {values['left']} | {values['right']} | {values['ties']} |")
    lines.extend(
        [
            "",
            "## Resultados por caso",
            "",
        "| Caso | Modelo | Passou | Latencia | Tokens | Custo | PT-PT | Perguntas | Suposicoes | Falhas do modelo | Falhas do pipeline | Resposta |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---|---|---|",
        ]
    )
    for case_id in subset:
        for report, indexed in ((left, left_cases), (right, right_cases)):
            row = indexed.get(case_id)
            if row is None:
                lines.append(f"| {case_id} | {_label(report)} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | em falta | em falta | _em falta_ |")
                continue
            result = row.get("result") or {}
            response = str(result.get("final_response") or "")
            input_tokens = result.get("input_tokens")
            output_tokens = result.get("output_tokens")
            token_label = _token_label(input_tokens, output_tokens)
            cost = float(result.get("estimated_cost_usd") or 0.0)
            lines.append(
                "| "
                + " | ".join(
                    [
                        case_id,
                        _label(report),
                        "sim" if row.get("passed") else "nao",
                        f"{float(result.get('latency_ms') or 0.0):.0f} ms",
                        token_label,
                        f"${cost:.6f}",
                        "ok" if _ptpt_ok(response) else "rever",
                        str(response.count("?")),
                        _assumptions(response),
                        _escape_cell(", ".join(_model_failed_assertions(row)) or "nenhuma"),
                        _escape_cell(", ".join(_pipeline_failed_assertions(row)) or "nenhuma"),
                        _escape_cell(response[:260]),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Revisao humana", ""])
    for case_id in subset:
        lines.extend(_human_review_block(case_id, left_label, right_label))

    lines.extend(
        [
            "## Resumo final de revisao humana",
            "",
            "| Metrica | Llama | Claude | Empates |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in HUMAN_REVIEW_METRICS:
        lines.append(f"| {metric} |  |  |  |")
    return "\n".join(lines) + "\n"


def _automatic_summary(left: dict[str, Any], right: dict[str, Any], subset: list[str]) -> dict[str, dict[str, object]]:
    left_rows = _rows_for(left, subset)
    right_rows = _rows_for(right, subset)
    return {
        "casos passados": _left_right_ties(sum(1 for r in left_rows if r.get("passed")), sum(1 for r in right_rows if r.get("passed"))),
        "casos falhados": _left_right_ties(sum(1 for r in left_rows if not r.get("passed")), sum(1 for r in right_rows if not r.get("passed"))),
        "latencia media ms": _metric_values(_average_latency(left_rows), _average_latency(right_rows), lower_is_better=True),
        "input tokens": _metric_values(_sum_result(left_rows, "input_tokens"), _sum_result(right_rows, "input_tokens"), lower_is_better=True),
        "output tokens": _metric_values(_sum_result(left_rows, "output_tokens"), _sum_result(right_rows, "output_tokens"), lower_is_better=True),
        "custo total USD": _metric_values(_sum_result(left_rows, "estimated_cost_usd"), _sum_result(right_rows, "estimated_cost_usd"), lower_is_better=True),
        "falhas do modelo": _metric_values(_count_model_failures(left_rows), _count_model_failures(right_rows), lower_is_better=True),
        "perguntas desnecessarias": _metric_values(_count_model_failed(left_rows, "max_questions"), _count_model_failed(right_rows, "max_questions"), lower_is_better=True),
        "alegacoes nao fundamentadas": _metric_values(
            _count_failed(left_rows, "no_ungrounded_computer_observation"),
            _count_failed(right_rows, "no_ungrounded_computer_observation"),
            lower_is_better=True,
        ),
        "falhas partilhadas do pipeline": _left_right_ties(_count_shared_pipeline_failures(left_rows, right_rows), ""),
        "alteracoes de memoria inesperadas": _metric_values(
            _count_failed(left_rows, "memory_write_action") + _count_failed(left_rows, "forbid_memory_write"),
            _count_failed(right_rows, "memory_write_action") + _count_failed(right_rows, "forbid_memory_write"),
            lower_is_better=True,
        ),
        "violacoes de PT-PT": _metric_values(_count_model_failed(left_rows, "no_brazilian_portuguese"), _count_model_failed(right_rows, "no_brazilian_portuguese"), lower_is_better=True),
    }


def _left_right_ties(left: object, right: object) -> dict[str, object]:
    return {"left": left, "right": right, "ties": ""}


def _metric_values(left: float, right: float, *, lower_is_better: bool) -> dict[str, object]:
    if left == right:
        return {"left": _format_number(left), "right": _format_number(right), "ties": 1}
    return {"left": _format_number(left), "right": _format_number(right), "ties": 0}


def _format_number(value: float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(int(value))


def _rows_for(report: dict[str, Any], subset: list[str]) -> list[dict[str, Any]]:
    indexed = _index_cases(report)
    return [indexed[case_id] for case_id in subset if case_id in indexed]


def _average_latency(rows: list[dict[str, Any]]) -> float:
    values = [float((row.get("result") or {}).get("latency_ms") or 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _sum_result(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = (row.get("result") or {}).get(key)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def _count_failed(rows: list[dict[str, Any]], assertion_name: str) -> int:
    return sum(1 for row in rows if assertion_name in _failed_assertions(row))


def _count_model_failed(rows: list[dict[str, Any]], assertion_name: str) -> int:
    return sum(1 for row in rows if assertion_name in _model_failed_assertions(row))


def _count_model_failures(rows: list[dict[str, Any]]) -> int:
    return sum(len(_model_failed_assertions(row)) for row in rows)


def _count_shared_pipeline_failures(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> int:
    count = 0
    for left, right in zip(left_rows, right_rows):
        if set(_pipeline_failed_assertions(left)) & set(_pipeline_failed_assertions(right)):
            count += 1
    return count


def _failed_assertions(row: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for assertion in row.get("assertions") or []:
        if not assertion.get("passed"):
            failed.append(str(assertion.get("name") or ""))
    return failed


def _pipeline_failed_assertions(row: dict[str, Any]) -> list[str]:
    return [name for name in _failed_assertions(row) if name in PIPELINE_ASSERTIONS]


def _model_failed_assertions(row: dict[str, Any]) -> list[str]:
    return [name for name in _failed_assertions(row) if name not in PIPELINE_ASSERTIONS]


def _human_review_block(case_id: str, left_label: str, right_label: str) -> list[str]:
    lines = [f"### {case_id}", ""]
    for metric in HUMAN_REVIEW_METRICS:
        lines.append(f"- {metric}: [ ] {left_label} / [ ] {right_label} / [ ] empate")
    lines.append("- notas:")
    lines.append("")
    return lines


def _index_cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for case in report.get("cases", []):
        turns = case.get("turns") or []
        if turns:
            indexed[str(case.get("id"))] = turns[-1]
    return indexed


def _validate_same_cases(left_cases: dict[str, dict[str, Any]], right_cases: dict[str, dict[str, Any]], subset: list[str]) -> None:
    expected = set(subset)
    missing_left = sorted(expected - set(left_cases))
    missing_right = sorted(expected - set(right_cases))
    extras_left = sorted(set(left_cases) - expected)
    extras_right = sorted(set(right_cases) - expected)
    if missing_left or missing_right or extras_left or extras_right:
        raise ValueError(
            "Os relatórios não contêm exactamente os mesmos casos esperados. "
            f"missing_left={missing_left}; missing_right={missing_right}; "
            f"extras_left={extras_left}; extras_right={extras_right}."
        )


def _validate_same_inputs(left: dict[str, Any], right: dict[str, Any], subset: list[str]) -> None:
    left_inputs = _case_inputs(left)
    right_inputs = _case_inputs(right)
    mismatches = []
    for case_id in subset:
        if left_inputs.get(case_id) != right_inputs.get(case_id):
            mismatches.append(
                f"{case_id}: left={left_inputs.get(case_id)!r}; right={right_inputs.get(case_id)!r}"
            )
    if mismatches:
        raise ValueError("Os relatórios não usaram exactamente os mesmos inputs. " + "; ".join(mismatches))


def _case_inputs(report: dict[str, Any]) -> dict[str, list[str]]:
    indexed: dict[str, list[str]] = {}
    for case in report.get("cases", []):
        indexed[str(case.get("id"))] = [str(turn.get("user_message") or "") for turn in case.get("turns") or []]
    return indexed


def _validate_model_quality_llm_calls(
    left_cases: dict[str, dict[str, Any]], right_cases: dict[str, dict[str, Any]], subset: list[str]
) -> None:
    if not any(case_id.startswith("model_quality_") for case_id in subset):
        return

    failures: list[str] = []
    for label, cases in (("left", left_cases), ("right", right_cases)):
        for case_id in subset:
            if not case_id.startswith("model_quality_"):
                continue
            row = cases.get(case_id)
            result = (row or {}).get("result") or {}
            llm_calls = int(result.get("llm_calls") or 0)
            if llm_calls != 1:
                failures.append(f"{label}:{case_id}:llm_calls={llm_calls}")

    if failures:
        raise ValueError(
            "A comparação model_quality exige exactamente uma chamada LLM por caso. "
            + "; ".join(failures)
        )


def _label(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    provider = str(summary.get("provider") or "?")
    model = str(summary.get("model") or "?")
    return f"{provider}/{model}"


def _token_label(input_tokens: object, output_tokens: object) -> str:
    left = "?" if input_tokens is None else str(input_tokens)
    right = "?" if output_tokens is None else str(output_tokens)
    return f"{left}/{right}"


def _ptpt_ok(text: str) -> bool:
    normalized = _normalize(text)
    return not any(re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in PTPT_VIOLATION_MARKERS)


def _assumptions(text: str) -> str:
    normalized = _normalize(text)
    found = [marker for marker in ASSUMPTION_MARKERS if marker in normalized]
    return ", ".join(found) if found else "nao"


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(without_marks.split())


def _escape_cell(text: str) -> str:
    compact = " ".join(str(text or "").split())
    return compact.replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
