"""Case generation (sections 2.8 and 2.9).

Two independent modes:

- Template mode (default): deterministic phrasing variations of a few known
  patterns (memory-recall follow-ups, search negative controls). Written to
  evals/cases/generated/ with generated=true, review_status=unreviewed —
  NOT included in a normal `run_evals.py` run unless --include-generated
  is passed.
- --from-logs <path>: scans a plaintext log containing this project's
  [TURN TRACE] blocks for turns that looked wrong (exception, empty
  response, an unsupported claim, a memory write that happened during a
  pure question, a tool claimed without being called) and saves each as an
  unassessed candidate in evals/cases/candidates/. These are NOT cases —
  they have no `expected` block, because inferring the "correct" behavior
  automatically is exactly what section 2.9 says not to do.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GENERATED_DIR = Path(__file__).resolve().parent / "cases" / "generated"
CANDIDATES_DIR = Path(__file__).resolve().parent / "cases" / "candidates"


def _write_case(directory: Path, case: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case['id']}.json"
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --- 2.8: template-based generation -----------------------------------------

_MEMORY_RECALL_TEMPLATES = [
    "Lembras-te do meu exame?",
    "Ainda te lembras do meu exame?",
    "O que tens guardado sobre o meu exame?",
    "Qual era mesmo o meu exame?",
    "Não tens isso na memória?",
]

_SEARCH_NEGATIVE_TEMPLATES = [
    "Gosto de pesquisar sobre Picasso.",
    "Estive a pesquisar Picasso ontem.",
    "Quero pesquisar Picasso mais tarde.",
]

_EXAM_SETUP = [
    {
        "fact_type": "academic_event",
        "fields": {"event": "exame", "discipline": "Estratégias Algorítmicas", "date_reference": "para a semana", "status": "upcoming"},
    }
]


def generate_template_variations() -> list[dict]:
    cases: list[dict] = []
    for index, phrasing in enumerate(_MEMORY_RECALL_TEMPLATES, start=1):
        cases.append(
            {
                "id": f"gen_memory_recall_phrasing_{index:03d}",
                "category": "memory",
                "description": f"Variação de formulação de recall: {phrasing!r}",
                "setup": _EXAM_SETUP,
                "clear_conversation_before": True,
                "turns": [
                    {
                        "user": phrasing,
                        "expected": {
                            "response_not_empty": True,
                            "forbid_unsupported_memory_claim": True,
                        },
                    }
                ],
                "tags": ["memory", "generated"],
                "generated": True,
                "review_status": "unreviewed",
                "generator": "template",
                "source_case_id": "memory_exam_recall_001",
            }
        )
    for index, phrasing in enumerate(_SEARCH_NEGATIVE_TEMPLATES, start=1):
        cases.append(
            {
                "id": f"gen_search_negative_{index:03d}",
                "category": "search",
                "description": f"Controlo negativo de pesquisa: {phrasing!r}",
                "setup": [],
                "clear_conversation_before": True,
                "turns": [
                    {
                        "user": phrasing,
                        "expected": {
                            "forbidden_paths": ["RESEARCH_REQUEST"],
                            "must_not_contain": ["Ainda não tenho uma ferramenta de pesquisa ligada."],
                        },
                    }
                ],
                "tags": ["search", "generated", "negative"],
                "generated": True,
                "review_status": "unreviewed",
                "generator": "template",
                "source_case_id": "search_negative_control_004",
            }
        )
    return cases


# --- 2.9: candidates from real logs -----------------------------------------

_TURN_BLOCK_RE = re.compile(r"\[TURN TRACE\](.*?)\[/TURN TRACE\]", re.DOTALL)
_FIELD_RE = re.compile(r"^([a-zA-Z_]+)=(.*)$")


def _parse_turn_blocks(log_text: str) -> list[dict]:
    turns = []
    for block in _TURN_BLOCK_RE.findall(log_text):
        fields: dict[str, str] = {}
        for line in block.strip().splitlines():
            match = _FIELD_RE.match(line)
            if match:
                fields[match.group(1)] = match.group(2)
        if fields:
            turns.append(fields)
    return turns


def _flag_reasons(turn: dict) -> list[str]:
    reasons = []
    if turn.get("exception_type") not in (None, "None", ""):
        reasons.append(f"excecao:{turn.get('exception_type')}")
    if not (turn.get("final_response") or "").strip():
        reasons.append("resposta_vazia")
    if turn.get("unsupported_memory_claim_detected") == "True":
        reasons.append("claim_memoria_sem_grounding")
    if turn.get("unsupported_tool_claim_detected") == "True":
        reasons.append("claim_ferramenta_sem_chamada")
    if turn.get("memory_recall_detected") == "True" and turn.get("memory_write_action") not in (None, "None", ""):
        reasons.append("escrita_de_memoria_durante_pergunta")
    return reasons


def extract_candidates_from_log(log_path: Path) -> list[dict]:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    candidates = []
    for index, turn in enumerate(_parse_turn_blocks(log_text)):
        reasons = _flag_reasons(turn)
        if not reasons:
            continue
        candidates.append(
            {
                "id": f"candidate_{log_path.stem}_{index:03d}",
                "category": "uncategorized",
                "description": f"Extraído de {log_path.name}, sinalizado por: {', '.join(reasons)}",
                "setup": [],
                "clear_conversation_before": True,
                "turns": [{"user": turn.get("user_message", ""), "expected": {}}],
                "tags": ["candidate"] + reasons,
                "generated": True,
                "review_status": "unreviewed",
                "generator": "log_extraction",
                "source_case_id": "",
                "_observed": {
                    "final_response": turn.get("final_response"),
                    "selected_path": turn.get("selected_path"),
                    "response_source": turn.get("response_source"),
                },
            }
        )
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate eval case candidates")
    parser.add_argument("--from-logs", default=None, help="path to a log file containing [TURN TRACE] blocks")
    args = parser.parse_args(argv)

    if args.from_logs:
        log_path = Path(args.from_logs)
        candidates = extract_candidates_from_log(log_path)
        for candidate in candidates:
            path = _write_case(CANDIDATES_DIR, candidate)
            print(f"[generate_cases] candidato: {path}")
        print(f"[generate_cases] {len(candidates)} candidatos extraídos de {log_path}")
        return 0

    cases = generate_template_variations()
    for case in cases:
        path = _write_case(GENERATED_DIR, case)
        print(f"[generate_cases] gerado: {path}")
    print(f"[generate_cases] {len(cases)} casos gerados (generated=true, review_status=unreviewed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
