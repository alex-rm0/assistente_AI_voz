"""Preliminary automatic classification of a failed turn (section 2.6).

Best-effort only — the spec is explicit that this classification can be
wrong and is meant to be reviewed manually afterwards. It looks at WHICH
assertion(s) failed, not at the response text itself, so it stays cheap and
deterministic.
"""

from __future__ import annotations

from evals.schemas import AssertionOutcome, TurnResult

INFRASTRUCTURE = "INFRASTRUCTURE"
ROUTING = "ROUTING"
MEMORY = "MEMORY"
TOOL = "TOOL"
MODEL_OUTPUT = "MODEL_OUTPUT"
LANGUAGE = "LANGUAGE"
PRESENTATION = "PRESENTATION"
UNKNOWN = "UNKNOWN"

_NAME_TO_CATEGORY = {
    "expected_exception": INFRASTRUCTURE,
    "no_unexpected_exception": INFRASTRUCTURE,
    "selected_path": ROUTING,
    "forbidden_paths": ROUTING,
    "expected_tools": TOOL,
    "forbidden_tools": TOOL,
    "no_tools_used": TOOL,
    "no_unsupported_tool_claim": TOOL,
    "forbidden_contexts": ROUTING,
    "expected_memory_ids": MEMORY,
    "memory_write_action": MEMORY,
    "forbid_memory_write": MEMORY,
    "no_unsupported_memory_claim": MEMORY,
    "response_grounded": MEMORY,
    "no_brazilian_portuguese": LANGUAGE,
    "max_questions": PRESENTATION,
    "response_not_empty": MODEL_OUTPUT,
    "max_latency_ms": UNKNOWN,
    "llm_calls_min": ROUTING,
    "llm_calls_max": ROUTING,
}

# Priority when several assertions fail on the same turn: the most
# actionable/specific cause wins over a vaguer one.
_PRIORITY = (INFRASTRUCTURE, ROUTING, TOOL, MEMORY, LANGUAGE, PRESENTATION, MODEL_OUTPUT, UNKNOWN)


def classify_failure(assertions: list[AssertionOutcome], result: TurnResult | None) -> str:
    failed = [a for a in assertions if not a.passed]
    if not failed:
        return ""

    if result is not None and result.exception_type is not None:
        return INFRASTRUCTURE

    categories: set[str] = set()
    for outcome in failed:
        name = outcome.name.split(":", 1)[0]
        if name.startswith("must_contain") or name.startswith("must_not_contain"):
            categories.add(MODEL_OUTPUT)
            continue
        categories.add(_NAME_TO_CATEGORY.get(name, UNKNOWN))

    for category in _PRIORITY:
        if category in categories:
            return category
    return UNKNOWN
