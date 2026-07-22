"""Part 6: flag turns for human review — never block automatically.

These are best-effort heuristics, not a quality gate. A turn that trips one
of these still counts as PASSED if its assertions passed; `human_review_required`
and `review_reasons` are extra signal for a person to look at, nothing more.
"""

from __future__ import annotations

import re

from evals.assertions import _has_word, _normalize
from evals.schemas import TurnResult

_UNSUPPORTED_ENTITY_PHRASES = (
    "e um cliente importante",
    "e uma empresa conhecida",
    "e um parceiro de confianca",
    "e um projeto prioritario",
)

_ENTHUSIASM_PATTERN = re.compile(r"!{2,}")
_ENTHUSIASM_WORDS = ("otimo", "fantastico", "incrivel", "excelente noticia", "que maravilha")


def _looks_unsupported_entity_claim(normalized_response: str, result: TurnResult) -> bool:
    if result.grounding_sources:
        return False
    return any(_has_word(normalized_response, phrase.split()[0]) and phrase in normalized_response for phrase in _UNSUPPORTED_ENTITY_PHRASES)


def _looks_excessively_enthusiastic(normalized_response: str) -> bool:
    if _ENTHUSIASM_PATTERN.search(normalized_response):
        return True
    return any(_has_word(normalized_response, word.split()[0]) and word in normalized_response for word in _ENTHUSIASM_WORDS)


def _repeats_a_question_already_answered(result: TurnResult, prior_final_responses: list[str]) -> bool:
    if "?" not in result.final_response:
        return False
    normalized_current = _normalize(result.final_response)
    for prior in prior_final_responses:
        if not prior.strip():
            continue
        normalized_prior = _normalize(prior)
        # Cheap overlap check: same question stem repeated verbatim-ish.
        if normalized_current[:40] and normalized_current[:40] in normalized_prior:
            return True
    return False


def detect_review_reasons(result: TurnResult, prior_final_responses: list[str] | None = None) -> list[str]:
    """Returns a list of reason codes, empty if nothing looks worth a human's time."""
    prior_final_responses = prior_final_responses or []
    normalized_response = _normalize(result.final_response)
    reasons: list[str] = []

    if result.unsupported_memory_claim_detected:
        reasons.append("unsupported_memory_claim")
    if result.unsupported_tool_claim_detected:
        reasons.append("unsupported_tool_claim")
    if _looks_unsupported_entity_claim(normalized_response, result):
        reasons.append("unsupported_entity_claim")
    if _looks_excessively_enthusiastic(normalized_response):
        reasons.append("excessive_enthusiasm")
    if _repeats_a_question_already_answered(result, prior_final_responses):
        reasons.append("question_already_answered_in_history")

    return reasons
