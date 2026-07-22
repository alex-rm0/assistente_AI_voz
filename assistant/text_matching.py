"""Shared, safe keyword/phrase matching helpers.

Plain substring checks ("erro" in text) false-positive on any word that
happens to contain that sequence of letters as a fragment — "ferro" contains
"erro", "digital" contains "git", "Quero" contains "que". Every detector in
this project that decides something from a fixed keyword/phrase list should
go through here instead of a raw `in` check.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


def normalize_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


@lru_cache(maxsize=512)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # \s+ between words tolerates the input having different internal
    # whitespace than the literal marker ("nao   sabes" still matches "nao sabes").
    escaped = r"\s+".join(re.escape(word) for word in phrase.split())
    return re.compile(rf"\b{escaped}\b")


def contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Word-boundary-safe check for one (already-normalized) marker phrase."""
    return _phrase_pattern(normalize_for_matching(phrase)).search(normalized_text) is not None


def contains_any_phrase(normalized_text: str, phrases) -> bool:
    return any(contains_phrase(normalized_text, phrase) for phrase in phrases)


def find_near_pair_span(normalized_text: str, first_words, second_words, max_gap: int = 25) -> str:
    """Word-boundary-safe proximity check: one of `first_words` followed
    within `max_gap` characters by one of `second_words` (in that order).

    Narrower than "any word from list A and any word from list B anywhere in
    the message" (the pattern that caused the ferro/erro false positive):
    the two concepts still have to be near each other, not just both present
    somewhere in an unrelated sentence. Useful for a subject+state pair like
    "janelas ... detetadas" where a verb ("tens", "estao") can sit between
    them, so an exact adjacent-phrase match would miss legitimate phrasing.
    """
    first_alt = "|".join(re.escape(word) for word in first_words)
    second_alt = "|".join(re.escape(word) for word in second_words)
    pattern = re.compile(rf"\b(?:{first_alt})\b.{{0,{max_gap}}}\b(?:{second_alt})\b")
    match = pattern.search(normalized_text)
    return match.group(0) if match else ""


@lru_cache(maxsize=512)
def _prefix_pattern(phrase: str) -> re.Pattern[str]:
    escaped = r"\s+".join(re.escape(word) for word in phrase.split())
    return re.compile(rf"^{escaped}\b")


def starts_with_phrase(normalized_text: str, phrase: str) -> bool:
    """Word-boundary-safe "starts with" check — `text.startswith("esquece")`
    would wrongly match "esqueceria isso", since that's a real string prefix
    but not the same word."""
    return _prefix_pattern(normalize_for_matching(phrase)).match(normalized_text) is not None


def starts_with_any_phrase(normalized_text: str, phrases) -> bool:
    return any(starts_with_phrase(normalized_text, phrase) for phrase in phrases)


def find_evidence_span(text: str, phrases) -> str:
    """Returns the first marker phrase (as given, not normalized) found in
    `text`, or "" if none match. Used as the human-readable "why" behind a
    tool/context decision — never invented, always one of the phrases
    actually searched for."""
    normalized_text = normalize_for_matching(text)
    for phrase in phrases:
        if contains_phrase(normalized_text, phrase):
            return phrase
    return ""


def find_prefix_evidence_span(text: str, phrases) -> str:
    """Like find_evidence_span, but only counts a phrase at the very start
    of the message (see starts_with_phrase)."""
    normalized_text = normalize_for_matching(text)
    for phrase in phrases:
        if starts_with_phrase(normalized_text, phrase):
            return phrase
    return ""
