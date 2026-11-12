"""Keeps "Sir" occasional in en-GB replies.

The style guide asks the model for a very occasional "Sir"; a small model
adds it to half its replies. This is applied when a reply is finalised: one
reply in SIR_EVERY may keep a single "Sir", the others lose it. Only the
vocative is removed. A title ("Sir Isaac Newton") or a letter opening
("Dear Sir,") is left as written.
"""

from __future__ import annotations

import re

from assistant.response_language import ENGLISH_GB, normalize_response_language

SIR_EVERY = 5
_END = r"(?=\s*(?:[,.!?;:…]|$))"
# In order: "Sir, ..." at the start of a sentence, ", Sir" before punctuation, " Sir" before punctuation.
_LEADING = re.compile(r"(^|[.!?…]\s+|\n\s*)sir\s*[,:–—-]\s*(\w?)", re.IGNORECASE)
_AFTER_COMMA = re.compile(rf"\s*,\s*sir\b{_END}", re.IGNORECASE | re.MULTILINE)
_BARE = re.compile(rf"(?<!\bdear)[^\S\n]+sir\b{_END}", re.IGNORECASE | re.MULTILINE)
_ANY = re.compile(rf"{_LEADING.pattern}|{_AFTER_COMMA.pattern}|{_BARE.pattern}", re.IGNORECASE | re.MULTILINE)


def strip_sir(text: str) -> str:
    """Remove every vocative "Sir". A reply that would be left empty is returned unchanged."""
    result = _LEADING.sub(lambda match: match.group(1) + match.group(2).upper(), text)
    result = _AFTER_COMMA.sub("", result)
    result = _BARE.sub("", result)
    return result if re.search(r"\w", result) else text


def limit_sir(owner: object, text: str, response_language: str) -> str:
    """Apply the budget to one finalised reply. `owner` carries the count between replies."""
    if normalize_response_language(response_language) != ENGLISH_GB:
        return text
    since_last = getattr(owner, "_replies_since_sir", SIR_EVERY - 1)
    first = _ANY.search(text)
    if first is not None and since_last >= SIR_EVERY - 1:
        owner._replies_since_sir = 0
        return text[: first.end()] + strip_sir(text[first.end():]) if text[first.end():].strip() else text
    owner._replies_since_sir = since_last + 1
    return strip_sir(text)
