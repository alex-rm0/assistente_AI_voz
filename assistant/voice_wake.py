"""Phase 7.6: deterministic wake/sleep intent classification for Voice
Standby Mode -- NOT an LLM call (section 3 explicitly forbids invoking one
for obvious wake/sleep phrases). A small, pure, easily-testable text
classifier: normalize (lowercase, strip accents/punctuation), then match
against a bounded list of known Portuguese/English phrases with word
boundaries so "acorda" doesn't accidentally match inside an unrelated word.

This is intentionally simple pattern matching, not a general intent
parser -- false negatives (an unrecognized phrasing gets ignored) are safe
and expected; the goal is only to catch the obvious cases from section 3's
examples without ever needing a model call.
"""
from __future__ import annotations

import re
import unicodedata

_WAKE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bhey echo\b",
        r"\becho\b",
        r"\bwake up\b",
        r"\bstart listening\b",
        r"\bacorda\b",
        r"\bpodes ouvir[- ]me\b",
        r"\bcomeca a ouvir\b",
    )
]

_SLEEP_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bgo to sleep\b",
        r"\byou can go to sleep\b",
        r"\bstop listening\b",
        r"\byou can stop listening\b",
        r"\bvai dormir\b",
        r"\bpode ir dormir\b",
        r"\bpodes ir dormir\b",
        r"\bpode dormir\b",
        r"\bpodes dormir\b",
        r"\bpodes parar de ouvir\b",
        r"\bpara de ouvir\b",
        r"\bja podes parar\b",
        r"\bpode descansar\b",
    )
]

# Phase 7.6.5 (section 3/5): bare single-word sleep triggers ("dorme",
# "dormir", "sleep", plus the truncated STT variant "dorm" seen in a real
# session -- e.g. "Dorm.") are NOT matched as a substring anywhere in the
# text like the phrases above, unlike "pode ir dormir" these single words
# commonly appear inside completely unrelated sentences ("ontem fui dormir
# tarde", "acho que devia dormir mais", "o computador entrou em sleep") --
# section 4's false-positive protection. Instead they only count as a sleep
# command when the ENTIRE utterance (modulo a small, explicit set of
# leading acknowledgement fillers such as "ok"/"pronto"/"nao") reduces to
# just that one word -- see _is_bare_sleep_word_utterance below.
_SLEEP_BARE_WORDS = {"dorme", "dormir", "dorm", "sleep"}
_SLEEP_LEADING_FILLERS = {"nao", "ok", "okay", "pronto", "entao"}


def _is_bare_sleep_word_utterance(normalized: str) -> bool:
    tokens = normalized.split()
    if not tokens:
        return False
    index = 0
    while index < len(tokens) - 1 and tokens[index] in _SLEEP_LEADING_FILLERS:
        index += 1
    remaining = tokens[index:]
    return len(remaining) == 1 and remaining[0] in _SLEEP_BARE_WORDS


def _normalize_wake_text(text: str) -> str:
    """Lowercase, strip accents (so "começa"/"comeca" both match), and
    collapse punctuation to spaces so word-boundary patterns still work
    across "Echo, acorda!" / "Echo acorda" / "echo... acorda"."""

    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = without_accents.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


# Phase 7.6.5 (section 4): every required sleep phrase, including with its
# allowed conversational prefix ("não, pode ir dormir", "that's all, go to
# sleep"), normalizes to at most this many tokens. Sleep patterns are only
# matched within that bound so an unrelated LONGER sentence that happens to
# contain one as a substring ("ele pode ir dormir mais cedo hoje") isn't
# misread as a command clearly targeting Echo -- a short imperative
# utterance is what "clearly targets Echo / voice control" looks like.
_SLEEP_MAX_TOKENS = 6


def classify_voice_wake_intent(text: str) -> str:
    """Returns "wake", "sleep", or "none". Sleep phrases are checked first
    since they are all multi-word and more specific than the loose bare
    "echo"/"acorda" wake triggers -- in practice the two sets never
    overlap, but checking sleep first keeps that guarantee explicit."""

    normalized = _normalize_wake_text(text)
    if not normalized:
        return "none"
    if len(normalized.split()) <= _SLEEP_MAX_TOKENS and any(
        pattern.search(normalized) for pattern in _SLEEP_PATTERNS
    ):
        return "sleep"
    if _is_bare_sleep_word_utterance(normalized):
        return "sleep"
    if any(pattern.search(normalized) for pattern in _WAKE_PATTERNS):
        return "wake"
    return "none"


# Phase 7.6.7 (section 3/4): the gate that decides whether a fast (tiny)
# STANDBY transcript that did NOT match a wake command is still worth one
# Medium fallback pass. A real session paid stt_ms=9184 for a Medium
# fallback that also ended in wake_ignored -- "tiny non-match -> always
# Medium" is worse UX than simply rearming, since the user can naturally
# say "Echo" again. So Medium now only runs when the fast transcript is
# plausibly a MISHEARD wake word (short, and containing a token that is
# phonetically close to one of the wake keywords); empty or clearly
# unrelated fast results rearm immediately instead.
#
# This is ONLY a "should we spend a second, slower pass?" gate -- it never
# accepts anything itself. A near-wake candidate still has to pass the
# strict classify_voice_wake_intent() on Medium's transcript to actually
# wake Echo, so this loose similarity check cannot cause a false wake.
_NEAR_WAKE_MAX_TOKENS = 4
_NEAR_WAKE_KEYWORDS = ("eko", "akorda", "wake")


def _phonetic_wake_key(token: str) -> str:
    """Collapses the spelling variants Whisper produces for a short wake
    word ("echo"/"eco"/"eko"/"ecco"/"ekko"/"hecho") onto one key, so a
    single small edit-distance bound covers them without a word list."""
    key = token.lstrip("h")
    key = key.replace("ch", "k").replace("q", "k").replace("c", "k")
    return re.sub(r"(.)\1+", r"\1", key)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left_char != right_char))
            )
        previous = current
    return previous[-1]


def assess_fast_wake_transcript(text: str) -> str:
    """Phase 7.6.7: buckets a fast-recognizer STANDBY transcript into
    "wake" (the strict parser already matched -- trust it), "empty"
    (nothing usable -- rearm), "near_wake" (plausibly a misheard wake word
    -- worth one Medium pass) or "unrelated" (clearly something else,
    including a sleep phrase, which means nothing in STANDBY -- rearm)."""
    normalized = _normalize_wake_text(text)
    if not normalized:
        return "empty"
    if classify_voice_wake_intent(text) == "wake":
        return "wake"
    tokens = normalized.split()
    if len(tokens) > _NEAR_WAKE_MAX_TOKENS:
        return "unrelated"
    for token in tokens:
        if len(token) < 3:
            continue
        key = _phonetic_wake_key(token)
        for keyword in _NEAR_WAKE_KEYWORDS:
            if _edit_distance(key, keyword) <= (1 if len(keyword) <= 4 else 2):
                return "near_wake"
    return "unrelated"


def wake_acknowledgement_text(response_language: str) -> str:
    """Section 6: a concise, deterministic (non-LLM) acknowledgement --
    chosen over silently transitioning to listening because it gives the
    user (who has no screen focus in a voice-first flow) clear confirmation
    that the wake word worked, without any accidental-self-capture risk:
    the existing TTS lifecycle already guarantees the mic stays closed
    until this finishes speaking (see controller.py's _wake_into_active_conversation)."""

    if str(response_language or "").strip() == "pt-PT":
        return "Estou a ouvir."
    return "I'm listening."
