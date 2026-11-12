"""Questions about Alexandre's own life that Echo has no answer for.

R3 step 4: a small model guesses ("Tens um irmão.") even when told not to. For
a few common topics this module decides before the model is called: when the
question is about one of them and nothing Echo was given mentions that topic,
the reply is a fixed "I don't know". Everything else is left to the model,
which carries the same rule in its prompt.
"""

from __future__ import annotations

import re
import unicodedata

from assistant.response_language import ENGLISH_GB, normalize_response_language

# topic: (what the question mentions, what counts as evidence about the topic)
_TOPICS: dict[str, tuple[str, str]] = {
    "siblings": (r"\birma(?:o|os|s)?\b|\bsiblings?\b|\bbrothers?\b|\bsisters?\b",) * 2,
    # "pais" is not asked about here: without its accent it is also "país".
    "parents": (r"\b(?:pai|mae|father|mother|parents|dad|mum|mom)\b", r"\b(?:pai|mae|pais|father|mother|parents|dad|mum|mom)\b"),
    "partner": (r"\bnamorad[oa]\b|\bgirlfriend\b|\bboyfriend\b|\bwife\b|\bhusband\b|\bmarido\b|\besposa\b",) * 2,
    "children": (r"\bfilh[oa]s?\b|\bchildren\b|\bkids?\b|\bsons?\b|\bdaughters?\b",) * 2,
    "address": (
        r"\bmorada\b|\bendereco\b|\baddress\b|\bcodigo postal\b|\bpostcode\b",
        r"\bmorada\b|\bendereco\b|\baddress\b|\bcodigo postal\b|\bpostcode\b|\brua\b|\bavenida\b|\bstreet\b",
    ),
    "birthday": (
        r"\baniversario\b|\bfaco anos\b|\bquantos anos tenho\b|\bque idade\b|\bbirthday\b|\bhow old am i\b|\bwas i born\b|\bnasci\b|\bdata de nascimento\b",
        r"\baniversario\b|\bnasc\w+|\bborn\b|\bbirthday\b|\bidade\b|\byears old\b|\banos de idade\b",
    ),
    "coach": (r"\btreinador[a]?\b|\bcoach(?:'s)?\b",) * 2,
    "pets": (r"\b(?:cao|gato|animal de estimacao|pets?|dog|cat)\b",) * 2,
    "phone": (r"\btelefone\b|\btelemovel\b|\bphone\b",) * 2,
}
_FIRST_PERSON = r"\b(?:eu|meu|minha|meus|minhas|tenho|sou|faco|nasci|moro|vivo|i|my|mine)\b"
_QUESTION_START = (
    r"^(?:quant[oa]s?|qual|quais|onde|quando|como|quem|que|o que|em que|"
    r"how|what(?:'s| is)?|where|when|who|which|do i|am i|have i)\b"
)
# Asking for advice or help is not asking for a fact.
_NOT_A_FACT_QUESTION = r"\b(?:achas|devo|devia|podes|consegues|ajuda\w*|should|can you|could you|would you|help|do you think)\b"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "").replace("’", "'").lower())
    return " ".join("".join(ch for ch in text if not unicodedata.combining(ch)).split())


def unknown_reply(response_language: str) -> str:
    if normalize_response_language(response_language) == ENGLISH_GB:
        return "I'm afraid I don't know that — tell me and I'll remember it, if you like."
    return "Receio não saber isso — diz-me e eu guardo, se quiseres."


def missing_personal_fact_reply(user_message: str, evidence: list[str], response_language: str) -> str | None:
    """The fixed "I don't know" reply, or None when the model should answer.

    `evidence` is everything Echo was given that may hold the answer: the core
    profile, what the user said earlier, memory facts and context.
    """
    text = _normalize(user_message).strip(" .!")
    is_question = "?" in text or re.search(_QUESTION_START, text)
    if not is_question or not re.search(_FIRST_PERSON, text) or re.search(_NOT_A_FACT_QUESTION, text):
        return None
    known = _normalize(" \n ".join(evidence))
    for asked, proof in _TOPICS.values():
        if re.search(asked, text) and not re.search(proof, known):
            return unknown_reply(response_language)
    return None
