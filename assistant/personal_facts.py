"""Questions about Alexandre's own life that Echo has no answer for.

R3 step 4: a small model guesses ("Tens um irmão.") even when told not to. For
a few common topics this module decides before the model is called: when the
question is about one of them and nothing Echo was given states that fact,
the reply is a fixed "I don't know". A word that merely occurs is not a fact:
a profile saying Alexandre is a coach does not say who his coach is.
Everything else is left to the model, which carries the same rule in its prompt.
"""

from __future__ import annotations

import re
import unicodedata

from assistant.response_language import ENGLISH_GB, normalize_response_language

_POSSESSIVE = r"(?:meu|minha|meus|minhas|teu|tua|teus|tuas|seu|sua|seus|suas|my|your|his)"


def _relation_fact(noun: str, extra: str = "") -> str:
    """Evidence that a relation of Alexandre's is actually stated, not just that
    the word occurs. "É treinador de voleibol" says what he is; "o treinador
    dele é o Paulo", "Treinador: Paulo" and "tem dois irmãos" state a fact."""
    patterns = [
        rf"\b{_POSSESSIVE} (?:{noun})\b",                                        # o meu treinador, my coach
        rf"\b(?:{noun}) (?:dele|do alexandre)\b",                                # a irmã dele
        rf"\b(?:tenho|tem|tens|has|have) (?:\w+ ){{0,3}}(?:{noun})\b",           # tem dois irmãos, has a dog
        rf"(?:^|\n)\W*(?:{noun})\s*:",                                           # Treinador: Paulo
        rf"\b(?:o|a|os|as|the) (?:{noun})\b[^.\n]{{0,30}}\b(?:eh|chama-se|chamam-se|sao|is|are)\b",  # o treinador ... é o Paulo
    ]
    return "|".join(patterns + ([extra] if extra else []))


_SIBLINGS = r"irma(?:o|os|s)?|siblings?|brothers?|sisters?"
_PARTNER = r"namorad[oa]|girlfriend|boyfriend|wife|husband|marido|esposa"
_CHILDREN = r"filh[oa]s?|children|kids?|sons?|daughters?"
_COACH = r"treinador(?:a|es)?|coach(?:es)?"
_PETS = r"cao|caes|gatos?|animal de estimacao|animais de estimacao|pets?|dogs?|cats?"
# topic: (what the question mentions, what counts as a stated fact about the topic)
_TOPICS: dict[str, tuple[str, str]] = {
    "siblings": (rf"\b(?:{_SIBLINGS})\b", _relation_fact(_SIBLINGS)),
    # "pais" is not asked about here: without its accent it is also "país".
    "parents": (r"\b(?:pai|mae|father|mother|parents|dad|mum|mom)\b", _relation_fact(r"pai|mae|pais|father|mother|parents|dad|mum|mom")),
    "partner": (rf"\b(?:{_PARTNER})\b", _relation_fact(_PARTNER)),
    "children": (rf"\b(?:{_CHILDREN})\b", _relation_fact(_CHILDREN)),
    "address": (
        r"\bmorada\b|\bendereco\b|\baddress\b|\bcodigo postal\b|\bpostcode\b",
        r"\b(?:mora|moro|morada|endereco|address|lives?)\b[^.\n]{0,25}\b(?:rua|avenida|travessa|largo|street|road|avenue)\b"
        r"|\bcodigo postal\b|\bpostcode\b|\b\d{4}-\d{3}\b",
    ),
    "birthday": (
        r"\baniversario\b|\bfaco anos\b|\bquantos anos tenho\b|\bque idade\b|\bbirthday\b|\bhow old am i\b|\bwas i born\b|\bnasci\b|\bdata de nascimento\b",
        r"\bnasc(?:eu|i|ido|ida)\b|\bborn (?:on|in)\b|\bfa(?:z|co) anos\b|\b(?:aniversario|birthday)\b[^.\n]{0,20}\d"
        r"|\b\d+ anos de idade\b|\bidade\s*:|\byears old\b|\bdata de nascimento\b",
    ),
    "coach": (rf"\b(?:{_COACH})(?:'s)?\b", _relation_fact(_COACH, r"\btreinad[oa] (?:por|pel[oa])\b|\bcoached by\b")),
    "pets": (rf"\b(?:{_PETS})\b", _relation_fact(_PETS)),
    "phone": (r"\btelefone\b|\btelemovel\b|\bphone\b", r"\b(?:telefone|telemovel|phone|mobile)\b[^.\n]{0,20}\d{3}"),
}
_FIRST_PERSON = r"\b(?:eu|meu|minha|meus|minhas|tenho|sou|faco|nasci|moro|vivo|i|my|mine)\b"
_QUESTION_START = (
    r"^(?:quant[oa]s?|qual|quais|onde|quando|como|quem|que|o que|em que|"
    r"how|what(?:'s| is)?|where|when|who|which|do i|am i|have i)\b"
)
# Asking for advice or help is not asking for a fact.
_NOT_A_FACT_QUESTION = r"\b(?:achas|devo|devia|podes|consegues|ajuda\w*|should|can you|could you|would you|help|do you think)\b"


def _normalize(text: str) -> str:
    """Lower-case and accent-free, line by line. The verb "é" becomes "eh" first,
    so that it is not confused with the conjunction "e"."""
    text = re.sub(r"\bé\b", "eh", str(text or "").replace("’", "'").lower())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


def unknown_reply(response_language: str) -> str:
    if normalize_response_language(response_language) == ENGLISH_GB:
        return "I'm afraid I don't know that. Tell me and I'll remember it, if you like."
    return "Receio não saber isso. Diz-me e guardo essa informação, se quiseres."


def missing_personal_fact_reply(user_message: str, evidence: list[str], response_language: str) -> str | None:
    """The fixed "I don't know" reply, or None when the model should answer.

    `evidence` is everything Echo was given that may hold the answer: the core
    profile, what the user said earlier, memory facts and context.
    """
    text = _normalize(user_message).replace("\n", " ").strip(" .!")
    is_question = "?" in text or re.search(_QUESTION_START, text)
    if not is_question or not re.search(_FIRST_PERSON, text) or re.search(_NOT_A_FACT_QUESTION, text):
        return None
    known = _normalize(" \n ".join(evidence))
    for asked, proof in _TOPICS.values():
        if re.search(asked, text) and not re.search(proof, known):
            return unknown_reply(response_language)
    return None
