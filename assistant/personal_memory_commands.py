"""Consent and receipts for personal facts. Never derive a fact from assistant text."""
from __future__ import annotations

import hashlib
import re
import unicodedata

from assistant.personal_model import infer_category


def normalized(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def explicit_fact(message: str, previous_user: str) -> str | None:
    text = message.strip()
    clean = normalized(text)
    if re.search(r"\b(ficheiro|arquivo|file|workspace)\b", clean) and re.search(r"\b(em|num|no|to|in|as)\b", clean):
        return None  # Artifact exports belong to their existing workflow.
    prefix = re.match(r"^(?:lembra-te que|lembra que|nao te esquecas que|guarda isto|remember that|save this)\b[\s:,.]*", clean)
    suffix = re.search(r"(?:[.!?,]\s*|^)(?:podes[- ]te lembrar disso|podes lembrar-te disso|can you remember (?:that|this)|remember (?:that|this))[.!?\s]*$", clean)
    if prefix:
        fact = text[prefix.end():].strip(" .:!?")
    elif suffix:
        fact = text[:suffix.start()].strip(" .:!?")
    else:
        return None
    if not fact:
        fact = previous_user.strip(" .:!?") if "?" not in previous_user else ""
        if re.match(r"^(?:quem|qual|como|who|what|how|remember|lembra|guarda)\b", normalized(fact)):
            fact = ""
    return fact


def handle_personal_memory(engine, message: str) -> str | None:
    fact = explicit_fact(message, engine._previous_user_message())
    if fact is None:
        return None
    english = engine.response_language == "en-GB"
    if not fact:
        return "What exactly would you like me to remember?" if english else "O que queres que guarde, exatamente?"
    if engine.personal_model is None or not engine.presence.can_store_memory():
        return "I couldn't save that information." if english else "Não consegui guardar essa informação."
    try:
        entry = engine.personal_model.add_or_update_entry(
            category=infer_category(fact), key=hashlib.sha256(normalized(fact).encode("utf-8")).hexdigest(),
            description=fact, confidence=100, evidence=message, source="user_explicit", status="confirmado",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[PERSONAL_MEMORY] write_failed={type(exc).__name__}")
        return "I couldn't save that information." if english else "Não consegui guardar essa informação."
    engine._work_session.last_personal_memory_id = entry.id
    if engine._turn_trace is not None:
        engine._turn_trace["personal_memory_written"] = True
    return f'Saved: "{entry.description}".' if english else f'Guardei: "{entry.description}".'
