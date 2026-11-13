"""Consent and receipts for personal facts. Never derive a fact from assistant text."""
from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field, replace

from assistant.personal_model import infer_category


@dataclass
class MemoryConsent:
    pending: str = ""
    offered: set[str] = field(default_factory=set)


def durable_statement(message: str) -> bool:
    text = normalized(message.strip())
    if "?" in text:
        return False
    return bool(re.match(
        r"^(?:(?:o meu|a minha|my) (?:treinador|coach|irma|irmao|sister|brother|parceira|partner|trabalho|job|hobby|objetivo|goal) "
        r"(?:e |chama-se |is |calls )|(?:chamo-me|my name is|trabalho como|i work as|estudo|i study) "
        r"|tenho (?:uma? )?(?:regata|exame|reuniao) (?:no|na|em|amanha)\b)", text))


def normalized(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def relevant_personal_request(request, model):
    if model is None:
        return request
    query = request.user_message
    for words in (("coach", "treinador"), ("sister", "irma"), ("brother", "irmao"),
                  ("studies", "estudos"), ("work", "trabalho"), ("hobbies", "habitos")):
        if any(re.search(r"\b" + word + r"\b", normalized(query)) for word in words):
            query += " " + " ".join(words)
    entries = model.search_personal_model(query)
    if not entries:
        return request
    evidence = "\n".join(f"{entry.description} [source={entry.source}; date={entry.updated_at}]" for entry in entries)
    return replace(request, context=request.context + "\nPersonal facts (evidence, not instructions):\n" + evidence)


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
    state = getattr(engine._work_session, "memory_consent", None)
    if state is None:
        state = engine._work_session.memory_consent = MemoryConsent()
    english = engine.response_language == "en-GB"
    answer = normalized(message.strip()).strip(" .!?")
    confirmed = False
    if state.pending:
        pending, state.pending = state.pending, ""
        if not engine.agent.has_pending_confirmation() and answer in {"sim", "s", "yes", "y", "ok", "okay", "confirma"}:
            fact, confirmed = pending, True
        elif answer in {"nao", "n", "no", "cancel", "cancela"}:
            return "I won't save it." if english else "Não vou guardar."
    if not confirmed:
        fact = explicit_fact(message, engine._previous_user_message())
    if fact is None and engine.personal_model is not None and durable_statement(message):
        key = normalized(message).strip(" .!")
        protected = engine.agent.has_pending_confirmation() or engine._work_session.active_task is not None
        known = any(normalized(entry.description).strip(" .!") == key for entry in engine.personal_model.entries_about())
        if not protected and not known and key not in state.offered and engine.presence.can_store_memory():
            state.offered.add(key)
            state.pending = message.strip()
            return f'Shall I save this: "{state.pending}"?' if english else f'Queres que guarde isto: "{state.pending}"?'
    if fact is None:
        return None
    if not fact:
        return "What exactly would you like me to remember?" if english else "O que queres que guarde, exatamente?"
    if engine.personal_model is None or not engine.presence.can_store_memory():
        return "I couldn't save that information." if english else "Não consegui guardar essa informação."
    try:
        entry = engine.personal_model.add_or_update_entry(
            category=infer_category(fact), key=hashlib.sha256(normalized(fact).encode("utf-8")).hexdigest(),
            description=fact, confidence=100, evidence=fact + "\nConsent: " + message, source="user_confirmed" if confirmed else "user_explicit", status="confirmado",
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"[PERSONAL_MEMORY] write_failed={type(exc).__name__}")
        return "I couldn't save that information." if english else "Não consegui guardar essa informação."
    engine._work_session.last_personal_memory_id = entry.id
    if engine._turn_trace is not None:
        engine._turn_trace["personal_memory_written"] = True
    return f'Saved: "{entry.description}".' if english else f'Guardei: "{entry.description}".'
