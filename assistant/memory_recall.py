from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from assistant.long_term_memory import STRUCTURED_FACT_ATTRIBUTES, StructuredFact


MEMORY_RECALL_INSTRUCTION = (
    "Estás a responder a uma pergunta sobre memória.\n"
    "Usa apenas os factos apresentados em \"Memórias recuperadas\" ou no histórico atual.\n"
    "Não afirmes que te lembras de algo que não apareça nessas fontes.\n"
    "Não adivinhes nomes, datas, disciplinas, pessoas ou acontecimentos.\n"
    "Se faltar o facto pedido, diz claramente que não o tens guardado.\n"
    "Responde diretamente ao atributo pedido."
)


# --- Section 3: detecting memory questions ---------------------------------

# Regex, not substring matching: European Portuguese lets the clitic "te"
# attach before or after the verb ("lembras-te?" / "ainda te lembras?"), with
# or without a hyphen, and with arbitrary words in front ("então, ainda te
# lembras...", "mas não te lembras..."). A closed phrase list missed most of
# these. Deliberately excludes bare "lembra-te" (imperative, no "s"): that
# form is the existing "lembra-te que X" store command and must keep working
# — only the question form ("lembras", with the 's') counts as recall here.
_RECALL_PATTERN = re.compile(
    r"\bte\s+lembr[ae]s?\b"
    r"|\blembr[ae]s[\s-]+te\b"
    r"|\bte\s+record[ae]s?\b"
    r"|\brecord[ae]s[\s-]+te\b"
    r"|\bja\s+te\s+ti?nha\s+(?:dito|falado|contado)\b"
    r"|\bo\s+que\s+(?:e\s+que\s+)?te\s+disse\b"
    r"|\bfalamos\s+disso\b"
    r"|\btinhas\s+guardado\b"
    r"|\btens\s+guardado\b"
    r"|\bconfirma\s+na\s+memoria\b"
    r"|\bconfirma\s+na\s+memória\b"
    r"|\bconsulta\s+(?:a\s+)?memoria\b"
    r"|\bconsulta\s+(?:a\s+)?memória\b"
    r"|\b(?:verifica|confirma)\s+(?:que\s+)?exame\b"
    r"|\b(?:verifica|confirma)\s+(?:qual\s+e\s+|qual\s+é\s+)?(?:o\s+)?meu\s+exame\b"
    r"|\b(?:que|qual)\s+exame\s+(?:tenho|vou\s+ter)\b"
    r"|\bo\s+que\s+guardaste\s+sobre\s+o\s+meu\s+exame\b"
    r"|\bsobre\s+o\s+meu\s+exame\b"
)


_FOLLOWUP_ATTRIBUTE_MARKERS = (
    "qual era",
    "qual e",
    "qual foi",
    "quando era",
    "quando foi",
    "quem era",
    "quem foi",
    "onde era",
    "onde foi",
    "como se chamava",
    "como correu",
    "e a disciplina",
    "e a data",
    "e quando",
)


def _marker_pattern(marker: str) -> re.Pattern[str]:
    # Word-boundary matching, not a raw substring check: a short marker like
    # "e quando" would otherwise also match inside unrelated words that
    # merely end in "e" followed by "quando" as running text (e.g. "regista
    # que a disciplina..." contains the literal characters "e a disciplina"
    # inside "qu[e a disciplina]", which is not the standalone "E a
    # disciplina?" follow-up this marker is meant to catch).
    escaped = r"\s+".join(re.escape(word) for word in marker.split())
    return re.compile(rf"\b{escaped}\b")


_FOLLOWUP_ATTRIBUTE_PATTERNS = tuple(_marker_pattern(marker) for marker in _FOLLOWUP_ATTRIBUTE_MARKERS)

_FOLLOWUP_RECALL_MARKERS = (
    "nao tens",
    "não tens",
    "tens a certeza",
    "entao nao tens",
    "então não tens",
    "entao nao guardaste",
    "então não guardaste",
    "nao guardaste",
    "não guardaste",
    "afinal nao tens",
    "afinal não tens",
    "nao sabes",
    "não sabes",
    "nao tens isso guardado",
    "não tens isso guardado",
    "supostamente guardaste",
    "supostamente guardaste isso",
    "supostamente guardaste essa informacao",
    "supostamente guardaste essa informação",
    "confirma la",
    "confirma lá",
    "sobre o meu exame",
)


def is_memory_recall_question(normalized_text: str) -> bool:
    if _RECALL_PATTERN.search(normalized_text):
        return True
    # "Qual era/foi...?", "Quando era...?" etc. are generic Portuguese question
    # shapes used constantly outside memory ("qual é a tua língua base?"), so
    # they only count as a memory question when paired with a topic this
    # feature actually tracks (an exam, or one of its attributes) — never as
    # a bare pattern on their own.
    if any(pattern.search(normalized_text) for pattern in _FOLLOWUP_ATTRIBUTE_PATTERNS):
        topic_words = _EVENT_MARKERS + tuple(
            keyword for keywords in _ATTRIBUTE_KEYWORDS.values() for keyword in keywords
        )
        return any(word in normalized_text for word in topic_words)
    return False


_TASK_RECALL_MARKERS = (
    "tarefas pendentes",
    "que tarefas tenho",
    "que lembretes tenho",
    "o que tenho para fazer",
)


def is_task_recall_question(normalized_text: str) -> bool:
    return any(marker in normalized_text for marker in _TASK_RECALL_MARKERS)


# --- Sections 4-6: explicit MEMORY_WRITE commands ---------------------------
#
# Deliberately matches the raw message (not the accent-stripped normalized
# text): these are specific admin-style phrasings ("regista que...", "guarda
# que..."), not organic conversational paraphrasing, so accent-tolerance
# matters far less here than for recall detection above. Ordered most
# specific first so e.g. "atualiza a disciplina para X" doesn't get
# swallowed by the more generic "atualiza ..." pattern.
#
# field_kind tells the caller how to turn `content` into a candidate:
# "sentence" -> re-run the normal statement extractors on it (it's a full
# clause, e.g. "a disciplina é X" or "tenho de mandar..."); "discipline" /
# "date_reference" -> `content` is a bare value with no verb around it
# (e.g. "engenharia informatica", "segunda-feira"), so the field is set
# directly instead of depending on a sentence-shaped extractor to fire.
_MEMORY_WRITE_PATTERNS = (
    ("correct", "date_reference", re.compile(r"^corrige\s+a\s+data\s+para\s+(.+)$", re.IGNORECASE)),
    ("update", "discipline", re.compile(r"^atualiza\s+a\s+disciplina\s+para\s+(.+)$", re.IGNORECASE)),
    ("register", "sentence", re.compile(r"^regista\s+que\s+(.+)$", re.IGNORECASE)),
    ("register", "sentence", re.compile(r"^guarda\s+que\s+(.+)$", re.IGNORECASE)),
    ("register", "sentence", re.compile(r"^memoriza\s+que\s+(.+)$", re.IGNORECASE)),
    ("register", "sentence", re.compile(r"^fica\s+com\s+esta\s+informa[cç][aã]o[,:]?\s+(.+)$", re.IGNORECASE)),
    ("register", "sentence", re.compile(r"^lembra[- ]te\s+de\s+que\s+(.+)$", re.IGNORECASE)),
    ("register", "sentence", re.compile(r"^anota\s+que\s+(.+)$", re.IGNORECASE)),
    ("update", "sentence", re.compile(r"^atualiza[,:]?\s+(.+)$", re.IGNORECASE)),
    ("update", "sentence", re.compile(r"^afinal[,:]?\s+(.+)$", re.IGNORECASE)),
)


def parse_memory_write_command(message: str) -> tuple[str, str, str] | None:
    """Returns (verb_kind, field_kind, content) for an explicit write command, else None.

    verb_kind is one of register|update|correct — only used to phrase the
    confirmation ("Registado."/"Atualizado."/"Corrigido.").
    """
    text = message.strip()
    for verb_kind, field_kind, pattern in _MEMORY_WRITE_PATTERNS:
        match = pattern.match(text)
        if match:
            return verb_kind, field_kind, match.group(1).strip(" .!?")
    return None


def is_memory_write_command(message: str) -> bool:
    return parse_memory_write_command(message) is not None


_WRITE_CONFIRMATION_PREFIXES = {
    "register": "Registado.",
    "update": "Atualizado.",
    "correct": "Corrigido.",
}


def render_memory_write_confirmation(verb_kind: str, fact_type: str, canonical_fields: dict[str, str]) -> str:
    """Deterministic confirmation for a completed write — never sent to the LLM."""
    prefix = _WRITE_CONFIRMATION_PREFIXES.get(verb_kind, "Registado.")
    if fact_type == "task" and canonical_fields.get("action"):
        if canonical_fields.get("reminder_requested") == "true":
            return f"{prefix} Vou lembrar-te de {canonical_fields['action']}."
        return f"{prefix} Tens de {canonical_fields['action']}."
    if fact_type == "academic_event":
        if canonical_fields.get("discipline"):
            return f"{prefix} A disciplina é {canonical_fields['discipline']}."
        if canonical_fields.get("date_reference"):
            # date_reference already carries its own preposition ("para a
            # semana", "na sexta-feira") — do not prepend another one.
            return f"{prefix} O exame ficou registado {canonical_fields['date_reference']}."
        if canonical_fields.get("degree"):
            return f"{prefix} O curso é {canonical_fields['degree']}."
        if canonical_fields.get("status"):
            return f"{prefix} O estado ficou atualizado."
    return f"{prefix} A informação ficou guardada."


def is_memory_attribute_followup(normalized_text: str) -> bool:
    """A short follow-up like "Qual era a disciplina?" keeps a memory recall going."""
    return any(pattern.search(normalized_text) for pattern in _FOLLOWUP_ATTRIBUTE_PATTERNS)


def is_memory_recall_followup(normalized_text: str) -> bool:
    """Short challenge/follow-up after a grounded memory recall."""
    text = normalized_text.strip(" .,!?:;")
    return is_memory_attribute_followup(text) or any(marker in text for marker in _FOLLOWUP_RECALL_MARKERS)


_ATTRIBUTE_KEYWORDS = {
    # "event" is deliberately not a requestable attribute here: almost any
    # question about an exam mentions the word "exame", which would otherwise
    # make every query trivially "covered" and mask a genuinely missing
    # attribute like date or discipline.
    "discipline": ("disciplina", "cadeira", "materia", "unidade curricular"),
    "degree": ("licenciatura", "curso", "mestrado"),
    "date_reference": ("quando", "data", "dia"),
    "person": ("quem",),
    "location": ("onde", "local", "lugar"),
    "status": ("como correu", "que estado", "estado"),
    "outcome": ("resultado", "nota"),
}


def extract_requested_attributes(normalized_text: str) -> set[str]:
    found = set()
    for attribute, keywords in _ATTRIBUTE_KEYWORDS.items():
        if any(keyword in normalized_text for keyword in keywords):
            found.add(attribute)
    if "que exame" in normalized_text or "qual exame" in normalized_text:
        found.add("discipline")
    return found


ATTRIBUTE_LABELS = {
    "discipline": "a disciplina",
    "degree": "o curso",
    "date_reference": "a data",
    "person": "a pessoa",
    "location": "o local",
    "status": "o estado",
    "outcome": "o resultado",
    "event": "o acontecimento",
}


# --- Section 10: light structured extraction from user statements ---------

_EVENT_MARKERS = ("exame", "teste", "prova")

_DISCIPLINE_PATTERNS = (
    re.compile(r"exame\s+d[eo]\s+([^.,;:!?]+)", re.IGNORECASE),
    re.compile(r"\bé\s+d[eo]\s+([^.,;:!?]+)", re.IGNORECASE),
    re.compile(r"\ba disciplina\s+é\s+([^.,;:!?]+)", re.IGNORECASE),
)
_DEGREE_PATTERN = re.compile(r"licenciatura\s+em\s+([^.,;:!?]+)|curso\s+de\s+([^.,;:!?]+)", re.IGNORECASE)
_ACADEMIC_CONTEXT_PATTERN = re.compile(r",?\s*da\s+(faculdade|universidade)\b", re.IGNORECASE)

_DATE_REFERENCE_PATTERN = re.compile(
    r"\b(para a semana|na pr[oó]xima semana|depois de amanh[aã]|amanh[aã]|"
    r"(?:n[ao]\s+)?(?:segunda|ter[cç]a|quarta|quinta|sexta)-feira|s[aá]bado|domingo|"
    r"(?:no\s+)?dia\s+\d{1,2}|no pr[oó]ximo m[eê]s)\b",
    re.IGNORECASE,
)

_ASKED_WHICH_EVENT = (
    "que exame",
    "qual exame",
    "qual a disciplina",
    "qual e a disciplina",
    "qual era a disciplina",
    "que disciplina",
)

_STATUS_MARKERS = {
    "failed": ("chumbei", "reprovei", "correu mal", "corri mal"),
    "passed": ("passei", "tive positiva", "fui aprovado", "fui aprovada"),
    "completed": ("ja fiz o exame", "já fiz o exame", "o exame terminou", "fiz o exame ontem", "ja acabou", "já acabou", "o exame ja aconteceu", "o exame já aconteceu"),
    "cancelled": ("cancelaram", "foi cancelado", "adiaram"),
}


def extract_academic_event_candidate(message: str, previous_assistant_message: str = "") -> dict[str, str]:
    """Best-effort extraction of exam-related facts from a single user turn.

    Deliberately narrow (exams only, not a general-purpose fact extractor) —
    it mirrors the exact vocabulary this project's memory feature was asked
    to support, rather than attempting broad NLU it can't reliably do.
    """
    text = message.strip()
    normalized = _normalize(text)
    attributes: dict[str, str] = {}
    is_query = is_memory_recall_question(normalized)

    if any(marker in normalized for marker in _EVENT_MARKERS):
        attributes["event"] = "exame"

    context_match = _ACADEMIC_CONTEXT_PATTERN.search(text)
    if context_match:
        attributes["context"] = context_match.group(1).lower()
        text = _ACADEMIC_CONTEXT_PATTERN.sub("", text)

    date_match = _DATE_REFERENCE_PATTERN.search(text)
    if date_match:
        attributes["date_reference"] = date_match.group(1)
        text = _DATE_REFERENCE_PATTERN.sub("", text)

    for pattern in _DISCIPLINE_PATTERNS:
        discipline_match = pattern.search(text)
        if not discipline_match:
            continue
        discipline = _trim_trailing_time_reference(discipline_match.group(1))
        if discipline and not _looks_like_backreference(discipline):
            attributes["discipline"] = discipline
            attributes.setdefault("event", "exame")
            break

    degree_match = _DEGREE_PATTERN.search(text)
    if degree_match:
        attributes["degree"] = (degree_match.group(1) or degree_match.group(2)).strip(" .,!?:;")

    explicit_status = False
    for status, markers in _STATUS_MARKERS.items():
        if any(marker in normalized for marker in markers):
            attributes["status"] = status
            explicit_status = True
            attributes.setdefault("event", "exame")

    if attributes.get("event") == "exame" and "status" not in attributes:
        attributes["status"] = "upcoming"

    # A question ("Lembras-te do exame?") is not new evidence on its own —
    # associating a short reply with a preceding "Que exame?" only makes
    # sense when the current turn is itself a statement, not another query.
    previous_normalized = _normalize(previous_assistant_message)
    if not is_query:
        if "discipline" not in attributes and any(marker in previous_normalized for marker in _ASKED_WHICH_EVENT):
            candidate = _ACADEMIC_CONTEXT_PATTERN.sub("", text).strip(" .!?")
            if candidate and len(candidate.split()) <= 6:
                attributes["discipline"] = candidate
                attributes.setdefault("event", "exame")

    if "degree" in attributes and "event" not in attributes and any(marker in previous_normalized for marker in _EVENT_MARKERS):
        attributes["event"] = "exame"

    if "event" not in attributes and "discipline" not in attributes and not any(attributes.get(key) for key in ("degree", "course")):
        return {}

    if is_query:
        # A bare mention of "exame" inside a question is not new evidence —
        # only keep the candidate if something substantive beyond the
        # trivial event/default-status filler was actually captured.
        substantive_keys = set(attributes) - {"event", "status"}
        trivial_status = attributes.get("status") == "upcoming"
        if not substantive_keys and trivial_status:
            return {}

    if attributes.get("event") == "exame" and not is_query:
        has_concrete_detail = any(
            attributes.get(key)
            for key in ("discipline", "degree", "course", "date_reference", "context", "person", "location", "outcome")
        )
        if not has_concrete_detail and not explicit_status:
            return {}

    return attributes


# --- Sections 7-14: raw/canonical normalization ----------------------------
#
# Deterministic-only (no LLM): a small, explicit dictionary of accent and
# abbreviation corrections, applied word-by-word, plus title-casing for
# proper designations. This deliberately never invents or substitutes an
# entity — it can only fix spelling of the words the user actually typed.

NORMALIZATION_EXACT = "EXACT"
NORMALIZATION_SAFE = "SAFE_NORMALIZATION"
NORMALIZATION_AMBIGUOUS = "AMBIGUOUS"
NORMALIZATION_UNRESOLVED = "UNRESOLVED"

# Keys are fully de-accented, lowercase forms (as produced by _normalize),
# so this matches regardless of whether the user already typed some accents.
_ACCENT_CORRECTIONS = {
    "algoritmicas": "algorítmicas",
    "algoritmica": "algorítmica",
    "algoritmo": "algoritmo",
    "algoritmos": "algoritmos",
    "estrategias": "estratégias",
    "informatica": "informática",
    "informatico": "informático",
    "medico": "médico",
    "medica": "médica",
    "relatorio": "relatório",
    "matematica": "matemática",
    "fisica": "física",
    "quimica": "química",
    "historia": "história",
    "psicologia": "psicologia",
    "economia": "economia",
    "biologia": "biologia",
    "estatistica": "estatística",
    "engenharia": "engenharia",
}

_WORD_EXPANSIONS = {
    "msg": "mensagem",
}

_PHRASE_CORRECTIONS = {
    "mandar msg": "mandar uma mensagem",
    "enviar msg": "enviar uma mensagem",
}

_TITLE_CASE_STOPWORDS = {"de", "da", "do", "das", "dos", "e", "em", "a", "o", "as", "os", "para", "com"}


def _correct_word(word: str) -> tuple[str, bool]:
    key = _normalize(word)
    expanded = _WORD_EXPANSIONS.get(key)
    if expanded:
        return expanded, True
    corrected = _ACCENT_CORRECTIONS.get(key)
    if corrected and corrected != key:
        return corrected, True
    return word, False


def _looks_ambiguous_acronym(word: str) -> bool:
    return word.isupper() and 1 < len(word) <= 4 and word.isalpha()


def canonicalize_designation(raw: str) -> tuple[str, str]:
    """Canonicalizes a discipline/degree/course name: fixes spelling and
    capitalization, never substitutes it for a different designation.
    """
    text = raw.strip()
    if not text:
        return "", NORMALIZATION_UNRESOLVED

    words = text.split()
    if any(_looks_ambiguous_acronym(word) for word in words):
        # A short all-caps token ("EA") could mean several things; leave it
        # exactly as given rather than guessing an expansion.
        return text, NORMALIZATION_AMBIGUOUS

    changed = False
    corrected_words = []
    for word in words:
        corrected, was_changed = _correct_word(word)
        changed = changed or was_changed
        corrected_words.append(corrected)

    titled = [
        word if (word.lower() in _TITLE_CASE_STOPWORDS and index != 0) else (word[:1].upper() + word[1:])
        for index, word in enumerate(corrected_words)
    ]
    canonical = " ".join(titled)
    if canonical == text:
        return canonical, NORMALIZATION_EXACT
    return canonical, NORMALIZATION_SAFE


def canonicalize_name(raw: str) -> tuple[str, str]:
    """Canonicalizes a person's name: capitalization only, spelling untouched
    (we cannot safely guess whether a name has an accent or not).
    """
    text = raw.strip()
    if not text:
        return "", NORMALIZATION_UNRESOLVED
    titled = " ".join(word[:1].upper() + word[1:] if word else word for word in text.split())
    status = NORMALIZATION_EXACT if titled == text else NORMALIZATION_SAFE
    return titled, status


def canonicalize_action(raw: str, target_canonical: str = "") -> tuple[str, str]:
    """Canonicalizes a task's action phrase: expands unambiguous abbreviations
    and re-inserts the target's canonical capitalization, but keeps the verb
    phrase itself as the user framed it (destination, object and intent
    preserved verbatim in meaning).
    """
    text = raw.strip()
    if not text:
        return "", NORMALIZATION_UNRESOLVED

    lowered = text.lower()
    changed = False
    for phrase, replacement in _PHRASE_CORRECTIONS.items():
        if phrase in lowered:
            lowered = lowered.replace(phrase, replacement)
            changed = True

    words = lowered.split()
    corrected_words = []
    for word in words:
        corrected, was_changed = _correct_word(word)
        changed = changed or was_changed
        corrected_words.append(corrected)
    canonical = " ".join(corrected_words)

    if target_canonical:
        canonical = re.sub(
            rf"\b{re.escape(target_canonical.lower())}\b",
            target_canonical,
            canonical,
            flags=re.IGNORECASE,
        )

    if canonical == raw:
        return canonical, NORMALIZATION_EXACT
    return canonical, NORMALIZATION_SAFE if changed or canonical != raw else NORMALIZATION_EXACT


def normalize_candidate_fields(candidate: dict[str, str], fact_type: str) -> tuple[dict[str, str], dict[str, str]]:
    """Splits/normalizes a raw extraction candidate into (canonical_fields, raw_fields).

    canonical_fields is what gets stored in the existing columns (and is what
    the verbalizer must use); raw_fields captures the pre-normalization text
    for audit/reprocessing under "<field>_raw" keys.
    """
    canonical = dict(candidate)
    raw_fields: dict[str, str] = {}

    if fact_type == "academic_event":
        for field_name in ("discipline", "degree", "course"):
            value = candidate.get(field_name)
            if not value:
                continue
            canonical_value, _status = canonicalize_designation(value)
            raw_fields[f"{field_name}_raw"] = value
            canonical[field_name] = canonical_value
    elif fact_type == "task":
        target_value = candidate.get("target", "")
        target_canonical = ""
        if target_value:
            target_canonical, _status = canonicalize_name(target_value)
            raw_fields["target_raw"] = target_value
            canonical["target"] = target_canonical
        action_value = candidate.get("action", "")
        if action_value:
            action_canonical, _status = canonicalize_action(action_value, target_canonical)
            raw_fields["action_raw"] = action_value
            canonical["action"] = action_canonical

    return canonical, raw_fields


_BACKREFERENCE_MARKERS = ("que te", "que me", "falei", "falaste", "contei", "disse", "mencionei")


def _looks_like_backreference(value: str) -> bool:
    normalized = _normalize(value)
    return any(marker in normalized for marker in _BACKREFERENCE_MARKERS)


def _trim_trailing_time_reference(value: str) -> str:
    trimmed = value
    for pattern in (
        r"\s+(?:e|é|esta|está|foi)?\s*para\s+.*$",
        r"\s+(?:e|é|esta|está|foi)?\s*(?:no\s+)?dia\s+\d{1,2}.*$",
        r"\s+(?:e|é|esta|está|foi)?\s*(?:n[ao]\s+)?(?:segunda|ter[cç]a|quarta|quinta|sexta)-feira.*$",
        r"\s+n[ao]\s+semana.*$",
        r"\s+(?:hoje|amanha|depois de amanha).*$",
        r"\s+(?:e|é|esta|está|foi)\s*$",
    ):
        trimmed = re.sub(pattern, "", trimmed, flags=re.IGNORECASE)
    return trimmed.strip(" .,!?:;")


# --- Section 3/13: task extraction from casual statements -------------------

_TARGET_PERSON_PATTERN = re.compile(
    r"\b(?:ao|à|com\s+a|com\s+o)\s+([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\wÁÀÂÃÉÊÍÓÔÕÚÇáàâãéêíóôõúç]{1,40})"
)

_REMINDER_PATTERN = re.compile(r"^lembra[- ]me\s+(?:hoje\s+|amanh[aã]\s+)?de\s+(.+)$", re.IGNORECASE)
_OBLIGATION_PATTERNS = (
    re.compile(r"^tenho\s+(?:de|que)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^preciso\s+de\s+(.+)$", re.IGNORECASE),
)
_DESIRE_PATTERN = re.compile(r"^quero\s+(.+)$", re.IGNORECASE)


def extract_task_candidate(message: str) -> dict[str, str]:
    """Best-effort extraction of a pending task from a casual statement.

    Only fires on an explicit obligation/desire/reminder verb pattern, so a
    plain fact about someone ("O Pedro trabalha comigo.") is never mistaken
    for a task — there is simply no matching prefix to trigger on.
    """
    text = message.strip()
    normalized = _normalize(text)
    if is_memory_recall_question(normalized):
        return {}

    attributes: dict[str, str] = {}

    reminder_match = _REMINDER_PATTERN.match(text)
    if reminder_match:
        attributes["action"] = reminder_match.group(1).strip(" .!?")
        attributes["reminder_requested"] = "true"
    else:
        for pattern in _OBLIGATION_PATTERNS:
            match = pattern.match(text)
            if match:
                attributes["action"] = match.group(1).strip(" .!?")
                break
        else:
            desire_match = _DESIRE_PATTERN.match(text)
            if desire_match:
                attributes["action"] = desire_match.group(1).strip(" .!?")
                attributes["context"] = "desire"

    if not attributes.get("action"):
        return {}

    date_match = _DATE_REFERENCE_PATTERN.search(text)
    if date_match:
        attributes["date_reference"] = date_match.group(1)

    person_match = _TARGET_PERSON_PATTERN.search(attributes["action"])
    if person_match:
        attributes["target"] = person_match.group(1).strip(" .,!?:;")

    attributes["status"] = "pending"
    attributes["raw_user_text"] = text
    return attributes


# --- Section 5: grammatical person adaptation (evidence -> natural speech) -

_FIRST_TO_SECOND_PERSON = (
    (re.compile(r"^tenho\s+que\b", re.IGNORECASE), "tens que"),
    (re.compile(r"^tenho\s+de\b", re.IGNORECASE), "tens de"),
    (re.compile(r"^preciso\s+de\b", re.IGNORECASE), "precisas de"),
    (re.compile(r"^quero\b", re.IGNORECASE), "querias"),
)


def convert_first_person_to_second_person(text: str) -> str:
    """Rewrites a first-person modal opener ("preciso de...") into second
    person ("precisas de..."), for the rare case raw text must be echoed.

    Deliberately a small, explicit table (not general conjugation) — the
    task renderer avoids needing this by reconstructing from clean slots
    instead of reusing raw text, so this exists mainly as a defined,
    testable capability for future memory types that may need it.
    """
    for pattern, replacement in _FIRST_TO_SECOND_PERSON:
        if pattern.match(text):
            converted = pattern.sub(replacement, text, count=1)
            if text[:1].isupper():
                converted = converted[:1].upper() + converted[1:]
            return converted
    return text


# --- Section 4/8: retrieval -------------------------------------------------


@dataclass
class MemoryRetrieval:
    grounded: bool
    ambiguous: bool
    facts: list[str]
    sources: list[str]
    confidence: float
    attributes_covered: set[str]
    fallback_message: str
    disambiguation_question: str = ""
    matched_ids: list[str] = field(default_factory=list)
    rendered_answer: str = ""

    @property
    def final_answer(self) -> str:
        """The one deterministic text to show the user — never an LLM call."""
        if self.ambiguous:
            return self.disambiguation_question
        if self.grounded:
            return self.rendered_answer
        return self.fallback_message


def build_memory_retrieval(
    requested_attributes: set[str],
    history_text: str,
    structured_facts: list[StructuredFact],
) -> MemoryRetrieval:
    if not structured_facts:
        history_candidate = extract_academic_event_candidate(history_text)
        if history_candidate:
            return _retrieval_from_candidate(requested_attributes, history_candidate, source="CURRENT_HISTORY", confidence=0.7)
        return MemoryRetrieval(
            grounded=False,
            ambiguous=False,
            facts=[],
            sources=[],
            confidence=0.0,
            attributes_covered=set(),
            fallback_message="Não encontrei nenhuma memória sobre isso.",
        )

    distinct_disciplines = {fact.discipline for fact in structured_facts if fact.discipline}
    if len(distinct_disciplines) > 1 and not any(
        _normalize(discipline) in history_text for discipline in distinct_disciplines
    ):
        options = ", ".join(sorted(distinct_disciplines))
        return MemoryRetrieval(
            grounded=False,
            ambiguous=True,
            facts=[],
            sources=[f"PERSISTENT_MEMORY:{fact.id}" for fact in structured_facts],
            confidence=0.0,
            attributes_covered=set(),
            fallback_message="",
            disambiguation_question=f"Tenho mais do que um exame guardado: {options}. A qual te referes?",
        )

    best = structured_facts[0]
    covered = best.known_attributes()

    if requested_attributes and not (requested_attributes & covered):
        target = next(iter(requested_attributes))
        label = ATTRIBUTE_LABELS.get(target, "essa informação")
        event_label = best.event or "um acontecimento"
        return MemoryRetrieval(
            grounded=False,
            ambiguous=False,
            facts=[],
            sources=[f"PERSISTENT_MEMORY:{best.id}"],
            confidence=best.confidence,
            attributes_covered=covered,
            fallback_message=f"Lembro-me de teres mencionado {event_label}, mas não tenho {label} guardada.",
        )

    if not covered:
        return MemoryRetrieval(
            grounded=False,
            ambiguous=False,
            facts=[],
            sources=[],
            confidence=0.0,
            attributes_covered=set(),
            fallback_message="Não encontrei nenhuma memória sobre isso.",
        )

    rendered = render_academic_event_answer(best, requested_attributes)
    if not rendered:
        # We know *that* something happened (the event), but no informative
        # attribute beyond it — that is a vague memory, not a grounded answer.
        return MemoryRetrieval(
            grounded=False,
            ambiguous=False,
            facts=[],
            sources=[f"PERSISTENT_MEMORY:{best.id}"],
            confidence=best.confidence,
            attributes_covered=covered,
            fallback_message=(
                f"Lembro-me de teres mencionado {best.event or 'isso'}, "
                "mas não tenho a disciplina guardada."
            ),
        )

    return MemoryRetrieval(
        grounded=True,
        ambiguous=False,
        facts=_facts_from_fact(best),
        sources=[f"PERSISTENT_MEMORY:{best.id}"],
        confidence=best.confidence,
        attributes_covered=covered,
        fallback_message=f"Lembro-me de teres mencionado {best.event or 'isso'}, mas não tenho mais detalhes guardados.",
        matched_ids=[str(best.id)],
        rendered_answer=rendered,
    )


def _retrieval_from_candidate(requested_attributes: set[str], candidate: dict[str, str], source: str, confidence: float) -> MemoryRetrieval:
    covered = {key for key, value in candidate.items() if value}
    if requested_attributes and not (requested_attributes & covered):
        target = next(iter(requested_attributes))
        label = ATTRIBUTE_LABELS.get(target, "essa informação")
        event_label = candidate.get("event", "um acontecimento")
        return MemoryRetrieval(
            grounded=False,
            ambiguous=False,
            facts=[],
            sources=[source],
            confidence=confidence,
            attributes_covered=covered,
            fallback_message=f"Lembro-me de teres mencionado {event_label}, mas não tenho {label} guardada.",
        )
    rendered = render_academic_event_answer(_fact_from_candidate(candidate), requested_attributes)
    if not rendered:
        return MemoryRetrieval(
            grounded=False,
            ambiguous=False,
            facts=[],
            sources=[],
            confidence=0.0,
            attributes_covered=set(),
            fallback_message="Não encontrei nenhuma memória sobre isso.",
        )
    return MemoryRetrieval(
        grounded=True,
        ambiguous=False,
        facts=[],
        sources=[source],
        confidence=confidence,
        attributes_covered=covered,
        fallback_message="Não tenho essa informação guardada.",
        rendered_answer=rendered,
    )


def _fact_from_candidate(candidate: dict[str, str]) -> StructuredFact:
    return StructuredFact(id=0, fact_type="academic_event", **{k: v for k, v in candidate.items() if k in STRUCTURED_FACT_ATTRIBUTES})


def _facts_from_fact(fact: StructuredFact) -> list[str]:
    """Evidence-level sentences kept for telemetry/audit — never sent to an LLM."""
    facts = []
    if fact.discipline:
        facts.append(f"O exame era de {fact.discipline}.")
    if fact.degree:
        facts.append(f"Estava associado à licenciatura em {fact.degree}.")
    if fact.status == "failed":
        facts.append("O resultado foi negativo (chumbaste).")
    elif fact.status == "completed":
        facts.append("Já concluíste esse exame.")
    if fact.date_reference:
        facts.append(f"Era por volta de {fact.date_reference}.")
    return facts


# --- Section 4/7: deterministic verbalization ("MemoryVerbalizer") --------


def render_academic_event_answer(fact: StructuredFact, requested_attributes: set[str]) -> str:
    """Turns a stored academic_event fact into natural PT-PT text.

    Purely template-based — no LLM involved — so it can never add an
    attribute, a person, or a date that isn't already on the fact.
    """
    if requested_attributes == {"discipline"}:
        return f"Era {fact.discipline}." if fact.discipline else ""
    if requested_attributes == {"date_reference"}:
        return f"Tinhas dito que era {fact.date_reference}." if fact.date_reference else ""
    if requested_attributes == {"degree"}:
        return f"Era da licenciatura em {fact.degree}." if fact.degree else ""
    if requested_attributes == {"status"}:
        return _academic_status_sentence(fact)

    # Generic recall ("Lembras-te do exame?" / "Que sabes sobre o exame?"):
    # answer with everything grounded, omit anything that isn't.
    if not fact.discipline:
        return ""
    sentence = f"Sim. Era o exame de {fact.discipline}"
    if fact.degree:
        sentence += f" da tua licenciatura em {fact.degree}."
    else:
        sentence += "."
    if fact.date_reference:
        sentence += f" Tinhas dito que era {fact.date_reference}."
    status_sentence = _academic_status_sentence(fact)
    if status_sentence:
        sentence += f" {status_sentence}"
    return sentence


def _academic_status_sentence(fact: StructuredFact) -> str:
    if fact.status == "failed":
        return "Chumbaste nesse exame."
    if fact.status == "completed":
        return "Já concluíste esse exame."
    if fact.status == "upcoming":
        return "Ainda estava para acontecer."
    return ""


_TASK_COUNT_WORDS = {1: "uma", 2: "duas", 3: "três", 4: "quatro", 5: "cinco", 6: "seis"}


def render_task_list_answer(tasks: list[StructuredFact]) -> str:
    """Renders pending tasks conversationally — never a raw list of stored text."""
    pending = [task for task in tasks if task.status == "pending" and task.action]
    if not pending:
        return "Não tens tarefas pendentes guardadas."
    if len(pending) == 1:
        return _render_single_task(pending[0])

    actions = [task.action for task in pending]
    joined = ", ".join(actions[:-1]) + f" e {actions[-1]}"
    if len(actions) <= 2:
        return f"Tens de {joined}."
    count_word = _TASK_COUNT_WORDS.get(len(actions), str(len(actions)))
    return f"Tens {count_word} tarefas pendentes: {joined}."


def _render_single_task(task: StructuredFact) -> str:
    # "Pediste-me para te lembrar..." is only ever used when the user
    # explicitly asked for a reminder — never inferred from a plain "tenho de".
    if task.reminder_requested == "true":
        return f"Pediste-me para te lembrar de {task.action}."
    if task.context == "desire":
        return f"Querias {task.action}."
    return f"Tens de {task.action}."


def build_task_retrieval(tasks: list[StructuredFact]) -> MemoryRetrieval:
    pending = [task for task in tasks if task.status == "pending" and task.action]
    rendered = render_task_list_answer(tasks)
    if not pending:
        return MemoryRetrieval(
            grounded=False,
            ambiguous=False,
            facts=[],
            sources=[],
            confidence=0.0,
            attributes_covered=set(),
            fallback_message=rendered,
        )
    return MemoryRetrieval(
        grounded=True,
        ambiguous=False,
        facts=[task.action for task in pending],
        sources=[f"PERSISTENT_MEMORY:{task.id}" for task in pending],
        confidence=min(task.confidence for task in pending),
        attributes_covered={"pending_tasks"},
        fallback_message=rendered,
        matched_ids=[str(task.id) for task in pending],
        rendered_answer=rendered,
    )


# --- Section 7/9: guarding against unsupported memory claims ---------------

_MEMORY_CLAIM_MARKERS = (
    "lembro-me",
    "recordo-me",
    "ja me tinhas dito",
    "ja me falaste disso",
    "lembrei-me agora",
    "sim, ja sei",
    "era ",
    "chamava-se ",
    "foi em ",
)


def detect_unsupported_memory_claim(response: str, grounding_sources: list[str]) -> str:
    """Flags a response that talks like it remembers something with zero evidence behind it."""
    if grounding_sources:
        return ""
    normalized = _normalize(response)
    for marker in _MEMORY_CLAIM_MARKERS:
        if marker in normalized:
            return f"alegação de memória sem evidência ('{marker.strip()}')"
    return ""


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_marks.split())
