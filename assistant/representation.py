"""Phase 3 of the Cognitive Workspace Blueprint v2 -- REPRESENTATION
TRANSFORMATION.

Proves representation is a transformable VIEW over stable WorkSession data:
the same semantic objects move between styles (cards / list / side_by_side
/ conversation_only) without reconstruction, re-extraction, or identity
loss. This is about representation identity and transformability, not a
new domain.

This module is deliberately independent of assistant.conversation -- it
never imports it, at module level or lazily (mirrors the existing
assistant/interaction_interpreter.py pattern, one level further: there is
nothing here worth reusing FROM conversation.py, so there is no import to
break a cycle over). Every function below takes a `work_session` duck-typed
to expose `.objects` (dict[str, SemanticObject]) and `.representation`
(RepresentationState | None); nothing here reads or writes
WorkSession.knowledge, .active_task, .document_context, or any task/
document field -- that is the whole point of the "representation changes
only representation state" invariant (section 2 of the phase spec).

Styles supported in this phase (see SUPPORTED_STYLES): cards, list,
side_by_side, conversation_only. mind_map / chart / timeline /
decision_matrix and any other style word the Interaction Interpreter
already recognizes (section 11, Phase 2 vocabulary) are explicitly OUT of
scope here -- unsupported styles raise rather than silently no-op.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from dataclasses import dataclass, field

from assistant.text_matching import contains_any_phrase, normalize_for_matching

SUPPORTED_STYLES = ("cards", "list", "side_by_side", "conversation_only")

SOURCE_AUTO = "auto"
SOURCE_VOICE_COMMAND = "voice_command"
SOURCE_ECHO_SUGGESTION = "echo_suggestion"
SOURCE_POINTER = "pointer"


@dataclass
class SemanticObject:
    """One backend-visible semantic object -- the minimum registry entry
    needed to prove representation transformation (section 3). `id` is
    reused verbatim from whatever existing identity it already has
    wherever possible (e.g. a task artifact's `active_artifact_id`) --
    Phase 3 never mints a second, parallel id for the same visual object,
    and never carries frontend geometry (position/size stay a dead,
    frontend-only concern -- see WorkspaceObject in
    echo_workspace_objects.js)."""

    id: str
    kind: str = "object"
    title: str = ""
    payload: dict[str, object] = field(default_factory=dict)
    task_id: str = ""
    state: str = "active"


@dataclass
class RepresentationState:
    """Semantic-only representation instructions (section 4): a style plus
    which already-known object ids are visible / major / hidden, plus
    minimal grouping/ordering/param state. NEVER a pixel, a coordinate, or
    a layout transition -- the frontend computes 100% of geometry, exactly
    as it already does for WorkspaceObject/WorkspaceScene."""

    style: str = "cards"
    style_params: dict[str, object] = field(default_factory=dict)
    visible_object_ids: list[str] = field(default_factory=list)
    major_object_ids: list[str] = field(default_factory=list)
    hidden_object_ids: list[str] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    ordering: list[str] = field(default_factory=list)
    reasoning_visible: bool = False
    source: str = SOURCE_AUTO
    # One level of history -- enough for "não quero ver nada" ->
    # conversation_only -> "mostra outra vez" to restore exactly the
    # representation that was active right before hiding, with no
    # reconstruction (section 6). Deliberately NOT a general undo stack --
    # each transform simply keeps a pointer to the state it replaced.
    previous: "RepresentationState | None" = None


def register_object(work_session, obj: SemanticObject) -> SemanticObject:
    """Adds/updates one semantic object in the registry (section 3).
    Registering (or re-registering, e.g. a task artifact's later revision)
    an object never touches representation state by itself -- it does not
    make the object visible anywhere until a transform_representation call
    says so."""
    work_session.objects[obj.id] = obj
    return obj


def transform_representation(
    work_session,
    style: str,
    *,
    style_params: dict[str, object] | None = None,
    visible_ids: list[str] | None = None,
    major_ids: list[str] | None = None,
    hidden_ids: list[str] | None = None,
    ordering: list[str] | None = None,
    source: str = SOURCE_AUTO,
) -> RepresentationState:
    """THE single central representation-transition operation (section 5).
    Mutates ONLY work_session.representation -- never .objects, .knowledge,
    or any task/document field. Every voice-command bridge call and every
    Phase 3 test goes through this one function; there is no per-style
    transform_to_cards()/transform_to_list() that would duplicate state
    logic (style-specific RENDERING is a frontend concern, not this)."""
    if style not in SUPPORTED_STYLES:
        raise ValueError(f"unsupported Phase 3 representation style: {style!r}")

    known_ids = list(work_session.objects.keys())
    current = work_session.representation

    if style == "conversation_only":
        # Hidden defaults to EVERY currently-known object -- "não quero ver
        # nada" hides the whole workspace, it does not delete anything
        # (section 7): known_ids still exist in work_session.objects.
        new_hidden = list(hidden_ids) if hidden_ids is not None else list(known_ids)
        new_visible: list[str] = list(visible_ids) if visible_ids is not None else []
        new_major: list[str] = list(major_ids) if major_ids is not None else []
    elif style == "side_by_side":
        new_major = list(major_ids) if major_ids is not None else known_ids[:2]
        new_hidden = list(hidden_ids) if hidden_ids is not None else []
        new_visible = (
            list(visible_ids) if visible_ids is not None
            else [oid for oid in known_ids if oid not in new_hidden]
        )
    else:  # cards | list
        new_hidden = list(hidden_ids) if hidden_ids is not None else []
        new_visible = (
            list(visible_ids) if visible_ids is not None
            else [oid for oid in known_ids if oid not in new_hidden]
        )
        new_major = list(major_ids) if major_ids is not None else []

    new_state = RepresentationState(
        style=style,
        style_params=dict(style_params) if style_params is not None else {},
        visible_object_ids=new_visible,
        major_object_ids=new_major,
        hidden_object_ids=new_hidden,
        groups=dict(current.groups) if current is not None else {},
        ordering=list(ordering) if ordering is not None else list(new_visible),
        reasoning_visible=current.reasoning_visible if current is not None else False,
        source=source,
        previous=current,
    )
    work_session.representation = new_state
    return new_state


def set_reasoning_visible(work_session, visible: bool, *, source: str = SOURCE_VOICE_COMMAND) -> RepresentationState | None:
    """Phase 4 (section 19): toggles ONLY reasoning_visible on the CURRENT
    representation -- never touches style/visible/major/hidden ids. A
    no-op (returns None) when there is no active representation yet, since
    there is nothing to externalize a rationale about. Keeps the same
    one-level `.previous` history any other transform leaves behind, so
    restoring afterward still works exactly as it does for any other
    change."""
    current = work_session.representation
    if current is None:
        return None
    new_state = dataclasses.replace(current, reasoning_visible=bool(visible), source=source, previous=current)
    work_session.representation = new_state
    return new_state


def restore_representation(work_session, *, source: str = SOURCE_VOICE_COMMAND) -> RepresentationState | None:
    """Sections 6/8: restores whatever representation was active
    immediately before the current one -- e.g. undoing "não quero ver
    nada" with "mostra outra vez". A no-op (returns the current state
    unchanged) if there is nothing to restore to. Never touches .objects
    or .knowledge either."""
    current = work_session.representation
    if current is None or current.previous is None:
        return current
    restored = dataclasses.replace(current.previous, source=source)
    work_session.representation = restored
    return restored


def resolve_mentioned_object_ids(raw_message: str, work_session) -> list[str]:
    """Generic, domain-agnostic name resolution used by the Phase 3 voice
    bridge's side_by_side handling (section 11): finds which ALREADY-
    REGISTERED semantic objects are mentioned by title in raw_message, in
    the order they appear in the text. Works identically for "Suécia" /
    "Itália" / "Açores" synthetic test objects as for any other object's
    title -- there is no travel-specific (or any other domain-specific)
    logic here, just a substring match against known titles."""
    normalized_message = normalize_for_matching(raw_message)
    matches: list[tuple[int, str]] = []
    for obj_id, obj in work_session.objects.items():
        title = normalize_for_matching(obj.title) if obj.title else ""
        if not title:
            continue
        index = normalized_message.find(title)
        if index >= 0:
            matches.append((index, obj_id))
    matches.sort(key=lambda pair: pair[0])
    return [obj_id for _, obj_id in matches]


# --- Phase 3B integration fix (section "FIX 3"): the smallest domain-
# agnostic mechanism to turn an explicitly-enumerated list of named
# alternatives ("vamos pensar em três opções: Suécia, Itália e Açores")
# into stable generic SemanticObjects. Deliberately NOT a travel/decision/
# any-other-domain feature: it never infers a fact about the alternatives,
# never validates them against any knowledge source, and the trigger
# phrases below name no domain at all -- only the generic verbs
# "pensar"/"comparar"/"considerar" a user would use to introduce ANY set
# of named options (travel destinations, job candidates, book titles...).

ALTERNATIVES_TRIGGER_PHRASES = (
    "pensar em", "quero comparar", "vamos comparar", "comparar as opcoes",
    "quero considerar", "vamos considerar", "considerar as opcoes",
)

_ALTERNATIVE_SPLIT_RE = re.compile(r",| e (?=\w)| com (?=\w)|\bvs\.?\s+(?=\w)", re.IGNORECASE)
_LOWERCASE_OPTION_STOPWORDS = {
    "preciso", "quero", "queria", "gostava", "vamos", "comparar", "adicionar", "adiciona",
    "acrescenta", "junta", "telefone", "telemovel", "telemóvel", "novo", "nova", "opcoes",
    "opções", "documentos", "ficheiros",
}

# Phase 4 integration fix (section 9): a trailing style qualifier ("...em
# cartões") describes HOW to show the enumerated alternatives, not one more
# alternative -- stripped before splitting so the LAST named alternative
# never absorbs it as part of its own title ("Guarda em cartões" instead of
# "Guarda"). Small, local list (not imported from interaction_interpreter's
# own style-word table -- this module never imports assistant.conversation
# or interaction_interpreter, see module docstring) covering the same
# synonyms the rest of the app already recognizes as style words.
_TRAILING_STYLE_SUFFIX_RE = re.compile(
    r"\s+em\s+(cart[oõ]es|cards|caixas|lista|tabela|mapa|clusters|grafico|gr[aá]fico|barras)\s*[.!?]*$",
    re.IGNORECASE,
)


def _strip_trailing_style_suffix(text: str) -> str:
    return _TRAILING_STYLE_SUFFIX_RE.sub("", text)


def _strip_leading_option_filler(text: str) -> str:
    return re.sub(r"^(?:o|a|os|as|um|uma|uns|umas|tamb[ée]m)\s+", "", text.strip(), flags=re.IGNORECASE)


def _looks_like_named_option(text: str) -> bool:
    if not text or len(text) > 40:
        return False
    if text[0].isupper() or any(char.isupper() for char in text[1:]):
        return True
    normalized = normalize_for_matching(text)
    words = [word for word in re.findall(r"\b[\wÀ-ÿ]{2,}\b", normalized)]
    return bool(words) and len(words) <= 3 and not any(word in _LOWERCASE_OPTION_STOPWORDS for word in words)


def _normalize_option_title(text: str) -> str:
    cleaned = text.strip(" .!?…")
    if not cleaned:
        return ""
    if cleaned[0].isupper() or any(char.isupper() for char in cleaned[1:]):
        return cleaned
    return " ".join(part[:1].upper() + part[1:] for part in cleaned.split())


def slugify_title(title: str) -> str:
    """A stable, id-safe slug for a title -- the SAME title always
    produces the SAME slug, so re-mentioning an already-materialized
    alternative never creates a second, duplicate SemanticObject."""
    normalized = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(char for char in normalized if not unicodedata.combining(char))
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug or "option"


def extract_explicit_alternatives(raw_message: str) -> list[str]:
    """Purely structural extraction: a generic trigger phrase, followed by
    a comma/" e "-separated list of capitalized tokens (the light,
    domain-agnostic heuristic for "these look like named alternatives").
    Returns [] unless at least two candidates are found -- a single
    capitalized word after "pensar em" is far too weak a signal on its
    own. Never looks up or infers anything about the extracted names."""
    normalized = normalize_for_matching(raw_message)
    if not contains_any_phrase(normalized, ALTERNATIVES_TRIGGER_PHRASES):
        return []

    tail = raw_message
    if ":" in raw_message:
        tail = raw_message.rsplit(":", 1)[1]
    else:
        lowered = raw_message.lower()
        for phrase in ALTERNATIVES_TRIGGER_PHRASES:
            index = lowered.find(phrase)
            if index != -1:
                tail = raw_message[index + len(phrase):]
                break

    tail = _strip_trailing_style_suffix(tail)
    candidates: list[str] = []
    for part in _ALTERNATIVE_SPLIT_RE.split(tail):
        cleaned = _strip_leading_option_filler(part.strip(" .!?…"))
        if _looks_like_named_option(cleaned):
            candidates.append(_normalize_option_title(cleaned))
    return candidates if len(candidates) >= 2 else []


# --- Phase 3C (section 10/11), generalized in the Phase 4 manual-test fix
# (section 3): "add one (or more) alternative(s) to an already-materialized
# option set" -- reuses the same domain-agnostic idea as
# extract_explicit_alternatives above (generic trigger verb + a
# comma/" e "-separated list of capitalized names), anchored on the "add"
# trigger vocabulary instead of the "introduce a new set" one.

ADD_ALTERNATIVE_TRIGGER_PHRASES = (
    "acrescenta tambem", "acrescenta", "adiciona tambem", "adiciona",
    "junta tambem", "junta",
)

COMPARISON_CRITERIA_TRIGGER_PHRASES = (
    "vamos comparar", "quero comparar", "preciso de comparar", "compara", "comparar",
)

_ADD_TARGET_SUFFIX_RE = re.compile(
    r"\s+(?:aos|às|as|ao|à|a\s+os|a\s+as|nos|nas|no|na)\s+(.+?)\s*[.!?…]*$",
    re.IGNORECASE,
)


def split_add_target_suffix(text: str) -> tuple[str, str]:
    """Splits "Pixel aos telefones" into ("Pixel", "telefones").

    The suffix names the destination collection/activity, not the new
    object's title. This stays structural: no domain word is hardcoded.
    """
    cleaned = text.strip(" .!?…")
    match = _ADD_TARGET_SUFFIX_RE.search(cleaned)
    if not match:
        return cleaned, ""
    title_part = cleaned[:match.start()].strip(" .!?…")
    target_part = match.group(1).strip(" .!?…")
    return title_part, target_part


def extract_add_target_reference(raw_message: str) -> str:
    normalized = normalize_for_matching(raw_message)
    if not contains_any_phrase(normalized, ADD_ALTERNATIVE_TRIGGER_PHRASES):
        return ""
    lowered = raw_message.lower()
    tail = raw_message
    for phrase in sorted(ADD_ALTERNATIVE_TRIGGER_PHRASES, key=len, reverse=True):
        index = lowered.find(phrase)
        if index != -1:
            tail = raw_message[index + len(phrase):]
            break
    tail = re.sub(r"^\s*tamb[ée]m\b", "", tail, flags=re.IGNORECASE)
    _title, target = split_add_target_suffix(_strip_trailing_style_suffix(tail))
    return target

def extract_added_alternatives(raw_message: str) -> list[str]:
    """Phase 4 manual-test fix (section 3): generalizes the original
    single-name extractor to an explicit ENUMERATION -- "acrescenta
    Madeira, Islândia, Noruega e Japão" -- reusing the exact same comma/
    " e "-separated capitalized-token split extract_explicit_alternatives
    already uses for introducing a brand-new set, just anchored on the
    "add"-vocabulary trigger instead. Returns [] unless at least ONE
    candidate is found -- unlike extract_explicit_alternatives' own >=2
    floor (that floor exists there to avoid over-triggering on a single
    bare capitalized word when there is no other signal that a NEW set is
    being introduced; here the explicit "add" trigger verb is itself
    already a confident, narrow signal, so a single added name is a
    perfectly ordinary case: "acrescenta Madeira")."""
    normalized = normalize_for_matching(raw_message)
    if not contains_any_phrase(normalized, ADD_ALTERNATIVE_TRIGGER_PHRASES):
        return []

    lowered = raw_message.lower()
    tail = raw_message
    for phrase in sorted(ADD_ALTERNATIVE_TRIGGER_PHRASES, key=len, reverse=True):
        index = lowered.find(phrase)
        if index != -1:
            tail = raw_message[index + len(phrase):]
            break
    # The combined "acrescenta também" trigger phrase only matches when the
    # message is written WITHOUT the accent ("tambem") -- an accented
    # "também" (the normal spelling) falls through to the bare "acrescenta"
    # trigger instead, leaving "também" stuck as a leading filler word in
    # front of the first name. Strip it explicitly so "acrescenta também
    # Madeira" isolates "Madeira" the same way "acrescenta Madeira" would;
    # this is the ONE filler word the trigger vocabulary already
    # anticipates (see ADD_ALTERNATIVE_TRIGGER_PHRASES), never a general
    # stop-word list.
    tail = re.sub(r"^\s*tamb[ée]m\b", "", tail, flags=re.IGNORECASE)
    tail = _strip_trailing_style_suffix(tail)
    tail, _target_reference = split_add_target_suffix(tail)

    candidates: list[str] = []
    for part in _ALTERNATIVE_SPLIT_RE.split(tail):
        cleaned = _strip_leading_option_filler(part.strip(" .!?…"))
        if _looks_like_named_option(cleaned):
            candidates.append(_normalize_option_title(cleaned))
    return candidates


def extract_comparison_criteria(raw_message: str) -> list[str]:
    """Extracts generic comparison dimensions, not alternatives.

    This is intentionally structural and domain-agnostic. A phrase like
    "comparar por preço e consumo" or, when an existing comparison is
    already referenced by another clause, "vamos comparar bateria e
    armazenamento" yields criteria objects. It never decides what the
    options themselves are.
    """
    normalized = normalize_for_matching(raw_message)
    if not contains_any_phrase(normalized, COMPARISON_CRITERIA_TRIGGER_PHRASES):
        return []

    lowered = raw_message.lower()
    tail = raw_message
    explicit_axis = re.search(r"\b(?:por|em)\s+(.+)$", raw_message, flags=re.IGNORECASE)
    if explicit_axis:
        tail = explicit_axis.group(1)
    else:
        for phrase in sorted(COMPARISON_CRITERIA_TRIGGER_PHRASES, key=len, reverse=True):
            index = lowered.find(phrase)
            if index != -1:
                tail = raw_message[index + len(phrase):]
                break

    tail = _strip_trailing_style_suffix(tail)
    tail = re.sub(r"\b(?:aos|às|as|ao|à|nos|nas|no|na)\s+.+$", "", tail, flags=re.IGNORECASE)
    candidates: list[str] = []
    for part in _ALTERNATIVE_SPLIT_RE.split(tail):
        cleaned = _strip_leading_option_filler(part.strip(" .!?…"))
        if _looks_like_named_option(cleaned):
            candidates.append(_normalize_option_title(cleaned))
    return candidates if len(candidates) >= 1 else []


def strip_add_alternatives_phrase(raw_message: str, added_titles: list[str]) -> str:
    """Section 12's false-success-prevention support: removes the matched
    "add" trigger phrase and EVERY extracted title from raw_message,
    leaving whatever text (if any) the utterance ALSO contained -- a
    generic, structural signal that something beyond the supported "add
    option(s)" sub-intent was requested, even though today's Interaction
    Interpreter has no axis to classify what it was (spatial-by-name,
    grouping...). Stripping only the FIRST title here was the Phase 4
    manual-test bug: for a pure multi-item enumeration, every title after
    the first was left in the residual and misread as a second,
    unaddressed sub-intent. Never tries to interpret the residual, only
    measures whether it is substantial enough to be a real, separate ask."""
    lowered = raw_message.lower()
    residual = raw_message
    for phrase in sorted(ADD_ALTERNATIVE_TRIGGER_PHRASES, key=len, reverse=True):
        index = lowered.find(phrase)
        if index != -1:
            residual = raw_message[:index] + raw_message[index + len(phrase):]
            break
    for title in added_titles:
        residual = residual.replace(title, "")
    # Phase 4 integration fix (section 9): the trailing style qualifier
    # ("em cartões") this module's extraction already strips off the last
    # title is not a second, unaddressed sub-intent either -- without this,
    # every multi-name "adiciona ... em cartões" would misreport itself as
    # partial ("ainda não sei mover cartões...") purely because the style
    # word is no longer inside one of the extracted titles to cancel out.
    residual = _strip_trailing_style_suffix(residual)
    return residual.strip(" ,.!?…")


# --- Phase 4 (Cognitive Workspace Blueprint v2): ATTENTION + PROACTIVE
# REPRESENTATION + SPACE PRESSURE. Everything below stays in THIS module
# (not attention.py) because it answers "how should information be shown"
# (representation), not "how important is it" (attention) -- section 3's
# invariant. RepresentationConstraint activates WorkSession's Phase-1-
# reserved `constraints` field exactly the way Phase 3 activated
# `representation` itself: the reserved placeholder's type changes from a
# generic bag (`list[object]`) to the concrete structure the phase that
# needs it actually defines, never a second parallel field.

@dataclass
class RepresentationConstraint:
    """Sections 9/10: a session/task-level, explicit, user-set constraint
    on HOW auto-representation is allowed to behave -- never persisted to
    long-term memory yet (section 27). `avoid_style`/`prefer_style` are
    deliberately just style names (not a stack/history) -- section 10 only
    asks for "at minimum" these fields, and the phase spec explicitly warns
    against building a bigger constraint system than that."""

    avoid_style: str = ""
    prefer_style: str = ""
    conversation_only: bool = False
    keep_visible_ids: list[str] = field(default_factory=list)
    do_not_minimize_ids: list[str] = field(default_factory=list)
    source: str = SOURCE_VOICE_COMMAND


@dataclass
class SpacePressureState:
    """Section 11: the ONLY thing the frontend is allowed to tell the
    backend about layout -- a semantic capacity signal, never a pixel or a
    coordinate. `pressure` is one of "low" | "medium" | "high"."""

    pressure: str = "low"
    visible_object_count: int = 0
    major_surface_count: int = 0
    overflow_risk: bool = False


SPACE_PRESSURE_LEVELS = ("low", "medium", "high")


@dataclass
class RepresentationProposal:
    """Section 7's deterministic decision contract's return value.
    `should_apply` lets the caller distinguish "here is my best guess, but
    I'm not confident enough to act on it unasked" (should_apply=False)
    from "this is safe to apply automatically" (should_apply=True) --
    section 8's reversible-initiative rule."""

    style: str
    style_params: dict[str, object] = field(default_factory=dict)
    reason_code: str = ""
    confidence: float = 0.0
    should_apply: bool = False


# Generic comparison-intent vocabulary (section 6/7): names no domain, just
# the verbs a user would say about ANY pair of things being weighed against
# each other ("comparar A e C", "quero comparar", "compara estas duas").
COMPARISON_INTENT_PHRASES = ("comparar", "compara ", "comparação", "comparacao")


def message_implies_comparison(raw_message: str) -> bool:
    return contains_any_phrase(normalize_for_matching(raw_message), COMPARISON_INTENT_PHRASES)


def choose_representation(work_session, context: dict[str, object]) -> RepresentationProposal:
    """Section 7: a small, deterministic, GENERIC representation policy --
    reasons about semantic SHAPE (object count, comparison intent), never a
    domain name ("travel", "email"...). No LLM call, ever.

    `context` may carry (all optional):
      object_count: int -- how many semantic objects are in play right now.
      comparison_intent: bool -- did the utterance sound like "compare
        these"?

    Section 9/10 precedence: an explicit RepresentationConstraint (already
    set by a PRIOR voice command -- see
    assistant.conversation._try_representation_command) always wins over
    whatever this function would otherwise propose; auto-representation
    must never immediately undo an explicit user choice."""
    constraint = work_session.constraints
    object_count = int(context.get("object_count", 0) or 0)
    comparison_intent = bool(context.get("comparison_intent", False))

    if constraint is not None and constraint.conversation_only:
        return RepresentationProposal(
            style="conversation_only", reason_code="explicit_conversation_only_constraint",
            confidence=1.0, should_apply=True,
        )
    if constraint is not None and constraint.prefer_style:
        return RepresentationProposal(
            style=constraint.prefer_style, reason_code="explicit_prefer_style_constraint",
            confidence=1.0, should_apply=True,
        )

    if object_count >= 2 and comparison_intent:
        natural_style, reason = "side_by_side", "two_or_more_objects_being_compared"
    elif object_count >= 2:
        natural_style, reason = "cards", "multiple_named_alternatives"
    elif object_count == 1:
        natural_style, reason = "cards", "single_object_context"
    else:
        current = work_session.representation
        return RepresentationProposal(
            style=current.style if current is not None else "conversation_only",
            reason_code="no_semantic_structure", confidence=0.2, should_apply=False,
        )

    if constraint is not None and constraint.avoid_style and constraint.avoid_style == natural_style:
        # Generic fallback -- avoid re-proposing the exact style the user
        # just rejected, without guessing at a domain-specific alternative:
        # cards <-> list are the two flat, no-comparison-implied styles, so
        # each is the other's natural fallback; anything else (side_by_side
        # avoided) falls back to the plain cards default.
        fallback = "list" if natural_style == "cards" else "cards"
        return RepresentationProposal(
            style=fallback, reason_code=f"avoided_{natural_style}_fallback", confidence=0.6, should_apply=True,
        )

    return RepresentationProposal(style=natural_style, reason_code=reason, confidence=0.8, should_apply=True)
