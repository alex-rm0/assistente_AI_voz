"""Phase 8.0/8.0.1: Presence Decision Foundation -- SHADOW MODE ONLY.

`PresenceDecision` is a small, per-turn diagnostic describing how Echo
*would* posture a reply (continuity/initiative/tone/address), computed from
already-available, bounded signals. It is observational only: nothing in
this module is consumed by routing, response composition, memory writes,
or TTS -- see assistant/conversation.py's _finalize_response, which calls
compute_presence_decision() strictly AFTER final_response is already
settled, purely to log it (Phase 8.0 section 4).

NOT the same concept as assistant/presence_manager.py's PresenceManager
(ACTIVE_CONVERSATION/PASSIVE_MONITORING/... -- whether Echo is allowed to
respond/observe at all). PresenceDecision never touches that state machine
and is never consulted by it; the shared word "presence" here means
"conversational demeanour for this turn", not "availability".

Design principles (Phase 8.0 section 1/17):
- Presence must reduce friction, never force warmth/questions/personal
  references/verbosity. The common case is the smallest, plainest posture
  (answer_only / none / neutral) -- defaults lean that way throughout.
- Cheap, deterministic/heuristic signals only. No LLM call, no network
  call, no paid inference, no RNG (Phase 8.0.1 section 9 explicitly rules
  out randomness even for "occasional" behaviour -- sparsity comes from
  bounded recency/eligibility gates instead).
- Every field is a small, closed enum/bool/tuple -- never free text, never
  raw chain-of-thought.

Phase 8.0.1 calibration (see the module-level audit note above
_missing_compose_email_fields/_missing_generic_activity_fields usage sites
in assistant/conversation.py for the required-vs-optional finding): this
pass corrected an acknowledgement/continuation precedence bug, added an
explicit "user issued a continue directive" signal, tightened address
eligibility to a two-tier (strong-moment vs weak-moment) recency gate so
"Sir" cannot become a periodic tic, and added a personal-relevance
eligibility gate so a self-contained factual/arithmetic question never
surfaces personal context even when a memory candidate scores high.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from assistant.text_matching import contains_any_phrase, normalize_for_matching

# --- Canonical enums (section 3) -- deliberately small, closed sets --------

CONTINUITY_VALUES = ("none", "light", "explicit")
INITIATIVE_VALUES = ("answer_only", "acknowledge", "suggest", "ask", "reconnect")
TONE_VALUES = ("neutral", "warm", "focused")
ADDRESS_MODE_VALUES = ("none", "name", "honorific")


@dataclass(frozen=True)
class PersonalCandidate:
    """One bounded, discrete personal-context candidate -- an id/score pair
    from LongTermMemory.search(), never a raw content dump in logs (section
    6's "do not expose sensitive/raw memory content in logs if avoidable")."""

    id: str
    category: str
    score: float


@dataclass(frozen=True)
class PresenceDecision:
    continuity: str = "none"
    initiative: str = "answer_only"
    question_needed: bool = False
    personal_candidates: tuple[PersonalCandidate, ...] = ()
    personal_selected: tuple[PersonalCandidate, ...] = ()
    address_mode: str = "none"
    tone: str = "neutral"
    confidence: float = 0.5
    reason_codes: tuple[str, ...] = ()

    @property
    def personal_candidates_count(self) -> int:
        return len(self.personal_candidates)

    @property
    def personal_selected_count(self) -> int:
        return len(self.personal_selected)


@dataclass(frozen=True)
class PreferredAddress:
    """Section 11: a parameterized address PREFERENCE -- not used to
    actually address the user anywhere yet (section 25). Deliberately does
    NOT contain any per-language translation table for honorifics: a
    language that has no explicit entry in honorific_by_language simply has
    no honorific candidate at all (never a generic fallback), which is what
    structurally prevents "Sir" ever auto-resolving to "Senhor" (section
    13's explicit prohibition) -- there is no code path that translates an
    honorific string, only a per-language lookup that is either present or
    absent."""

    default_mode: str = "none"  # "name" | "none" -- baseline when no special signal fires
    name: str = ""
    honorific_by_language: dict[str, str] = field(default_factory=dict)  # e.g. {"en-GB": "Sir"}
    frequency: str = "sparse"  # documented intent only; enforced via recency tracking, not a numeric rate here


# --- Reason codes (section 16) ---------------------------------------------

REASON_ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
REASON_NATURAL_CLOSURE = "NATURAL_CLOSURE"
REASON_MISSING_REQUIRED_INFO = "MISSING_REQUIRED_INFO"
REASON_TASK_CONTINUATION = "TASK_CONTINUATION"
REASON_RELEVANT_PERSONAL_CONTEXT = "RELEVANT_PERSONAL_CONTEXT"
REASON_PRIOR_TOPIC_RESUMPTION = "PRIOR_TOPIC_RESUMPTION"
REASON_NO_USEFUL_FOLLOWUP = "NO_USEFUL_FOLLOWUP"
REASON_DIRECT_ANSWER_SUFFICIENT = "DIRECT_ANSWER_SUFFICIENT"
REASON_ADDRESS_RECENTLY_USED = "ADDRESS_RECENTLY_USED"
# Phase 8.0.1 additions (section 16):
REASON_USER_CONTINUE_DIRECTIVE = "USER_CONTINUE_DIRECTIVE"
REASON_TASK_CAN_PROGRESS = "TASK_CAN_PROGRESS"
REASON_ADDRESS_MOMENT_ELIGIBLE = "ADDRESS_MOMENT_ELIGIBLE"
REASON_ADDRESS_CONTEXT_UNSUITABLE = "ADDRESS_CONTEXT_UNSUITABLE"
REASON_PERSONAL_CONTEXT_NOT_MATERIAL = "PERSONAL_CONTEXT_NOT_MATERIAL"
REASON_PERSONAL_CONTEXT_MATERIAL = "PERSONAL_CONTEXT_MATERIAL"
# Phase 8.0.2 addition (section 10/16):
REASON_SELF_REPRESENTATION_MOMENT = "SELF_REPRESENTATION_MOMENT"

# --- Closure / acknowledgement detection (section 10) ----------------------
# Bounded phrase lists, matched via contains_any_phrase (word-boundary safe,
# see assistant/text_matching.py's own "ferro contains erro" warning) --
# not rigid exact-phrase-only matching: each entry matches as a phrase
# anywhere in the (normalized) message, so "sim, agora já percebi, obrigado"
# still matches "percebi" even with surrounding words.
_ACKNOWLEDGEMENT_PHRASES_PT = (
    "ja percebi",
    "percebi",
    "ok",
    "certo",
    "obrigado",
    "obrigada",
    "boa",
    "perfeito",
    "era isso",
    "esta bem",
)
_ACKNOWLEDGEMENT_PHRASES_EN = (
    "got it",
    "okay",
    "makes sense",
    "thanks",
    "thank you",
    "perfect",
    "thats it",
    "understood",
)
_ACKNOWLEDGEMENT_PHRASES = _ACKNOWLEDGEMENT_PHRASES_PT + _ACKNOWLEDGEMENT_PHRASES_EN

# Phase 8.0.1 section 5/6: an explicit "keep going" directive. When one of
# these co-occurs with an acknowledgement filler word ("ok, continua"), the
# directive must win -- an acknowledgement prefix must not swallow the
# actual instruction that follows it (section 6).
_CONTINUE_DIRECTIVE_PHRASES_PT = (
    "continua a escrever",
    "continua isso",
    "continua",
    "segue",
    "avanca",
    "podes continuar",
)
_CONTINUE_DIRECTIVE_PHRASES_EN = (
    "continue writing",
    "keep going",
    "carry on",
    "continue",
    "go ahead",
)
_CONTINUE_DIRECTIVE_PHRASES = _CONTINUE_DIRECTIVE_PHRASES_PT + _CONTINUE_DIRECTIVE_PHRASES_EN

# Phase 8.0.1 section 6: "ok, mas antes preciso de..." -- the user is adding
# more content/a caveat, not closing. Also disqualifies a bare
# acknowledgement filler from being treated as natural closure.
_TRAILING_CONTENT_MARKERS_PT = ("mas antes", "so que", "preciso de", "quero tambem", "quero também")
_TRAILING_CONTENT_MARKERS_EN = ("but first", "before that", "i also need", "i still need")
_TRAILING_CONTENT_MARKERS = _TRAILING_CONTENT_MARKERS_PT + _TRAILING_CONTENT_MARKERS_EN

# Phase 8.0.1 section 8: a direct Echo-address moment ("Echo, ...") is one
# of the few "strong" address-eligible moments -- bounded, not a general
# vocative parser.
_DIRECT_ADDRESS_MARKERS = ("echo,", "hey echo", "ok echo", "echo pergunto-te", "echo can i ask")

# Phase 8.0.2 section 10: a bounded, deterministic capability/self-
# representation detector -- "who are you"/"what can you do" style
# questions. Deliberately a fixed phrase list, not a capability subsystem
# (section 10 explicitly forbids building one); only ever contributes a
# reason code and (section 11) a strong address-eligibility signal, never
# changes the actual answer.
_SELF_REPRESENTATION_MARKERS_PT = (
    "o que consegues fazer",
    "como e que funcionas",
    "como funcionas",
    "quem es",
    "o que sabes fazer",
)
_SELF_REPRESENTATION_MARKERS_EN = (
    "what can you do",
    "how do you work",
    "who are you",
    "what are you capable of",
)
_SELF_REPRESENTATION_MARKERS = _SELF_REPRESENTATION_MARKERS_PT + _SELF_REPRESENTATION_MARKERS_EN

# Section 9: a message that is itself a direct, self-contained factual
# question -- Echo can just answer it, no follow-up question is useful, and
# (Phase 8.0.1 section 13) personal context is never material to it either.
_QUESTION_WORDS_PT = ("qual", "quem", "quando", "onde", "como", "quanto", "quantos", "quantas", "porque", "porquê")
_QUESTION_WORDS_EN = ("what", "who", "when", "where", "how", "why", "which")

# Phase 8.0.1 section 9: two-tier address recency gate, deterministic (no
# RNG). A "strong" moment (restart / direct address / important task
# completion) only needs a modest gap since the last address. A "weak"
# moment (a bare, otherwise-ordinary acknowledgement) needs a much larger
# gap -- this is what keeps "occasional short English acknowledgement"
# (section 8) genuinely occasional instead of a predictable every-N-turns
# tic, without introducing any randomness.
_ADDRESS_STRONG_MOMENT_MIN_GAP = 3
_ADDRESS_WEAK_MOMENT_MIN_GAP = 10


def _looks_like_direct_question(normalized_message: str) -> bool:
    if normalized_message.endswith("?"):
        return True
    first_word = normalized_message.split(" ", 1)[0] if normalized_message else ""
    return first_word in _QUESTION_WORDS_PT or first_word in _QUESTION_WORDS_EN


def _detect_acknowledgement(normalized_message: str) -> bool:
    return contains_any_phrase(normalized_message, _ACKNOWLEDGEMENT_PHRASES)


def _detect_continue_directive(normalized_message: str) -> bool:
    return contains_any_phrase(normalized_message, _CONTINUE_DIRECTIVE_PHRASES)


def _detect_trailing_content_marker(normalized_message: str) -> bool:
    return contains_any_phrase(normalized_message, _TRAILING_CONTENT_MARKERS)


def _detect_direct_address(normalized_message: str) -> bool:
    return contains_any_phrase(normalized_message, _DIRECT_ADDRESS_MARKERS)


def _detect_self_representation(normalized_message: str) -> bool:
    return contains_any_phrase(normalized_message, _SELF_REPRESENTATION_MARKERS)


# Section 7: phrases that explicitly ask to resume/continue earlier state --
# only ever escalate continuity to "explicit" when CONCRETE prior state is
# also available (see _detect_continuity); on their own they are not
# sufficient evidence.
_RESUME_PHRASES = (
    "lembras te do que estavamos a fazer",
    "podemos continuar aquilo",
    "continuar aquilo que estavamos a fazer",
    "onde ficamos",
    "onde e que ficamos",
    "voltar aquilo",
    "continue where we left off",
    "what were we doing",
    "back to what we were doing",
)


def _detect_resume_request(normalized_message: str) -> bool:
    return contains_any_phrase(normalized_message, _RESUME_PHRASES)


def _detect_continuity(
    *, normalized_message: str, active_task_present: bool, active_task_unresolved: bool, is_restart: bool
) -> tuple[str, list[str]]:
    """Section 7: continuity=explicit requires STRONG, concrete evidence --
    an actual active task/topic to point to, not just a vague "some memory
    probably relates to this" feeling. A resume request with nothing
    concrete behind it stays none/light, never explicit."""
    reason_codes: list[str] = []
    asks_to_resume = _detect_resume_request(normalized_message)
    if active_task_present:
        if asks_to_resume or is_restart:
            reason_codes.append(REASON_PRIOR_TOPIC_RESUMPTION)
            return "explicit", reason_codes
        reason_codes.append(REASON_TASK_CONTINUATION)
        return "light", reason_codes
    # No concrete active task to point to -- even an explicit resume request
    # cannot be honoured as "explicit" continuity (section 7's example).
    return "none", reason_codes


def _decide_initiative_and_question(
    *,
    normalized_message: str,
    is_acknowledgement: bool,
    is_continue_directive: bool,
    active_task_present: bool,
    active_task_unresolved: bool,
    continuity: str,
) -> tuple[str, bool, list[str]]:
    """Section 8/9 (Phase 8.0) + sections 2/3/5 (Phase 8.0.1): never
    defaults to "ask". question_needed=True requires a CONCRETE reason --
    in practice, only a genuinely unresolved active task (missing_information
    non-empty; audited in assistant/conversation.py's
    _compute_presence_decision_shadow docstring to already be a
    required-only list across every task type, so no separate
    required-vs-optional split is needed here). A continue directive on an
    unblocked task explicitly prefers answer_only/focused over inventing a
    follow-up question (section 5)."""
    reason_codes: list[str] = []
    if is_acknowledgement:
        reason_codes.append(REASON_ACKNOWLEDGEMENT)
        reason_codes.append(REASON_NATURAL_CLOSURE)
        reason_codes.append(REASON_NO_USEFUL_FOLLOWUP)
        return "acknowledge", False, reason_codes
    if active_task_present and active_task_unresolved:
        reason_codes.append(REASON_MISSING_REQUIRED_INFO)
        reason_codes.append(REASON_TASK_CONTINUATION)
        return "ask", True, reason_codes
    if active_task_present and is_continue_directive:
        reason_codes.append(REASON_USER_CONTINUE_DIRECTIVE)
        reason_codes.append(REASON_TASK_CAN_PROGRESS)
        return "answer_only", False, reason_codes
    if continuity == "explicit":
        reason_codes.append(REASON_PRIOR_TOPIC_RESUMPTION)
        return "reconnect", False, reason_codes
    if _looks_like_direct_question(normalized_message):
        reason_codes.append(REASON_DIRECT_ANSWER_SUFFICIENT)
        reason_codes.append(REASON_NO_USEFUL_FOLLOWUP)
        return "answer_only", False, reason_codes
    reason_codes.append(REASON_NO_USEFUL_FOLLOWUP)
    return "answer_only", False, reason_codes


def _decide_tone(*, is_acknowledgement: bool, continuity: str, active_task_present: bool) -> str:
    """Section 15 (Phase 8.0) + section 3 (Phase 8.0.1): still just three
    broad postures, not sentiment analysis. An in-progress task now reads as
    focused even during ordinary continuation (previously this fell through
    to "warm" whenever continuity was non-none, which mislabelled plain
    task continuation as an emotionally warm moment instead of a focused
    one); a pure acknowledgement stays warm."""
    if is_acknowledgement:
        return "warm"
    if active_task_present:
        return "focused"
    if continuity != "none":
        return "warm"
    return "neutral"


def select_personal_relevance(
    candidates: list[PersonalCandidate], *, relevance_threshold: float = 0.6, max_selected: int = 2
) -> tuple[PersonalCandidate, ...]:
    """Section 6: 0-2 candidates, and 0 for most turns. `candidates` is
    expected to already be the bounded, ranked output of LongTermMemory.
    search(user_message, limit=2) mapped to PersonalCandidate -- this
    function only decides which of those (if any) clear the relevance bar,
    it never fetches memory itself."""
    relevant = [candidate for candidate in candidates if candidate.score >= relevance_threshold]
    return tuple(relevant[:max_selected])


def _personal_relevance_eligible(*, is_acknowledgement: bool, normalized_message: str) -> bool:
    """Phase 8.0.1 section 13: a high semantic-search score does not by
    itself mean a candidate should surface. A self-contained factual/
    arithmetic question ("Quanto é 8x7?") or a bare acknowledgement has no
    turn-level use for personal context regardless of what memory returns
    -- gate eligibility on the TURN, before ever looking at candidate
    scores."""
    if is_acknowledgement:
        return False
    if _looks_like_direct_question(normalized_message):
        return False
    return True


def resolve_address_candidate(
    preferred: PreferredAddress, response_language: str
) -> tuple[str, str]:
    """Section 13/14: returns (address_mode, resolved_value). The honorific
    lookup is an EXACT per-language dict lookup with no fallback logic at
    all -- if response_language has no entry in honorific_by_language, no
    honorific is ever produced for it, which is what makes "Sir" -> "Senhor"
    structurally impossible here (there is no translation step to skip;
    there is simply no entry). Section 10 (Phase 8.0.1): Portuguese must
    not artificially fall back to name just to compensate for the missing
    honorific -- that fallback already only happens when preferred.name is
    itself non-empty (a real configured preference), never as compensation."""
    honorific = preferred.honorific_by_language.get(response_language, "")
    if honorific:
        return "honorific", honorific
    if preferred.name:
        return "name", preferred.name
    return "none", ""


def _decide_address(
    *,
    preferred: PreferredAddress,
    response_language: str,
    is_restart: bool,
    is_acknowledgement: bool,
    is_direct_address: bool,
    is_self_representation: bool,
    task_recently_completed: bool,
    active_task_dense: bool,
    turns_since_address: int,
) -> tuple[str, list[str]]:
    """Phase 8.0.1 sections 7-9/12/14: sparse by construction, and now
    genuinely non-periodic rather than "eligible again after exactly N
    turns" for every kind of moment. A dense task always suppresses
    addressing outright. Otherwise two tiers apply:
      - "strong" moments (restart / direct Echo-address / just completed an
        important action) are eligible after a modest recency gap;
      - a bare acknowledgement ("occasional short English acknowledgement",
        section 8) is eligible only after a much larger gap, which is what
        keeps it rare instead of a predictable tic (section 9) without any
        randomness.
    Ordinary factual Q&A, rapid back-and-forth, and task clarification
    loops never reach a "moment" branch at all (none of is_restart/
    is_direct_address/task_recently_completed/is_acknowledgement holds), so
    they fall straight through to "none" -- matching the "poor candidates"
    list in section 8."""
    reason_codes: list[str] = []
    if active_task_dense:
        reason_codes.append(REASON_ADDRESS_CONTEXT_UNSUITABLE)
        return "none", reason_codes

    # Section 11 (Phase 8.0.2): a self-representation moment ("who are
    # you?") is a strong candidate -- for en-GB this may resolve to the
    # configured honorific if recency permits; for pt-PT (no honorific
    # entry) resolve_address_candidate already falls back to name/none, so
    # no separate per-language branch is needed here.
    strong_moment = is_restart or is_direct_address or task_recently_completed or is_self_representation
    weak_moment = is_acknowledgement

    if strong_moment and turns_since_address >= _ADDRESS_STRONG_MOMENT_MIN_GAP:
        mode, _value = resolve_address_candidate(preferred, response_language)
        if mode != "none":
            reason_codes.append(REASON_ADDRESS_MOMENT_ELIGIBLE)
            return mode, reason_codes
    if weak_moment and turns_since_address >= _ADDRESS_WEAK_MOMENT_MIN_GAP:
        mode, _value = resolve_address_candidate(preferred, response_language)
        if mode != "none":
            reason_codes.append(REASON_ADDRESS_MOMENT_ELIGIBLE)
            return mode, reason_codes

    if (strong_moment or weak_moment) and turns_since_address < _ADDRESS_WEAK_MOMENT_MIN_GAP:
        reason_codes.append(REASON_ADDRESS_RECENTLY_USED)
    else:
        reason_codes.append(REASON_ADDRESS_CONTEXT_UNSUITABLE)
    return "none", reason_codes


def compute_presence_decision(
    *,
    user_message: str,
    response_language: str,
    active_task_present: bool,
    active_task_unresolved: bool,
    active_task_dense: bool,
    is_restart: bool,
    personal_candidates: list[PersonalCandidate] | None = None,
    preferred_address: PreferredAddress | None = None,
    turns_since_address: int = 0,
    task_recently_completed: bool = False,
) -> PresenceDecision:
    """The single, pure entry point -- no engine/IO access, no LLM call
    (section 17). Callers (assistant/conversation.py) are responsible for
    gathering the bounded inputs (task state, restart flag, up to a couple
    of ranked memory candidates, turns since Presence last chose to
    address) and for treating the result as observational only (section
    4)."""
    normalized_message = normalize_for_matching(user_message)
    raw_is_acknowledgement = _detect_acknowledgement(normalized_message)
    is_continue_directive = _detect_continue_directive(normalized_message)
    has_trailing_content = _detect_trailing_content_marker(normalized_message)
    # Phase 8.0.1 section 6: a continue-directive or a "but I still need
    # X" trailing-content marker overrides a merely-co-occurring
    # acknowledgement filler word ("ok, continua" / "ok, mas antes
    # preciso de..." are NOT natural closure).
    is_acknowledgement = raw_is_acknowledgement and not is_continue_directive and not has_trailing_content
    is_direct_address = _detect_direct_address(normalized_message)
    is_self_representation = _detect_self_representation(normalized_message)

    continuity, continuity_reasons = _detect_continuity(
        normalized_message=normalized_message,
        active_task_present=active_task_present,
        active_task_unresolved=active_task_unresolved,
        is_restart=is_restart,
    )
    initiative, question_needed, initiative_reasons = _decide_initiative_and_question(
        normalized_message=normalized_message,
        is_acknowledgement=is_acknowledgement,
        is_continue_directive=is_continue_directive,
        active_task_present=active_task_present,
        active_task_unresolved=active_task_unresolved,
        continuity=continuity,
    )
    tone = _decide_tone(is_acknowledgement=is_acknowledgement, continuity=continuity, active_task_present=active_task_present)

    candidates = tuple(personal_candidates or ())
    personal_reasons: list[str] = []
    if _personal_relevance_eligible(is_acknowledgement=is_acknowledgement, normalized_message=normalized_message):
        selected = select_personal_relevance(list(candidates))
    else:
        selected = ()
    if candidates:
        personal_reasons.append(REASON_PERSONAL_CONTEXT_MATERIAL if selected else REASON_PERSONAL_CONTEXT_NOT_MATERIAL)

    preferred = preferred_address or PreferredAddress()
    address_mode, address_reasons = _decide_address(
        preferred=preferred,
        response_language=response_language,
        is_restart=is_restart,
        is_acknowledgement=is_acknowledgement,
        is_direct_address=is_direct_address,
        is_self_representation=is_self_representation,
        task_recently_completed=task_recently_completed,
        active_task_dense=active_task_dense,
        turns_since_address=turns_since_address,
    )
    self_representation_reasons = [REASON_SELF_REPRESENTATION_MOMENT] if is_self_representation else []

    reason_codes: list[str] = []
    for code in (*continuity_reasons, *initiative_reasons, *personal_reasons, *address_reasons, *self_representation_reasons):
        if code not in reason_codes:
            reason_codes.append(code)

    # Confidence: a small, honest signal -- higher when a strong, explicit
    # marker fired (exact acknowledgement phrase, concrete task state),
    # lower for the plain "nothing special detected" default. Never
    # presented as a precise probability -- just a coarse ranking.
    if is_acknowledgement or continuity == "explicit":
        confidence = 0.9
    elif active_task_present:
        confidence = 0.75
    else:
        confidence = 0.5

    return PresenceDecision(
        continuity=continuity,
        initiative=initiative,
        question_needed=question_needed,
        personal_candidates=candidates,
        personal_selected=selected,
        address_mode=address_mode,
        tone=tone,
        confidence=confidence,
        reason_codes=tuple(reason_codes),
    )
