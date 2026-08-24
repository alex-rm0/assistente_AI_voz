"""Phase 2 of the Cognitive Workspace Blueprint v2.

This module classifies one user utterance into MULTIPLE simultaneous
semantic intents (activity / content / spatial / representation / reasoning
/ conversational). It is deliberately read-only and side-effect-free:

- it never mutates WorkSession (or the ActiveTaskContext/
  ActiveDocumentContext it holds);
- it never calls an LLM or a provider;
- its output is consumed by the conversation layer as the canonical
  routing decision for interpreter-owned axes, while execution remains in
  AssistantEngine.

Design choice: rather than a new set of isolated `_try_*` functions (the
exact pattern the blueprint calls out as the root cause of the current
system's fragility), every axis below is a pure function of the normalized
utterance (plus, for activity/content, a read-only peek at WorkSession),
and `interpret()` runs ALL of them unconditionally and accumulates results
-- never short-circuiting after the first match. This is what makes
multi-intent utterances ("não faças mapa, deixa só as duas opções lado a
lado e diz-me qual escolherias") representable as one interpretation with
three axes present at once, instead of three sequential turns.

Reuse, not reinvention: activity and spatial classification call shared
pure primitives in `assistant.interaction_primitives`. The interpreter has
no dependency on the execution layer; AssistantEngine imports this module
and acts as the executor for the interpreted axes.

Representation and reasoning are genuinely new axes (nothing to reuse) and
are implemented here as small, self-contained phrase-list recognizers.

Deterministic only in Phase 2 -- no optional LLM fallback is implemented.
The blueprint permits one but does not require it, and Phase 2's own
guidance is to "prefer keeping Phase 2 deterministic unless actual
ambiguity requires more"; nothing so far does.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from assistant.interaction_primitives import (
    detect_task_control_action,
    extract_role_course_name,
    interpret_activity,
    purpose_marker_keys,
    topic_about_re,
    workspace_command_actions,
)
from assistant.representation import extract_added_alternatives, extract_explicit_alternatives
from assistant.text_matching import contains_any_phrase, normalize_for_matching

# --- Intent dataclasses -----------------------------------------------

_KNOWN_ROUTES_COMMENT = """
Route labels used by expected_route()/compare_route() below intentionally
reuse the SAME strings as production `selected_path` telemetry
(WORKSPACE_COMMAND, TASK_CONTINUATION, GENERAL_CONVERSATION) so a direct
string comparison is meaningful. TASK_EXECUTION and REPRESENTATION_COMMAND
have no production equivalent today -- see compare_route().
"""


@dataclass(frozen=True)
class ActivityIntent:
    present: bool = False
    starts_task: bool = False
    continues_task: bool = False
    refines_task: bool = False
    pauses_task: bool = False
    resumes_task: bool = False
    switches_task: bool = False
    activity_family: str = "unknown"
    domain: str = "unknown"
    artifact_type: str = "unknown"
    task_type: str = ""
    goal: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class ContentIntent:
    """Section 3: deliberately structural for Phase 2 -- classifies the
    KIND of content change (entity/topic/constraint/purpose/control) when
    confidence is sufficient, without migrating every domain-specific
    extractor (e.g. _extract_compose_email_fields) into this module."""

    present: bool = False
    kind: str = ""  # entity | topic | constraint | purpose | control_execute | unspecified
    confidence: float = 0.0
    evidence: str = ""


@dataclass(frozen=True)
class SpatialIntent:
    present: bool = False
    action: str = ""
    target: str = ""
    scope: str = ""
    region: str = ""
    confidence: float = 0.0
    # Section 14.M: "mantém estes dois documentos lado a lado" carries a
    # constraint-like signal even though real WorkspaceConstraint behavior
    # (typed, persisted, strength-ranked) belongs to a later phase -- this
    # is classification only, a hint for that future phase, not a stored
    # constraint.
    constraint_like: bool = False


@dataclass(frozen=True)
class RepresentationIntent:
    present: bool = False
    action: str = ""  # requested_style | rejected_style | simplify | increase_visual_structure
    #                   | hide_representation | restore_representation | filter
    #                   | grouping_requested | ordering | parameter_mutation
    #                   | show_reasoning | hide_reasoning
    style: str = ""
    confidence: float = 0.0
    evidence: str = ""
    # Phase 3C final patch (section 4/5): a grouping/organization request
    # ("separa os cards dos destinos internacionais dos destinos
    # portugueses", "agrupa por prioridade") is semantically distinct from
    # a style request ("mostra em cartões"), even when the utterance
    # happens to mention "cards"/"cartões" as a plain noun rather than a
    # style change ask -- the real Qt bug this fixes: such an utterance
    # was being read as requested_style="cards" purely because the bare
    # word "cards" appears in it. Phase 3C still does not implement
    # semantic grouping execution; these fields only let the caller
    # recognize the REQUEST and respond truthfully instead of silently
    # reapplying style=cards as a false success.
    grouping_requested: bool = False
    grouping_instruction: str = ""
    grouping_criterion: str = ""


@dataclass(frozen=True)
class ReasoningIntent:
    present: bool = False
    action: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class AttentionIntent:
    """Phase 4 (section 5/20): explicit attention/visibility commands --
    genuinely distinct from SpatialIntent (which is about WHERE something
    sits, e.g. echo's own placement or a side_by_side arrangement) and from
    RepresentationIntent (which is about HOW the whole set is shown).
    `target_phrase` is the raw utterance, left for the caller to resolve
    against the live object registry (assistant.representation.
    resolve_mentioned_object_ids) -- this module never touches
    work_session.objects itself, mirroring every other axis here."""

    present: bool = False
    action: str = ""  # keep_visible | do_not_minimize | hide_object | minimize_object | focus_object | show_again
    target_phrase: str = ""
    confidence: float = 0.0
    evidence: str = ""


@dataclass(frozen=True)
class ConversationalIntent:
    present: bool = False
    confidence: float = 0.0


@dataclass(frozen=True)
class InteractionInterpretation:
    activity: ActivityIntent
    content: ContentIntent
    spatial: SpatialIntent
    representation: RepresentationIntent
    reasoning: ReasoningIntent
    attention: AttentionIntent
    conversational: ConversationalIntent
    confidence: float = 0.0
    ambiguities: tuple[str, ...] = ()

    @property
    def multi_intent_count(self) -> int:
        return sum(
            1
            for axis in (
                self.activity, self.content, self.spatial, self.representation,
                self.reasoning, self.attention, self.conversational,
            )
            if axis.present
        )


@dataclass(frozen=True)
class TurnRoutingDecision:
    primary_route: str
    secondary_intents: tuple[str, ...] = ()
    confidence: float = 0.0
    unsupported_intents: tuple[str, ...] = ()
    ambiguity: bool = False
    reason_code: str = ""
    authority: str = "InteractionInterpreter"


# --- Representation vocabulary -----------------------------------------
# All markers below are matched via contains_any_phrase against a
# normalize_for_matching()'d utterance -- accents/case are handled there,
# these lists are written in plain readable Portuguese.

_REPRESENTATION_SHOW_REASONING_PHRASES = ("mostra o raciocinio", "mostra-me porque escolheste isto", "mostra-me porque")
_REPRESENTATION_HIDE_REASONING_PHRASES = (
    "esconde o raciocinio", "nao preciso de ver o raciocinio", "nao quero ver o raciocinio",
    "nao preciso de ver o teu raciocinio",
)
_REPRESENTATION_REASONING_VISIBILITY_PHRASES = _REPRESENTATION_SHOW_REASONING_PHRASES + _REPRESENTATION_HIDE_REASONING_PHRASES
_REPRESENTATION_HIDE_PHRASES = (
    "nao quero ver nada", "fala so comigo", "esconde isso", "esconde tudo",
    # Phase 3B integration fix: "vamos so conversar" on its own carries the
    # same intent as "nao quero ver nada" (hide the workspace, keep
    # talking) -- added after a real Qt run showed it falling through to
    # DOCUMENT_TASK when it didn't co-occur with the longer phrase.
    "vamos so conversar",
)
_REPRESENTATION_RESTORE_PHRASES = (
    "mostra outra vez", "mostra me outra vez", "mostra-me outra vez",
    "traz de volta", "mostra de novo", "mostra me de novo", "poe outra vez",
)
_REPRESENTATION_SIMPLIFY_PHRASES = (
    "nao compliques", "faz algo mais simples", "faz mais simples", "simplifica",
)
_REPRESENTATION_PARAM_PHRASES = (
    "troca os eixos", "troca o eixo", "usa barras", "muda o eixo",
)
_REPRESENTATION_FILTER_PHRASES = (
    "mostra so", "quero ver so",
)
_REPRESENTATION_GROUP_PHRASES = (
    "agrupa", "agrupar", "agrupar as ideias",
    # Phase 3C final patch: "separa"/"distingue" cover "separa os cards dos
    # destinos internacionais dos destinos portugueses" -- a grouping/
    # organization ask, not a style change, even though it names "cards".
    # Deliberately does NOT include "organiza por" -- that stays with the
    # existing, narrower _REPRESENTATION_ORDER_PHRASES family below so
    # "organiza por prioridade" keeps its established "ordering" reading.
    "separa", "separar", "distingue", "distinguir",
    "junta as opcoes parecidas", "junta as parecidas",
)
_GROUPING_CRITERION_RE = re.compile(r"\bpor\s+([a-zà-ÿ]+(?:\s+[a-zà-ÿ]+){0,2})\b")
_REPRESENTATION_ORDER_PHRASES = (
    "organiza por prioridade", "poe por prioridade", "ordena por prioridade",
)
_REPRESENTATION_VISUAL_PHRASES = (
    "faz mais visual", "mais visual",
)
# Bare style words -- a match here only decides `style`; whether the turn
# is a request or a rejection is decided separately (see _representation_intent).
_REPRESENTATION_STYLE_WORDS = {
    "mapa": "mind_map",
    "tabela": "table",
    "lista": "list",
    "matriz": "decision_matrix",
    "grafico": "chart",
    "barras": "chart",
    "clusters": "clusters",
    "caixas": "cards",
    "cards": "cards",
    # Phase 3 (Cognitive Workspace Blueprint v2, section 11): "cartões" is
    # the phrasing the Phase 3 voice bridge's acceptance flow actually uses
    # ("mostra em cartões") -- added as another synonym for the existing
    # "cards" style, not a new style.
    "cartoes": "cards",
}
_NEGATION_RE = re.compile(r"\bnao\b")

_REASONING_PHRASES = (
    "qual escolherias", "que escolherias", "porque estas a preferir", "porque estas a favorecer",
    "mostra-me os trade-offs", "mostra os trade-offs", "quais sao os riscos", "o que pesa mais",
    "qual recomendas", "recomendas qual", "qual e a tua recomendacao",
)

_CONVERSATIONAL_MARKERS = (
    "ola", "boa tarde", "bom dia", "boa noite", "tudo bem", "como estas", "obrigado", "obrigada",
)

# --- Attention vocabulary (Phase 4, section 5) --------------------------
# Same contains_any_phrase(normalize_for_matching(...)) convention as every
# other axis. Distinct wording from the representation/spatial families
# above so a real utterance never double-fires two axes for the same
# clause -- e.g. "não escondas" (this ONE object) vs. "esconde tudo"
# (the whole workspace, RepresentationIntent's hide_representation).

_ATTENTION_KEEP_VISIBLE_PHRASES = (
    "nao escondas", "deixa aberto", "deixa isto aberto", "deixa isso aberto",
    "mantem isto aqui", "mantem isto", "mantem aqui", "nao minimizes", "nao minimize",
    "nao feches",
)
# Phase 4 manual-test fix (section 1): the literal phrases above were built
# for the DEICTIC, object-less case ("mantém isto", "mantém aqui") -- a
# named object between the verb and the state word ("mantém Açores
# visível", "deixa o Documento A aberto") never matches any of them
# verbatim, since the object name varies. This generic verb+state
# co-occurrence check catches ANY named object in that shape without a
# phrase per object, while staying narrow enough (BOTH a keep-open verb
# AND an explicit visible/open state word) not to collide with Echo's own
# spatial "mantém-te aqui" placement commands, which never mention
# visível/aberto.
_ATTENTION_KEEP_VISIBLE_VERBS = ("mantem", "deixa")
_ATTENTION_KEEP_VISIBLE_STATE_WORDS = ("visivel", "aberto", "aberta")


def _is_keep_visible_phrase(normalized: str) -> bool:
    if contains_any_phrase(normalized, _ATTENTION_KEEP_VISIBLE_PHRASES):
        return True
    return contains_any_phrase(normalized, _ATTENTION_KEEP_VISIBLE_VERBS) and contains_any_phrase(
        normalized, _ATTENTION_KEEP_VISIBLE_STATE_WORDS
    )
_ATTENTION_HIDE_OBJECT_PHRASES = (
    "nao preciso de ver isto", "isto nao preciso de ver", "isto nao preciso ver",
    "nao preciso de ver isso", "isso nao preciso de ver",
)
_ATTENTION_MINIMIZE_PHRASES = ("minimiza",)
_ATTENTION_FOCUS_PHRASES = (
    "para a frente", "foca so nisto", "foca nisto", "foca no", "foca na",
    "poe em foco", "foca-te nisto", "foca so",
)
_ATTENTION_SHOW_VERBS = ("mostra", "abre", "traz")
_ATTENTION_RECOVERY_MARKERS = (
    "outra vez", "de novo", "que minimizaste", "que escondeste", "que minimizou", "que tinhas escondido",
)


def _is_show_again_phrase(normalized: str) -> bool:
    """A bare "mostra outra vez"/"mostra de novo" (no named/deictic object)
    is the EXISTING whole-workspace restore_representation trigger family
    (assistant.representation's own Phase 3 "mostra outra vez" phrase) --
    this axis must only claim an utterance that ALSO names something
    beyond the bare show-verb + recovery-marker pair ("mostra o email
    outra vez", "abre o documento que minimizaste"), never the bare phrase
    itself, or it would silently steal every whole-workspace restore."""
    if not any(marker in normalized for marker in _ATTENTION_RECOVERY_MARKERS):
        return False
    if not any(verb in normalized for verb in _ATTENTION_SHOW_VERBS):
        return False
    residual = normalized
    for verb in _ATTENTION_SHOW_VERBS:
        residual = residual.replace(verb, " ")
    for marker in _ATTENTION_RECOVERY_MARKERS:
        residual = residual.replace(marker, " ")
    return len(residual.strip()) >= 3


def _is_visual_return_to_active_task(normalized: str, work_session) -> bool:
    if not contains_any_phrase(normalized, ("volta ao", "volta a", "volta para")):
        return False
    writing_context = getattr(work_session, "writing_context", None) if work_session is not None else None
    if writing_context is not None and getattr(writing_context, "active_artifact_id", ""):
        if contains_any_phrase(normalized, ("email", "mail", "mensagem")):
            return True
    active_task = work_session.active_task if work_session is not None else None
    paused_tasks = getattr(work_session, "paused_task_stack", []) if work_session is not None and active_task is not None else []
    for paused in paused_tasks:
        task_type = getattr(paused, "task_type", "")
        domain = getattr(paused, "domain", "")
        artifact_type = getattr(paused, "artifact_type", "")
        if task_type == "compose_email" or domain == "email" or artifact_type == "email":
            if contains_any_phrase(normalized, ("email", "mail", "mensagem")):
                return True
        if task_type == "plan_trip" or domain == "travel":
            if contains_any_phrase(normalized, ("ferias", "viagem", "planeamento")):
                return True
    if active_task is None:
        return False
    task_type = getattr(active_task, "task_type", "")
    domain = getattr(active_task, "domain", "")
    artifact_type = getattr(active_task, "artifact_type", "")
    if task_type == "compose_email" or domain == "email" or artifact_type == "email":
        return contains_any_phrase(normalized, ("email", "mail", "mensagem"))
    if task_type == "plan_trip" or domain == "travel":
        return contains_any_phrase(normalized, ("ferias", "viagem", "planeamento"))
    return False


def _normalize(text: str) -> str:
    return normalize_for_matching(text).strip(" .,!?;:")


def _activity_intent(raw_message: str, normalized: str, work_session) -> ActivityIntent:
    active_task = work_session.active_task if work_session is not None else None
    paused = work_session.paused_task_stack if work_session is not None else []
    interp = interpret_activity(raw_message, normalized, active_task, paused)
    present = bool(
        interp.starts_task or interp.continues_task or interp.refines_task
        or interp.pauses_task or interp.resumes_task or interp.confidence > 0.0
    )
    switches_task = bool(
        interp.starts_task and active_task is not None and interp.task_type and interp.task_type != active_task.task_type
    )
    return ActivityIntent(
        present=present,
        starts_task=interp.starts_task,
        continues_task=interp.continues_task,
        refines_task=interp.refines_task,
        pauses_task=interp.pauses_task,
        resumes_task=interp.resumes_task,
        switches_task=switches_task,
        activity_family=interp.activity_family,
        domain=interp.domain,
        artifact_type=interp.artifact_type,
        task_type=interp.task_type,
        goal=interp.goal,
        confidence=interp.confidence,
    )


def _content_intent(raw_message: str, normalized: str, work_session, activity: ActivityIntent) -> ContentIntent:
    active_task = work_session.active_task if work_session is not None else None

    if active_task is not None:
        control_action = detect_task_control_action(normalized, active_task)
        if control_action == "execute":
            return ContentIntent(present=True, kind="control_execute", confidence=0.9, evidence="control_execute")

    pattern_match = extract_role_course_name(raw_message)
    if pattern_match:
        return ContentIntent(present=True, kind="entity", confidence=0.75, evidence="entity")

    topic_match = topic_about_re().search(raw_message.strip(" .,!?;:"))
    if topic_match:
        return ContentIntent(present=True, kind="topic", confidence=0.7, evidence="topic")

    if re.search(r"\d+\s*(euros?|eur\b|€)", normalized):
        return ContentIntent(present=True, kind="constraint", confidence=0.7, evidence="constraint")

    if contains_any_phrase(normalized, purpose_marker_keys()):
        return ContentIntent(present=True, kind="purpose", confidence=0.65, evidence="purpose")

    if active_task is not None and (activity.continues_task or activity.refines_task) and normalized:
        return ContentIntent(present=True, kind="unspecified", confidence=0.4, evidence="generic_continuation")

    return ContentIntent()


def _spatial_intent(normalized: str) -> SpatialIntent:
    found, _remaining = workspace_command_actions(normalized)
    if not found:
        return SpatialIntent()
    primary = found[0]
    constraint_like = contains_any_phrase(normalized, ("mantem", "manter"))
    return SpatialIntent(
        present=True,
        action=str(primary.get("action", "")),
        target=str(primary.get("target_phrase", "")),
        scope=str(primary.get("scope", "")),
        region=str(primary.get("region", "")),
        confidence=0.85,
        constraint_like=constraint_like,
    )


def _extract_grouping_criterion(normalized: str) -> str:
    """Best-effort, deterministic-only extraction (section 5: 'if
    deterministically extractable') -- a trailing 'por X' clause
    ('agrupa por prioridade' -> 'prioridade', 'agrupa estas opções por
    país' -> 'pais'). Returns '' rather than guess when there is no such
    clause ('separa os cards dos destinos internacionais dos destinos
    portugueses' has no 'por X' -- Phase 3C does not attempt to classify
    the two named groups, per the phase's explicit scope)."""
    match = _GROUPING_CRITERION_RE.search(normalized)
    return match.group(1).strip() if match else ""


def _representation_style(normalized: str) -> str:
    for word, mapped in _REPRESENTATION_STYLE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            return mapped
    return ""


_CLAUSE_SPLIT_RE = re.compile(r",|;|\bmas\b")


def _style_word_is_negated(normalized: str, mapped_style: str) -> bool:
    """Phase 4: a single utterance can both REJECT one style and REQUEST
    another in the same breath ("não quero cartões, põe em lista") -- a
    plain whole-text "does 'não' appear ANYWHERE" check would read this as
    rejecting "list" (the style _representation_style happened to return)
    just because an unrelated "não" exists earlier in a different clause.
    This checks negation only within whichever CLAUSE (split on comma/
    "mas") actually contains one of `mapped_style`'s own words, falling
    back to the whole-text check for a single-clause utterance (identical
    to the original behavior when there is nothing to disambiguate)."""
    words = [word for word, mapped in _REPRESENTATION_STYLE_WORDS.items() if mapped == mapped_style]
    clauses = _CLAUSE_SPLIT_RE.split(normalized)
    if len(clauses) <= 1:
        return bool(_NEGATION_RE.search(normalized))
    for clause in clauses:
        if any(re.search(rf"\b{re.escape(word)}\b", clause) for word in words):
            return bool(_NEGATION_RE.search(clause))
    return bool(_NEGATION_RE.search(normalized))


def _representation_intent(raw_message: str, normalized: str) -> RepresentationIntent:
    # Generic named-option materialization ("vamos comparar A, B e C") is
    # a representation/workspace-object request even before the user names
    # a specific style.
    explicit = extract_explicit_alternatives(raw_message)
    if explicit:
        style = "side_by_side" if "comparar" in normalized else "cards"
        return RepresentationIntent(present=True, action="materialize_alternatives", style=style, confidence=0.85, evidence="explicit_alternatives")
    added = extract_added_alternatives(raw_message)
    if added:
        return RepresentationIntent(present=True, action="add_alternatives", style="", confidence=0.85, evidence="added_alternatives")
    style = _representation_style(normalized)

    if contains_any_phrase(normalized, _REPRESENTATION_SHOW_REASONING_PHRASES):
        return RepresentationIntent(present=True, action="show_reasoning", style=style, confidence=0.85, evidence="reasoning_visibility")
    if contains_any_phrase(normalized, _REPRESENTATION_HIDE_REASONING_PHRASES):
        return RepresentationIntent(present=True, action="hide_reasoning", style=style, confidence=0.85, evidence="reasoning_visibility")
    if contains_any_phrase(normalized, _REPRESENTATION_HIDE_PHRASES):
        return RepresentationIntent(present=True, action="hide_representation", style="conversation_only", confidence=0.9, evidence="hide")
    if contains_any_phrase(normalized, _REPRESENTATION_RESTORE_PHRASES):
        return RepresentationIntent(present=True, action="restore_representation", style=style, confidence=0.85, evidence="restore")
    if contains_any_phrase(normalized, _REPRESENTATION_SIMPLIFY_PHRASES):
        return RepresentationIntent(present=True, action="simplify", style=style, confidence=0.85, evidence="simplify")
    if contains_any_phrase(normalized, _REPRESENTATION_PARAM_PHRASES):
        return RepresentationIntent(present=True, action="parameter_mutation", style=style, confidence=0.85, evidence="parameter_mutation")
    if contains_any_phrase(normalized, _REPRESENTATION_FILTER_PHRASES):
        return RepresentationIntent(present=True, action="filter", style=style, confidence=0.8, evidence="filter")
    if contains_any_phrase(normalized, _REPRESENTATION_GROUP_PHRASES):
        # Phase 3C final patch: style is deliberately NOT passed through
        # here (unlike every other branch above) -- a grouping/
        # organization request never counts as "a style was explicitly
        # requested" even if the utterance happens to name "cards"/
        # "cartões" as the thing being grouped, not a style to switch to.
        return RepresentationIntent(
            present=True,
            action="grouping_requested",
            style="",
            confidence=0.8,
            evidence="grouping",
            grouping_requested=True,
            grouping_instruction=normalized,
            grouping_criterion=_extract_grouping_criterion(normalized),
        )
    if contains_any_phrase(normalized, _REPRESENTATION_ORDER_PHRASES):
        return RepresentationIntent(present=True, action="ordering", style=style, confidence=0.8, evidence="ordering")
    if contains_any_phrase(normalized, _REPRESENTATION_VISUAL_PHRASES):
        return RepresentationIntent(present=True, action="increase_visual_structure", style=style, confidence=0.75, evidence="increase_visual_structure")
    if style and _style_word_is_negated(normalized, style):
        return RepresentationIntent(present=True, action="rejected_style", style=style, confidence=0.85, evidence="rejected_style")
    if style:
        return RepresentationIntent(present=True, action="requested_style", style=style, confidence=0.8, evidence="requested_style")
    return RepresentationIntent()


def _attention_intent(normalized: str, work_session=None) -> AttentionIntent:
    if contains_any_phrase(normalized, ("opcoes", "opções")) and contains_any_phrase(normalized, ("outra vez", "de novo")):
        return AttentionIntent()
    # Order matters: show_again's two-part check is the most specific
    # (requires BOTH a show-verb and a recovery marker), so a plain
    # "mostra outra vez" alone (no recovery marker other than "outra
    # vez" itself, which IS one of the markers) still resolves here first,
    # ahead of the narrower single-phrase checks below.
    if _is_show_again_phrase(normalized):
        return AttentionIntent(present=True, action="show_again", target_phrase=normalized, confidence=0.75, evidence="show_again")
    if _is_keep_visible_phrase(normalized):
        return AttentionIntent(present=True, action="keep_visible", target_phrase=normalized, confidence=0.85, evidence="keep_visible")
    if contains_any_phrase(normalized, _ATTENTION_HIDE_OBJECT_PHRASES):
        return AttentionIntent(present=True, action="hide_object", target_phrase=normalized, confidence=0.8, evidence="hide_object")
    if contains_any_phrase(normalized, _ATTENTION_MINIMIZE_PHRASES):
        return AttentionIntent(present=True, action="minimize_object", target_phrase=normalized, confidence=0.85, evidence="minimize_object")
    if contains_any_phrase(normalized, _ATTENTION_FOCUS_PHRASES):
        return AttentionIntent(present=True, action="focus_object", target_phrase=normalized, confidence=0.8, evidence="focus_object")
    if _is_visual_return_to_active_task(normalized, work_session):
        return AttentionIntent(present=True, action="focus_object", target_phrase=normalized, confidence=0.75, evidence="return_to_active_task")
    return AttentionIntent()


def _reasoning_intent(normalized: str) -> ReasoningIntent:
    if contains_any_phrase(normalized, _REASONING_PHRASES):
        return ReasoningIntent(present=True, action="recommend_or_explain", confidence=0.85)
    return ReasoningIntent()


def _conversational_intent(
    normalized: str,
    activity: ActivityIntent,
    content: ContentIntent,
    spatial: SpatialIntent,
    representation: RepresentationIntent,
    reasoning: ReasoningIntent,
    attention: AttentionIntent,
) -> ConversationalIntent:
    # Section 7: the residual bucket -- present only when nothing else is,
    # so it can never absorb another axis's language.
    if (
        activity.present or content.present or spatial.present or representation.present
        or reasoning.present or attention.present
    ):
        return ConversationalIntent()
    if contains_any_phrase(normalized, _CONVERSATIONAL_MARKERS):
        return ConversationalIntent(present=True, confidence=0.8)
    if normalized:
        return ConversationalIntent(present=True, confidence=0.4)
    return ConversationalIntent()


def _collect_ambiguities(
    activity: ActivityIntent,
    content: ContentIntent,
    spatial: SpatialIntent,
    representation: RepresentationIntent,
    reasoning: ReasoningIntent,
    attention: AttentionIntent,
) -> list[str]:
    ambiguities: list[str] = []
    if not any([activity.present, content.present, spatial.present, representation.present, reasoning.present, attention.present]):
        ambiguities.append("no_deterministic_intent_matched")
    return ambiguities


def interpret(raw_message: str, work_session=None) -> InteractionInterpretation:
    """Pure, read-only classification of one utterance. Never mutates
    `work_session` or anything it holds."""
    normalized = _normalize(raw_message)
    activity = _activity_intent(raw_message, normalized, work_session)
    spatial = _spatial_intent(normalized)
    content = _content_intent(raw_message, normalized, work_session, activity)
    representation = _representation_intent(raw_message, normalized)
    reasoning = _reasoning_intent(normalized)
    attention = _attention_intent(normalized, work_session)
    conversational = _conversational_intent(normalized, activity, content, spatial, representation, reasoning, attention)
    ambiguities = _collect_ambiguities(activity, content, spatial, representation, reasoning, attention)
    confidence = max(
        (
            axis.confidence
            for axis in (activity, content, spatial, representation, reasoning, attention, conversational)
            if axis.present
        ),
        default=0.0,
    )
    return InteractionInterpretation(
        activity=activity,
        content=content,
        spatial=spatial,
        representation=representation,
        reasoning=reasoning,
        attention=attention,
        conversational=conversational,
        confidence=confidence,
        ambiguities=tuple(ambiguities),
    )


def interpret_shadow(raw_message: str, work_session=None) -> tuple[InteractionInterpretation, float]:
    """Same as interpret(), plus wall-clock latency in milliseconds --
    section 17's `interpreter_shadow_latency_ms`."""
    started = time.perf_counter()
    interpretation = interpret(raw_message, work_session)
    latency_ms = (time.perf_counter() - started) * 1000
    return interpretation, latency_ms


# --- Route expectation (section 12) ------------------------------------

ROUTE_WORKSPACE_COMMAND = "WORKSPACE_COMMAND"
ROUTE_TASK_CONTINUATION = "TASK_CONTINUATION"
ROUTE_TASK_EXECUTION = "TASK_EXECUTION"
ROUTE_REPRESENTATION_COMMAND = "REPRESENTATION_COMMAND"
# Phase 4 (section 20): explicit attention/visibility commands are now a
# SECOND authoritative interpreter axis alongside REPRESENTATION_COMMAND --
# see conversation.py's _try_attention_command, which runs just before
# _try_representation_command in the real waterfall (mirrored here by
# checking attention.present before representation.present below).
ROUTE_ATTENTION_COMMAND = "ATTENTION_COMMAND"
ROUTE_WORKSPACE_REFERENCE = "WORKSPACE_REFERENCE"
ROUTE_GENERAL_CONVERSATION = "GENERAL_CONVERSATION"


def decide_turn_route(interpretation: InteractionInterpretation, work_session=None) -> TurnRoutingDecision:
    """Single semantic priority contract for interpreter-owned axes.

    The decision does not execute work. It only states which subsystem gets
    first authority when the same utterance contains several recognizable
    intents. Pending confirmations still sit above this contract in
    AssistantEngine because they are control-flow state, not semantic
    interpretation.
    """
    secondary: list[str] = []
    unsupported: list[str] = []

    def add_secondary(label: str, present: bool) -> None:
        if present and label not in secondary:
            secondary.append(label)

    if interpretation.spatial.present:
        primary = ROUTE_WORKSPACE_COMMAND
        reason = "explicit_spatial_intent"
    elif interpretation.activity.present and interpretation.representation.present:
        primary = ROUTE_TASK_CONTINUATION
        reason = "activity_with_representation_intent"
    elif interpretation.attention.present:
        primary = ROUTE_ATTENTION_COMMAND
        reason = "explicit_attention_intent"
    elif interpretation.representation.present:
        primary = ROUTE_REPRESENTATION_COMMAND
        reason = "explicit_representation_intent"
    elif interpretation.activity.present or interpretation.content.present:
        primary = ROUTE_TASK_EXECUTION if interpretation.content.kind == "control_execute" else ROUTE_TASK_CONTINUATION
        reason = "task_activity_or_content_intent"
    elif interpretation.reasoning.present:
        active_task_present = bool(work_session and work_session.active_task is not None)
        primary = ROUTE_TASK_CONTINUATION if active_task_present else ROUTE_GENERAL_CONVERSATION
        reason = "reasoning_intent"
    elif interpretation.conversational.present:
        primary = ROUTE_GENERAL_CONVERSATION
        reason = "ordinary_conversation"
    else:
        primary = ""
        reason = "no_deterministic_intent"

    add_secondary(ROUTE_WORKSPACE_COMMAND, interpretation.spatial.present and primary != ROUTE_WORKSPACE_COMMAND)
    add_secondary(ROUTE_REPRESENTATION_COMMAND, interpretation.representation.present and primary != ROUTE_REPRESENTATION_COMMAND)
    add_secondary(ROUTE_ATTENTION_COMMAND, interpretation.attention.present and primary != ROUTE_ATTENTION_COMMAND)
    add_secondary(ROUTE_TASK_EXECUTION if interpretation.content.kind == "control_execute" else ROUTE_TASK_CONTINUATION, (interpretation.activity.present or interpretation.content.present) and primary not in (ROUTE_TASK_CONTINUATION, ROUTE_TASK_EXECUTION))
    if interpretation.representation.present and interpretation.representation.action in ("simplify", "filter", "grouping_requested", "ordering", "parameter_mutation", "increase_visual_structure"):
        unsupported.append(interpretation.representation.action)

    return TurnRoutingDecision(
        primary_route=primary,
        secondary_intents=tuple(secondary),
        confidence=interpretation.confidence,
        unsupported_intents=tuple(unsupported),
        ambiguity=bool(interpretation.ambiguities),
        reason_code=reason,
    )


def expected_route(interpretation: InteractionInterpretation, work_session=None) -> str:
    """Diagnostics prediction backed by the same priority contract as
    decide_turn_route()."""
    return decide_turn_route(interpretation, work_session).primary_route

# Comparison verdicts -- section 12/13. Deliberately distinct from
# fault_detected (see conversation.py's _run_shadow_interpreter): this is
# an evidence-collection label, not a claim that production is broken.
COMPARISON_MATCH = "MATCH"
COMPARISON_MISMATCH = "MISMATCH"
COMPARISON_UNSUPPORTED = "UNSUPPORTED_BY_CURRENT_ROUTER"
COMPARISON_AMBIGUOUS = "AMBIGUOUS"


def compare_route(expected: str, actual: str) -> str:
    if not expected:
        return COMPARISON_AMBIGUOUS
    if expected == ROUTE_REPRESENTATION_COMMAND:
        # Phase 3 (Cognitive Workspace Blueprint v2) added a real, narrow
        # REPRESENTATION_COMMAND production route (see conversation.py's
        # _try_representation_command) -- when it actually fired, that is
        # a genuine MATCH now, not an unconditional UNSUPPORTED. When the
        # narrow bridge did not apply (e.g. WorkSession.objects is empty,
        # or the requested style is outside cards/list/side_by_side/
        # conversation_only), there is still no production route for it,
        # so UNSUPPORTED remains the correct verdict.
        return COMPARISON_MATCH if actual == ROUTE_REPRESENTATION_COMMAND else COMPARISON_UNSUPPORTED
    if expected == ROUTE_ATTENTION_COMMAND:
        # Same reasoning as ROUTE_REPRESENTATION_COMMAND above: this IS a
        # real, narrow production route now (conversation.py's
        # _try_attention_command), so a match is genuine -- not every
        # attention-phrase-shaped utterance necessarily has a resolvable
        # target object though, so UNSUPPORTED remains possible.
        return COMPARISON_MATCH if actual == ROUTE_ATTENTION_COMMAND else COMPARISON_UNSUPPORTED
    # No TASK_EXECUTION/TASK_CONTINUATION normalization here: production
    # actually sets selected_path="TASK_EXECUTION" as its own literal value
    # (conversation.py's _execute_active_task, active_task_selected_path)
    # -- confirmed directly against real turns (section 15), not assumed.
    return COMPARISON_MATCH if expected == actual else COMPARISON_MISMATCH
