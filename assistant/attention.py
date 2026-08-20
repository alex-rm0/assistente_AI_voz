"""Phase 4 of the Cognitive Workspace Blueprint v2 -- ATTENTION MODEL.

Answers "how important is this object right now?" -- a question deliberately
kept SEPARATE from assistant.representation's "how should the current
information be shown?" (section 3 of the phase spec). The same semantic
object can be RepresentationState-major (side_by_side) while being
AttentionLevel.ACTIVE rather than FOCUS, or vice-versa; nothing here reads
or writes RepresentationState, and nothing in representation.py reads or
writes AttentionState.

Mirrors assistant/representation.py's own conventions exactly:
- independent of assistant.conversation (never imports it, even lazily --
  there is nothing in conversation.py this module needs);
- every function takes a `work_session` duck-typed to expose `.attention`
  (AttentionState | None) and `.objects` (dict[str, SemanticObject], read-
  only here, never mutated);
- deterministic only -- no LLM/ML ranking, per the phase's explicit "do not
  build a large ML/LLM attention system yet" instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Attention levels (section 1/2) -------------------------------------
# Ordered loosely from most to least prominent; HIDDEN is NOT deleted (the
# object still exists in work_session.objects) -- it is simply not shown.

FOCUS = "focus"
ACTIVE = "active"
VISIBLE = "visible"
GLANCEABLE = "glanceable"
MINIMIZED = "minimized"
HIDDEN = "hidden"

ATTENTION_LEVELS = (FOCUS, ACTIVE, VISIBLE, GLANCEABLE, MINIMIZED, HIDDEN)

SOURCE_AUTO = "auto"
SOURCE_VOICE_COMMAND = "voice_command"
SOURCE_POINTER = "pointer"
SOURCE_ECHO_SUGGESTION = "echo_suggestion"


@dataclass
class AttentionState:
    """Section 1: attention belongs to semantic OBJECT ids, never to a DOM
    node and never duplicated as a second identity -- `object_states` keys
    are exactly the same ids assistant.representation.SemanticObject.id
    already uses. `revision` is a plain monotonic counter (mirrors
    RepresentationState's own lightweight versioning need) so a frontend
    can trivially tell "this attention_changed event is newer than the one
    I already applied" without comparing whole dicts."""

    object_states: dict[str, str] = field(default_factory=dict)
    focus_object_id: str = ""
    source: str = SOURCE_AUTO
    revision: int = 0


def get_attention_level(work_session, object_id: str) -> str:
    """Unknown objects (never assigned a level) read as HIDDEN -- the safe
    default; nothing is prominent until something explicitly says so."""
    state = work_session.attention
    if state is None:
        return HIDDEN
    return state.object_states.get(object_id, HIDDEN)


def _next_state(work_session, object_states: dict[str, str], focus_object_id: str, source: str) -> AttentionState:
    current = work_session.attention
    revision = (current.revision + 1) if current is not None else 1
    new_state = AttentionState(
        object_states=object_states, focus_object_id=focus_object_id, source=source, revision=revision,
    )
    work_session.attention = new_state
    return new_state


def set_attention(work_session, object_id: str, level: str, *, source: str = SOURCE_AUTO) -> AttentionState:
    """THE single per-object attention-transition operation (mirrors
    representation.transform_representation's "one central operation"
    design) -- section 2's FOCUS/ACTIVE/VISIBLE/GLANCEABLE/MINIMIZED/HIDDEN
    transitions all go through here. At most one object is ever FOCUS at a
    time: promoting a new one demotes whatever was FOCUS before to ACTIVE
    (never silently drops it to HIDDEN)."""
    if level not in ATTENTION_LEVELS:
        raise ValueError(f"unsupported attention level: {level!r}")
    current = work_session.attention
    object_states = dict(current.object_states) if current is not None else {}
    focus_object_id = current.focus_object_id if current is not None else ""

    object_states[object_id] = level
    if level == FOCUS:
        if focus_object_id and focus_object_id != object_id and object_states.get(focus_object_id) == FOCUS:
            object_states[focus_object_id] = ACTIVE
        focus_object_id = object_id
    elif focus_object_id == object_id:
        focus_object_id = ""

    return _next_state(work_session, object_states, focus_object_id, source)


def set_bulk_attention(work_session, levels: dict[str, str], *, source: str = SOURCE_AUTO) -> AttentionState:
    """Applies several attention-level assignments as ONE revision bump --
    e.g. materializing three new alternatives at once, or demoting every
    previously-ACTIVE object by one step when a new activity starts.
    Avoids the "one throwaway AttentionState per object" churn set_attention
    called in a loop would otherwise produce (each intermediate revision
    would queue its own frontend event for no reason)."""
    current = work_session.attention
    object_states = dict(current.object_states) if current is not None else {}
    focus_object_id = current.focus_object_id if current is not None else ""

    for object_id, level in levels.items():
        if level not in ATTENTION_LEVELS:
            raise ValueError(f"unsupported attention level: {level!r}")
        object_states[object_id] = level
        if level == FOCUS:
            if focus_object_id and focus_object_id != object_id and object_states.get(focus_object_id) == FOCUS:
                object_states[focus_object_id] = ACTIVE
            focus_object_id = object_id
        elif focus_object_id == object_id:
            focus_object_id = ""

    return _next_state(work_session, object_states, focus_object_id, source)


def promote_to_focus(work_session, object_id: str, *, source: str = SOURCE_VOICE_COMMAND) -> AttentionState:
    """Section 4: 'when the user explicitly selects/opens/works on an
    object -> promote it toward FOCUS.'"""
    return set_attention(work_session, object_id, FOCUS, source=source)


def hide_object(work_session, object_id: str, *, source: str = SOURCE_VOICE_COMMAND) -> AttentionState:
    return set_attention(work_session, object_id, HIDDEN, source=source)


def minimize_object(work_session, object_id: str, *, source: str = SOURCE_VOICE_COMMAND) -> AttentionState:
    return set_attention(work_session, object_id, MINIMIZED, source=source)


# --- Minimization candidate ranking (section 14) ------------------------
# A generic, deterministic ranking -- attention level is the primary key
# (GLANCEABLE objects are always safer to minimize than ACTIVE ones), never
# a hardcoded object `kind`/domain check. FOCUS/MINIMIZED/HIDDEN objects are
# never returned: FOCUS is what the user is doing right now (section 12:
# "do NOT automatically minimize a relevant ACTIVE/FOCUS object without
# user permission" -- excluding it from the candidate list entirely is the
# simplest way to make that true structurally, not just by convention),
# MINIMIZED/HIDDEN are already out of the way.

_MINIMIZE_RANK = {GLANCEABLE: 0, VISIBLE: 1, ACTIVE: 2}


def rank_minimization_candidates(work_session, *, exclude_ids: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Returns candidate object ids ordered from SAFEST-to-minimize first.
    `exclude_ids` is the caller's job to populate with anything protected --
    typically the current FOCUS id plus RepresentationConstraint.
    do_not_minimize_ids (section 5/10: explicit keep-open constraints always
    win). Stable order (Python dict iteration order, i.e. attention-
    assignment order) breaks ties deterministically rather than randomly."""
    state = work_session.attention
    if state is None:
        return []
    candidates = [
        object_id
        for object_id, level in state.object_states.items()
        if object_id not in exclude_ids and level in _MINIMIZE_RANK
    ]
    candidates.sort(key=lambda object_id: _MINIMIZE_RANK[state.object_states[object_id]])
    return candidates
