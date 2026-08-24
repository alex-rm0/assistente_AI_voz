"""Phase 4 of the Cognitive Workspace Blueprint v2 -- ATTENTION MODEL.

Standalone, engine-free unit tests for assistant/attention.py, mirroring
tests/test_automatic_routing_regressions.py's own style of testing
assistant/representation.py directly against a minimal duck-typed
work_session stand-in (a real WorkSession is not needed for these -- only
`.attention` and `.objects` are ever read/written)."""

from __future__ import annotations

from dataclasses import dataclass, field

from assistant.attention import (
    ACTIVE,
    FOCUS,
    GLANCEABLE,
    HIDDEN,
    MINIMIZED,
    VISIBLE,
    AttentionState,
    get_attention_level,
    hide_object,
    minimize_object,
    promote_to_focus,
    rank_minimization_candidates,
    set_attention,
    set_bulk_attention,
)


@dataclass
class _FakeSession:
    attention: AttentionState | None = None
    objects: dict = field(default_factory=dict)
    constraints: object | None = None
    active_task: object | None = None


@dataclass
class _FakeConstraints:
    keep_visible_ids: list[str] = field(default_factory=list)
    do_not_minimize_ids: list[str] = field(default_factory=list)


@dataclass
class _FakeTask:
    workspace_object_ids: list[str] = field(default_factory=list)
    active_artifact_id: str = ""


# --- A: AttentionState defaults ------------------------------------------

def test_a_attention_state_defaults() -> None:
    state = AttentionState()
    assert state.object_states == {}
    assert state.focus_object_id == ""
    assert state.source == "auto"
    assert state.revision == 0


def test_a_unknown_object_reads_as_hidden() -> None:
    session = _FakeSession()
    assert get_attention_level(session, "obj-x") == HIDDEN


# --- B: attention promotion/demotion --------------------------------------

def test_b_set_attention_promotes_and_bumps_revision() -> None:
    session = _FakeSession()
    state1 = set_attention(session, "obj-a", ACTIVE, source="auto")
    assert state1.object_states["obj-a"] == ACTIVE
    assert state1.revision == 1
    state2 = set_attention(session, "obj-a", FOCUS, source="voice_command")
    assert state2.object_states["obj-a"] == FOCUS
    assert state2.focus_object_id == "obj-a"
    assert state2.revision == 2


def test_b_demotion_from_focus_to_active() -> None:
    session = _FakeSession()
    set_attention(session, "obj-a", FOCUS)
    state = set_attention(session, "obj-a", ACTIVE)
    assert state.object_states["obj-a"] == ACTIVE
    assert state.focus_object_id == ""


# --- C: FOCUS/ACTIVE/VISIBLE/GLANCEABLE/MINIMIZED/HIDDEN transitions ------

def test_c_only_one_focus_at_a_time() -> None:
    session = _FakeSession()
    set_attention(session, "obj-a", FOCUS)
    state = set_attention(session, "obj-b", FOCUS)
    assert state.focus_object_id == "obj-b"
    assert state.object_states["obj-a"] == ACTIVE  # demoted, never dropped
    assert state.object_states["obj-b"] == FOCUS


def test_c_all_levels_are_reachable() -> None:
    session = _FakeSession()
    for level in (FOCUS, ACTIVE, VISIBLE, GLANCEABLE, MINIMIZED, HIDDEN):
        state = set_attention(session, "obj-a", level)
        assert state.object_states["obj-a"] == level


def test_c_hidden_is_not_deleted() -> None:
    session = _FakeSession()
    session.objects["obj-a"] = object()
    hide_object(session, "obj-a")
    assert "obj-a" in session.objects  # still exists semantically
    assert get_attention_level(session, "obj-a") == HIDDEN


def test_c_promote_to_focus_and_minimize_helpers() -> None:
    session = _FakeSession()
    promote_to_focus(session, "obj-a", source="voice_command")
    assert session.attention.focus_object_id == "obj-a"
    minimize_object(session, "obj-a", source="voice_command")
    assert session.attention.object_states["obj-a"] == MINIMIZED
    assert session.attention.focus_object_id == ""


def test_c_set_bulk_attention_single_revision() -> None:
    session = _FakeSession()
    state = set_bulk_attention(session, {"obj-a": FOCUS, "obj-b": ACTIVE, "obj-c": ACTIVE}, source="auto")
    assert state.revision == 1
    assert state.focus_object_id == "obj-a"
    assert state.object_states == {"obj-a": FOCUS, "obj-b": ACTIVE, "obj-c": ACTIVE}


def test_c_invalid_level_raises() -> None:
    session = _FakeSession()
    try:
        set_attention(session, "obj-a", "not_a_real_level")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unsupported attention level")


# --- I: FOCUS object never appears in minimization candidates ------------

def test_i_focus_object_never_a_minimization_candidate() -> None:
    session = _FakeSession()
    set_bulk_attention(session, {"obj-a": FOCUS, "obj-b": ACTIVE, "obj-c": GLANCEABLE})
    candidates = rank_minimization_candidates(session)
    assert "obj-a" not in candidates
    assert candidates[0] == "obj-c"  # GLANCEABLE ranks safest first
    assert "obj-b" in candidates


# --- J: do_not_minimize / exclude_ids respected ---------------------------

def test_j_excluded_ids_never_returned() -> None:
    session = _FakeSession()
    set_bulk_attention(session, {"obj-a": ACTIVE, "obj-b": GLANCEABLE, "obj-c": VISIBLE})
    candidates = rank_minimization_candidates(session, exclude_ids={"obj-b"})
    assert "obj-b" not in candidates
    assert candidates[0] == "obj-c"  # next-safest after excluding GLANCEABLE obj-b


def test_rank_minimization_candidates_empty_without_attention_state() -> None:
    session = _FakeSession()
    assert rank_minimization_candidates(session) == []


def test_rank_minimization_candidates_excludes_minimized_and_hidden() -> None:
    session = _FakeSession()
    set_bulk_attention(session, {"obj-a": MINIMIZED, "obj-b": HIDDEN, "obj-c": ACTIVE})
    candidates = rank_minimization_candidates(session)
    assert candidates == ["obj-c"]


def test_rank_minimization_candidates_respects_session_constraints() -> None:
    session = _FakeSession(constraints=_FakeConstraints(keep_visible_ids=["obj-a"], do_not_minimize_ids=["obj-b"]))
    set_bulk_attention(session, {"obj-a": GLANCEABLE, "obj-b": GLANCEABLE, "obj-c": VISIBLE})

    candidates = rank_minimization_candidates(session)

    assert candidates == ["obj-c"]


def test_rank_minimization_candidates_prefers_non_current_task_objects() -> None:
    session = _FakeSession(active_task=_FakeTask(workspace_object_ids=["task-owned"]))
    set_bulk_attention(session, {"task-owned": VISIBLE, "unrelated": VISIBLE})

    candidates = rank_minimization_candidates(session)

    assert candidates == ["unrelated", "task-owned"]
