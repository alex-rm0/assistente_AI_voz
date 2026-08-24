"""Phase 2 (Cognitive Workspace Blueprint v2): standalone, engine-free unit
tests for assistant/interaction_interpreter.py. These test the pure
classification function directly -- no AssistantEngine, no provider, no
WorkSession required for most cases (a few pass a real WorkSession to
prove read-only behavior explicitly)."""

from __future__ import annotations

from assistant.conversation import ActiveTaskContext, WorkSession
from assistant.interaction_interpreter import (
    COMPARISON_AMBIGUOUS,
    COMPARISON_MATCH,
    COMPARISON_MISMATCH,
    COMPARISON_UNSUPPORTED,
    ROUTE_ATTENTION_COMMAND,
    ROUTE_GENERAL_CONVERSATION,
    ROUTE_REPRESENTATION_COMMAND,
    ROUTE_TASK_CONTINUATION,
    ROUTE_WORKSPACE_COMMAND,
    compare_route,
    expected_route,
    interpret,
    interpret_shadow,
)


def test_interpreter_a_activity_present_for_email_trigger() -> None:
    """A: 'vamos escrever um email' -> ActivityIntent present, writing/
    communication classification."""
    interp = interpret("vamos escrever um email")

    assert interp.activity.present is True
    assert interp.activity.starts_task is True
    assert interp.activity.activity_family == "communication"
    assert interp.content.present is False
    assert interp.spatial.present is False


def test_interpreter_b_spatial_only() -> None:
    """B: 'chega-te para a esquerda' -> SpatialIntent only."""
    interp = interpret("chega-te para a esquerda")

    assert interp.spatial.present is True
    assert interp.spatial.action == "move"
    assert interp.spatial.region == "left"
    assert interp.activity.present is False
    assert interp.content.present is False
    assert interp.representation.present is False
    assert interp.reasoning.present is False
    assert interp.conversational.present is False


def test_interpreter_c_activity_and_spatial_simultaneously() -> None:
    """C: 'vamos escrever um email e chega-te para a esquerda' -> Activity
    + Spatial present together, in ONE classification pass (no
    short-circuiting after the first match)."""
    interp = interpret("vamos escrever um email e chega-te para a esquerda")

    assert interp.activity.present is True
    assert interp.spatial.present is True
    assert interp.spatial.region == "left"
    assert interp.multi_intent_count == 2


def test_interpreter_d_representation_hide_conversation_only() -> None:
    """D: 'não quero caixas, fala só comigo' -> RepresentationIntent:
    conversation_only / hide current visual representation."""
    interp = interpret("não quero caixas, fala só comigo")

    assert interp.representation.present is True
    assert interp.representation.action == "hide_representation"
    assert interp.representation.style == "conversation_only"


def test_interpreter_e_spatial_and_reasoning_simultaneously() -> None:
    """E: 'põe as duas opções lado a lado e diz-me qual escolherias' ->
    Spatial (side-by-side placement, reusing the existing workspace-command
    detector) + Reasoning (recommend between them) simultaneously."""
    interp = interpret("põe as duas opções lado a lado e diz-me qual escolherias")

    assert interp.spatial.present is True
    assert interp.spatial.action == "place_beside"
    assert interp.reasoning.present is True
    assert interp.multi_intent_count == 2


def test_interpreter_f_representation_table() -> None:
    """F: 'faz uma tabela' -> RepresentationIntent: table."""
    interp = interpret("faz uma tabela")

    assert interp.representation.present is True
    assert interp.representation.style == "table"
    assert interp.representation.action == "requested_style"


def test_interpreter_g_representation_parameter_mutation() -> None:
    """G: 'troca os eixos' -> RepresentationIntent: parameter mutation."""
    interp = interpret("troca os eixos")

    assert interp.representation.present is True
    assert interp.representation.action == "parameter_mutation"


def test_interpreter_h_filter_representation_not_reasoning_question() -> None:
    """H: 'mostra só os riscos' -- classified as RepresentationIntent
    (action=filter): this is an IMPERATIVE display instruction ("show me
    only X"), a projection over existing knowledge/reasoning, not a
    request for Echo to newly compute a judgement. The QUESTION form
    ('quais são os riscos?') is deliberately the one that classifies as
    ReasoningIntent instead (see test I) -- imperative-filter vs.
    question-asking-for-judgement is the chosen, documented distinction."""
    interp = interpret("mostra só os riscos")

    assert interp.representation.present is True
    assert interp.representation.action == "filter"
    assert interp.reasoning.present is False


def test_interpreter_i_reasoning_question_form() -> None:
    """I: 'porque estás a preferir o iPhone?' -> ReasoningIntent."""
    interp = interpret("porque estás a preferir o iPhone?")

    assert interp.reasoning.present is True
    assert interp.representation.present is False


def test_interpreter_j_conversational_only() -> None:
    """J: 'olá, tudo bem?' -> Conversational only."""
    interp = interpret("olá, tudo bem?")

    assert interp.conversational.present is True
    assert interp.activity.present is False
    assert interp.content.present is False
    assert interp.spatial.present is False
    assert interp.representation.present is False
    assert interp.reasoning.present is False


def test_interpreter_detects_generic_comparison_activity() -> None:
    interp = interpret("enquanto isso, estou a pensar comprar um telefone novo, vamos comparar Samsung com iPhone")

    assert interp.activity.present is True
    assert interp.activity.starts_task is True
    assert interp.activity.task_type == "generic_comparison"
    assert interp.activity.activity_family == "comparison"
    assert expected_route(interp) == ROUTE_TASK_CONTINUATION


def test_interpreter_detects_generic_organization_activity() -> None:
    interp = interpret("vamos organizar uma festa")

    assert interp.activity.present is True
    assert interp.activity.starts_task is True
    assert interp.activity.task_type == "generic_activity"
    assert interp.activity.activity_family == "activity"


def test_interpreter_does_not_turn_factual_questions_into_activities() -> None:
    for message in ("qual é a capital de França?", "quanto é 5x8?", "quem inventou o telefone?"):
        interp = interpret(message)
        assert interp.activity.present is False


def test_interpreter_k_three_simultaneous_intents() -> None:
    """K: 'não faças mapa, deixa só as duas opções lado a lado e diz-me
    qual escolherias' -> at least 3 simultaneous intents (representation +
    spatial + reasoning)."""
    interp = interpret("não faças mapa, deixa só as duas opções lado a lado e diz-me qual escolherias")

    assert interp.representation.present is True
    assert interp.representation.action == "rejected_style"
    assert interp.representation.style == "mind_map"
    assert interp.spatial.present is True
    assert interp.reasoning.present is True
    assert interp.multi_intent_count >= 3


def test_interpreter_l_simplify_without_naming_a_renderer() -> None:
    """L: 'não compliques' -> RepresentationIntent recognizes
    simplification even without naming a specific style/renderer."""
    interp = interpret("não compliques")

    assert interp.representation.present is True
    assert interp.representation.action == "simplify"
    assert interp.representation.style == ""


def test_interpreter_m_spatial_plus_constraint_like_signal() -> None:
    """M: 'mantém estes dois documentos lado a lado' -> Spatial intent,
    with a constraint_like signal set -- real WorkspaceConstraint behavior
    (typed, persisted, strength-ranked) belongs to a later phase; this is
    classification only."""
    interp = interpret("mantém estes dois documentos lado a lado")

    assert interp.spatial.present is True
    assert interp.spatial.action == "place_beside"
    assert interp.spatial.constraint_like is True


# --- Route expectation / comparison (section 12) ------------------------

def test_route_workspace_command_wins_over_simultaneous_activity() -> None:
    """Mirrors production ordering: _try_workspace_command runs before
    _try_active_task_turn, so a spatial+activity utterance's expected
    route is WORKSPACE_COMMAND, not TASK_CONTINUATION."""
    interp = interpret("vamos escrever um email e chega-te para a esquerda")

    assert expected_route(interp) == ROUTE_WORKSPACE_COMMAND


def test_route_representation_only_is_unsupported_by_current_router() -> None:
    interp = interpret("faz uma tabela")

    predicted = expected_route(interp)
    assert predicted == ROUTE_REPRESENTATION_COMMAND
    assert compare_route(predicted, "GENERAL_CONVERSATION") == COMPARISON_UNSUPPORTED


def test_route_reasoning_with_active_task_expects_task_continuation() -> None:
    work_session = WorkSession(active_task=ActiveTaskContext(id="task-1"))
    interp = interpret("porque estás a preferir o iPhone?", work_session)

    assert expected_route(interp, work_session) == ROUTE_TASK_CONTINUATION


def test_route_reasoning_without_active_task_expects_general_conversation() -> None:
    interp = interpret("porque estás a preferir o iPhone?", None)

    assert expected_route(interp, None) == ROUTE_GENERAL_CONVERSATION


def test_route_compare_match_and_mismatch() -> None:
    assert compare_route(ROUTE_GENERAL_CONVERSATION, "GENERAL_CONVERSATION") == COMPARISON_MATCH
    assert compare_route(ROUTE_GENERAL_CONVERSATION, "TASK_CONTINUATION") == COMPARISON_MISMATCH
    assert compare_route("", "GENERAL_CONVERSATION") == COMPARISON_AMBIGUOUS


# --- WorkSession is never mutated (section 16) ---------------------------

def test_interpreter_never_mutates_work_session() -> None:
    work_session = WorkSession(active_task=ActiveTaskContext(id="task-1", status="ready"))
    snapshot_status = work_session.active_task.status
    snapshot_entities = dict(work_session.active_task.entities)
    snapshot_paused = list(work_session.paused_task_stack)

    interpret("é para o meu professor de estrategias algoritmicas luis paquete", work_session)
    interpret("podes escrever", work_session)
    interpret("chega-te para a esquerda, mantém isto aqui", work_session)

    assert work_session.active_task.status == snapshot_status
    assert work_session.active_task.entities == snapshot_entities
    assert work_session.paused_task_stack == snapshot_paused


def test_interpret_shadow_reports_latency() -> None:
    interpretation, latency_ms = interpret_shadow("chega-te para a esquerda")

    assert interpretation.spatial.present is True
    assert latency_ms >= 0.0
    assert latency_ms < 50.0  # generous ceiling; deterministic cases are sub-millisecond in practice


# --- Phase 3C final patch: a grouping/organization request ("separa os
# cards dos destinos internacionais dos destinos portugueses") is
# semantically distinct from a style request ("mostra em cartões") even
# though it happens to name "cards" -- the real Qt bug was reading it as
# requested_style="cards" purely because that bare word appears in it.

def test_interpreter_f_style_request_is_not_grouping() -> None:
    """F: 'mostra em cartões' -> requested_style=cards, grouping_requested
    stays False."""
    interp = interpret("mostra em cartões")

    assert interp.representation.action == "requested_style"
    assert interp.representation.style == "cards"
    assert interp.representation.grouping_requested is False


def test_interpreter_g_grouping_is_not_read_as_cards_style() -> None:
    """G: naming 'cards' inside a grouping/separation request must NOT be
    read as requested_style=cards -- grouping_requested must be True and
    style must stay empty (no style was actually requested)."""
    interp = interpret("separa os cards dos destinos internacionais dos destinos portugueses")

    assert interp.representation.present is True
    assert interp.representation.grouping_requested is True
    assert interp.representation.style != "cards"
    assert interp.representation.action != "requested_style"


def test_interpreter_h_agrupa_por_prioridade_is_grouping_intent() -> None:
    """H: 'agrupa estas opções por prioridade' -> grouping intent, with a
    deterministically-extractable criterion."""
    interp = interpret("agrupa estas opções por prioridade")

    assert interp.representation.grouping_requested is True
    assert interp.representation.grouping_criterion == "prioridade"


def test_interpreter_grouping_separa_portugueses_dos_internacionais() -> None:
    """Additional example from the bug report: no 'por X' clause here, so
    grouping_criterion deterministically stays empty rather than guessing
    at the two named groups (Phase 3C does not classify destinations)."""
    interp = interpret("separa os portugueses dos internacionais")

    assert interp.representation.grouping_requested is True
    assert interp.representation.grouping_criterion == ""


def test_interpreter_grouping_junta_opcoes_parecidas() -> None:
    interp = interpret("junta as opções parecidas")

    assert interp.representation.grouping_requested is True


# --- Phase 4 (Cognitive Workspace Blueprint v2): AttentionIntent, a new
# axis distinct from SpatialIntent (WHERE) and RepresentationIntent (HOW
# THE WHOLE SET is shown) -- explicit per-object attention/visibility
# commands (section 5/20).

def test_interpreter_attention_keep_visible() -> None:
    interp = interpret("não escondas o documento")

    assert interp.attention.present is True
    assert interp.attention.action == "keep_visible"
    assert interp.representation.present is False
    assert interp.spatial.present is False


def test_interpreter_attention_hide_object() -> None:
    interp = interpret("isto não preciso de ver")

    assert interp.attention.present is True
    assert interp.attention.action == "hide_object"


def test_interpreter_attention_minimize_object() -> None:
    interp = interpret("minimiza o email")

    assert interp.attention.present is True
    assert interp.attention.action == "minimize_object"


def test_interpreter_attention_focus_object() -> None:
    interp = interpret("traz o relatório para a frente")

    assert interp.attention.present is True
    assert interp.attention.action == "focus_object"


def test_interpreter_attention_show_again_requires_a_named_object() -> None:
    """A generic "mostra outra vez" (nothing named beyond the bare show-
    verb + recovery marker) must NOT be read as an attention command -- it
    is the EXISTING Phase 3 restore_representation trigger for the WHOLE
    workspace, and this axis must never steal it (a real regression caught
    while wiring this axis into conversation.py's waterfall)."""
    bare = interpret("mostra outra vez")
    assert bare.attention.present is False
    assert bare.representation.present is True
    assert bare.representation.action == "restore_representation"

    named = interpret("mostra o email outra vez")
    assert named.attention.present is True
    assert named.attention.action == "show_again"

    named2 = interpret("abre o documento que minimizaste")
    assert named2.attention.present is True
    assert named2.attention.action == "show_again"


def test_interpreter_attention_absent_for_unrelated_text() -> None:
    interp = interpret("olá, tudo bem?")

    assert interp.attention.present is False


def test_interpreter_attention_included_in_multi_intent_count() -> None:
    interp = interpret("minimiza o email")

    assert interp.multi_intent_count == 1
    assert interp.attention.present is True


def test_route_attention_command_is_a_distinct_route() -> None:
    interp = interpret("minimiza o email")

    predicted = expected_route(interp)
    assert predicted == ROUTE_ATTENTION_COMMAND
    assert compare_route(predicted, "ATTENTION_COMMAND") == COMPARISON_MATCH
    assert compare_route(predicted, "GENERAL_CONVERSATION") == COMPARISON_UNSUPPORTED


def test_interpreter_reasoning_visibility_show_vs_hide_are_distinct_actions() -> None:
    """Phase 4 (section 19): the previously-unwired single
    "reasoning_visibility_request" action is split into two distinct,
    directly-actionable actions so conversation.py's handler doesn't have
    to re-parse the utterance to know which direction was asked for."""
    show = interpret("mostra-me porque escolheste isto")
    assert show.representation.present is True
    assert show.representation.action == "show_reasoning"

    hide = interpret("não preciso de ver o raciocínio")
    assert hide.representation.present is True
    assert hide.representation.action == "hide_reasoning"


def test_interpreter_never_mutates_work_session_with_attention_phrases() -> None:
    work_session = WorkSession(active_task=ActiveTaskContext(id="task-1", status="ready"))
    snapshot_status = work_session.active_task.status

    interpret("minimiza o email", work_session)
    interpret("não escondas o documento", work_session)
    interpret("mostra o email outra vez", work_session)

    assert work_session.active_task.status == snapshot_status
    assert work_session.attention is None  # interpret() never writes WorkSession itself


# --- Phase 4 manual-test fix: "mantém X visível"/"deixa X aberto" (a named
# object, not the deictic "isto"/"aqui" phrasing the original literal
# phrase list covered) must still classify as a deterministic
# AttentionIntent -- a real Qt run showed this falling through to
# GENERAL_CONVERSATION with a fabricated success-shaped LLM response and
# no attention_changed event at all.

def test_interpreter_mantem_x_visivel_is_keep_visible() -> None:
    interp = interpret("mantém Açores visível")

    assert interp.attention.present is True
    assert interp.attention.action == "keep_visible"
    assert interp.spatial.present is False  # never collides with Echo's own "mantém-te" placement axis


def test_interpreter_deixa_x_aberto_is_keep_visible() -> None:
    interp = interpret("deixa Açores aberto")

    assert interp.attention.present is True
    assert interp.attention.action == "keep_visible"


def test_interpreter_deixa_x_visivel_is_keep_visible() -> None:
    interp = interpret("deixa o Documento A visível")

    assert interp.attention.present is True
    assert interp.attention.action == "keep_visible"


def test_interpreter_mantem_te_placement_is_not_read_as_keep_visible() -> None:
    """The generic verb+state check must not fire on Echo's own spatial
    "mantém-te aqui" placement family, which never mentions visível/
    aberto."""
    interp = interpret("mantém-te aqui, para a esquerda")

    assert interp.attention.present is False
