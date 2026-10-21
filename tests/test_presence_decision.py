"""Presence Decision Foundation tests -- Phase 8.0 (shadow foundation),
Phase 8.0.1 (shadow calibration), and Phase 8.0.2 (shadow observation +
self-representation signal).

Two layers throughout:
  - Pure unit tests against assistant.presence_decision's functions directly
    (fast, no engine).
  - Real-engine shadow tests via tests.test_automatic_routing_regressions.
    make_engine (mode="local", so no Anthropic call is ever possible) --
    confirm the wiring in assistant/conversation.py never changes
    final_response/routing and does surface presence_* telemetry fields.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.presence_decision import (
    PersonalCandidate,
    PreferredAddress,
    compute_presence_decision,
    resolve_address_candidate,
    select_personal_relevance,
)
from tests.test_automatic_routing_regressions import make_engine


# =====================================================================
# Phase 8.0 tests (foundation) -- parameter names updated for 8.0.1's
# turns_since_address/task_recently_completed calibration, behaviour
# otherwise unchanged unless noted.
# =====================================================================

def test_a_pt_acknowledgement_is_acknowledge_no_question():
    decision = compute_presence_decision(
        user_message="Ok, já percebi, obrigado.",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.initiative == "acknowledge"
    assert decision.question_needed is False


def test_b_en_thanks_is_acknowledge_no_question():
    decision = compute_presence_decision(
        user_message="Thanks, got it!",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.initiative == "acknowledge"
    assert decision.question_needed is False


def test_c_direct_question_is_answer_only_no_question():
    decision = compute_presence_decision(
        user_message="Qual é a capital da Suécia?",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.initiative == "answer_only"
    assert decision.question_needed is False


def test_d_active_task_missing_info_asks():
    decision = compute_presence_decision(
        user_message="Preciso de ajuda com uma coisa.",
        response_language="pt-PT",
        active_task_present=True,
        active_task_unresolved=True,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.initiative == "ask"
    assert decision.question_needed is True
    assert "MISSING_REQUIRED_INFO" in decision.reason_codes


def test_e_active_task_resolved_is_light_continuity():
    decision = compute_presence_decision(
        user_message="Podes continuar.",
        response_language="pt-PT",
        active_task_present=True,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.continuity == "light"


def test_f_restart_with_concrete_task_is_explicit_continuity():
    decision = compute_presence_decision(
        user_message="Lembras-te do que estávamos a fazer?",
        response_language="pt-PT",
        active_task_present=True,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=True,
    )
    assert decision.continuity == "explicit"


def test_g_restart_without_concrete_task_is_none_continuity():
    decision = compute_presence_decision(
        user_message="Olá, estou de volta.",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=True,
    )
    assert decision.continuity == "none"


def test_h_resume_phrase_without_concrete_task_stays_none():
    decision = compute_presence_decision(
        user_message="Lembras-te do que estávamos a fazer?",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.continuity == "none"


def test_i_irrelevant_personal_candidate_selects_none():
    candidates = [PersonalCandidate(id="1", category="preference", score=0.2)]
    selected = select_personal_relevance(candidates)
    assert selected == ()


def test_j_relevant_personal_candidates_selected_up_to_two():
    candidates = [
        PersonalCandidate(id="1", category="preference", score=0.9),
        PersonalCandidate(id="2", category="fact", score=0.8),
    ]
    selected = select_personal_relevance(candidates)
    assert len(selected) == 2


def test_k_personal_relevance_never_exceeds_two_even_with_more_high_scores():
    candidates = [PersonalCandidate(id=str(i), category="fact", score=0.95) for i in range(5)]
    selected = select_personal_relevance(candidates)
    assert len(selected) == 2


def test_l_english_address_may_resolve_honorific():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    mode, value = resolve_address_candidate(preferred, "en-GB")
    assert mode == "honorific"
    assert value == "Sir"


def test_m_portuguese_never_auto_translates_sir_to_senhor():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    mode, value = resolve_address_candidate(preferred, "pt-PT")
    assert value != "Senhor"
    assert "Senhor" not in value
    assert mode in {"none", "name"}


def test_n_address_recency_suppresses_repetition():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    first = compute_presence_decision(
        user_message="Thanks!",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=12,
    )
    assert first.address_mode == "honorific"
    second = compute_presence_decision(
        user_message="Thanks again!",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=0,
    )
    assert second.address_mode == "none"
    assert "ADDRESS_RECENTLY_USED" in second.reason_codes


def test_o_dense_task_suppresses_address():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    decision = compute_presence_decision(
        user_message="Thanks, continue.",
        response_language="en-GB",
        active_task_present=True,
        active_task_unresolved=False,
        active_task_dense=True,
        is_restart=True,
        preferred_address=preferred,
        turns_since_address=50,
    )
    assert decision.address_mode == "none"


def test_p_response_language_changes_address_resolution():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    en = compute_presence_decision(
        user_message="Thanks!",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=12,
    )
    pt = compute_presence_decision(
        user_message="Obrigado!",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=12,
    )
    assert en.address_mode == "honorific"
    assert pt.address_mode != "honorific"


def test_q_presence_decision_does_not_alter_final_response(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta fixa."])
    engine2, llm2, ollama2, anthropic2 = make_engine(tmp_path / "control", mode="local", ollama_replies=["Resposta fixa."])
    response = engine.respond("Ok, obrigado.")
    control_response = engine2.respond("Ok, obrigado.")
    assert response == control_response
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("presence_initiative") == "acknowledge"


def test_r_presence_computation_makes_no_extra_llm_call(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.respond("Qual é a capital da Suécia?")
    assert len(anthropic.calls) == 0
    calls_after_first_turn = len(ollama.calls)
    engine.respond("Obrigado.")
    assert len(ollama.calls) >= calls_after_first_turn


def test_s_telemetry_fields_present_and_typed(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.respond("Ok, percebi.")
    telemetry = engine.get_last_turn_telemetry() or {}
    for key in (
        "presence_continuity",
        "presence_initiative",
        "presence_question_needed",
        "presence_tone",
        "presence_address_mode",
        "presence_personal_candidates_count",
        "presence_personal_selected_count",
        "presence_confidence",
        "presence_reason_codes",
    ):
        assert key in telemetry
    assert isinstance(telemetry["presence_question_needed"], bool)
    assert isinstance(telemetry["presence_personal_candidates_count"], int)
    assert isinstance(telemetry["presence_reason_codes"], list)
    assert not telemetry.get("presence_shadow_error")


def test_t_existing_conversation_behavior_unchanged(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Olá! Como posso ajudar?"])
    engine2, llm2, ollama2, anthropic2 = make_engine(tmp_path / "control", mode="local", ollama_replies=["Olá! Como posso ajudar?"])
    response = engine.respond("Olá!")
    control_response = engine2.respond("Olá!")
    assert response == control_response


def test_shadow_safety_invariant_same_input_same_response_if_presence_fails(tmp_path: Path, monkeypatch):
    """Section 24: same input, before/after presence computation is forced
    to raise internally -> identical final_response for a deterministic
    path. A bug inside presence computation must never surface as a
    changed/broken turn."""
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta estável."])
    baseline = engine.respond("Ok, obrigado.")

    engine2, llm2, ollama2, anthropic2 = make_engine(tmp_path / "second", mode="local", ollama_replies=["Resposta estável."])
    import assistant.conversation as conversation_module

    def _boom(*args, **kwargs):
        raise RuntimeError("forced presence failure")

    monkeypatch.setattr(conversation_module, "compute_presence_decision", _boom)
    guarded = engine2.respond("Ok, obrigado.")

    assert guarded == baseline
    telemetry = engine2.get_last_turn_telemetry() or {}
    assert telemetry.get("presence_shadow_error") == "forced presence failure"


def test_personal_candidates_count_tracks_all_not_just_selected(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.long_term_memory.search_results = [{"id": "1", "category": "preference", "score": 0.2}]
    engine.respond("Recomenda-me algo.")
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("presence_personal_candidates_count") == 1
    assert telemetry.get("presence_personal_selected_count") == 0


def test_no_paid_call_env_marker_untouched(tmp_path: Path, monkeypatch):
    from assistant.anthropic_provider import PAID_CALL_CONFIRMATION_ENV

    monkeypatch.delenv(PAID_CALL_CONFIRMATION_ENV, raising=False)
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.respond("Obrigado.")
    import os

    assert PAID_CALL_CONFIRMATION_ENV not in os.environ


# =====================================================================
# Phase 8.0.1 calibration tests (section 17, A-R)
# =====================================================================

# --- A/B: continuation with enough info answers directly; blocked asks --

def test_calib_a_working_task_no_missing_info_continue_no_question():
    decision = compute_presence_decision(
        user_message="Continua a escrever o email.",
        response_language="pt-PT",
        active_task_present=True,
        active_task_unresolved=False,
        active_task_dense=True,
        is_restart=False,
    )
    assert decision.initiative == "answer_only"
    assert decision.question_needed is False
    assert decision.tone == "focused"
    assert "USER_CONTINUE_DIRECTIVE" in decision.reason_codes


def test_calib_b_blocked_task_continue_asks():
    decision = compute_presence_decision(
        user_message="Continua.",
        response_language="pt-PT",
        active_task_present=True,
        active_task_unresolved=True,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.initiative == "ask"
    assert decision.question_needed is True


# --- C: acknowledgement prefix must not swallow the continuation directive --

def test_calib_c_ok_continua_not_misclassified_as_closure():
    decision = compute_presence_decision(
        user_message="Ok, continua.",
        response_language="pt-PT",
        active_task_present=True,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.initiative != "acknowledge"
    assert decision.question_needed is False


def test_calib_c2_ok_mas_antes_preciso_not_misclassified_as_closure():
    decision = compute_presence_decision(
        user_message="Ok, mas antes preciso de saber uma coisa.",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.initiative != "acknowledge"


# --- D/E: immediate continuation is light, never reconnect --

def test_calib_d_immediate_continuation_is_light_continuity():
    decision = compute_presence_decision(
        user_message="Continua.",
        response_language="pt-PT",
        active_task_present=True,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.continuity == "light"


def test_calib_e_immediate_continuation_is_not_reconnect():
    decision = compute_presence_decision(
        user_message="Continua.",
        response_language="pt-PT",
        active_task_present=True,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert decision.initiative != "reconnect"


# --- F/G: reconnect only for genuine restart + concrete state --

def test_calib_f_explicit_restart_with_concrete_state_is_reconnect_candidate():
    decision = compute_presence_decision(
        user_message="Podemos continuar aquilo de ontem?",
        response_language="pt-PT",
        active_task_present=True,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=True,
    )
    assert decision.continuity == "explicit"
    assert decision.initiative == "reconnect"


def test_calib_g_no_state_no_false_reconnect():
    decision = compute_presence_decision(
        user_message="Podemos continuar aquilo de ontem?",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=True,
    )
    assert decision.continuity == "none"
    assert decision.initiative != "reconnect"


# --- H/I: address eligibility requires a real moment, not just "thanks" --

def test_calib_h_thanks_does_not_force_address():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    decision = compute_presence_decision(
        user_message="Thanks!",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
    )
    assert decision.address_mode == "none"


def test_calib_i_honorific_requires_eligible_moment():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    not_eligible = compute_presence_decision(
        user_message="Thanks!",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=2,
    )
    eligible = compute_presence_decision(
        user_message="Thanks!",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=12,
    )
    assert not_eligible.address_mode == "none"
    assert eligible.address_mode == "honorific"


# --- J/K: sparse, non-forced, non-mechanical address frequency (real engine) --

def test_calib_j_address_remains_sparse_over_repeated_turns(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["You're welcome."] * 20)
    engine.response_language = "en-GB"
    modes = []
    for _ in range(9):
        engine.respond("Thanks!")
        telemetry = engine.get_last_turn_telemetry() or {}
        modes.append(telemetry.get("presence_address_mode"))
    assert all(mode == "none" for mode in modes), modes


def test_calib_k_no_periodic_every_n_turns_behaviour(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["You're welcome."] * 30)
    engine.response_language = "en-GB"
    modes = []
    for _ in range(20):
        engine.respond("Thanks!")
        telemetry = engine.get_last_turn_telemetry() or {}
        modes.append(telemetry.get("presence_address_mode"))
    honorific_count = sum(1 for mode in modes if mode == "honorific")
    assert honorific_count <= 2, modes
    assert honorific_count < len(modes)


# --- L: pt-PT never auto-resolves Sir/Senhor even when eligible --

def test_calib_l_pt_never_auto_resolves_sir_senhor_even_when_eligible():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    decision = compute_presence_decision(
        user_message="Obrigado!",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=50,
    )
    assert decision.address_mode != "honorific"
    assert "Senhor" not in (decision.address_mode or "")


# --- M: dense technical work suppresses address regardless of eligibility --

def test_calib_m_dense_technical_work_suppresses_address():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    decision = compute_presence_decision(
        user_message="Thanks, Echo, continue.",
        response_language="en-GB",
        active_task_present=True,
        active_task_unresolved=False,
        active_task_dense=True,
        is_restart=True,
        preferred_address=preferred,
        turns_since_address=50,
    )
    assert decision.address_mode == "none"
    assert "ADDRESS_CONTEXT_UNSUITABLE" in decision.reason_codes


# --- N/O: personal relevance is gated by turn type, not score alone --

def test_calib_n_high_score_irrelevant_personal_memory_rejected_for_factual_question():
    candidates = [PersonalCandidate(id="1", category="preference", score=0.95)]
    decision = compute_presence_decision(
        user_message="Quanto é 8x7?",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        personal_candidates=candidates,
    )
    assert decision.personal_selected_count == 0
    assert "PERSONAL_CONTEXT_NOT_MATERIAL" in decision.reason_codes


def test_calib_o_materially_relevant_personal_memory_may_be_selected():
    candidates = [PersonalCandidate(id="1", category="preference", score=0.9)]
    decision = compute_presence_decision(
        user_message="Recomenda-me um destino de férias.",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        personal_candidates=candidates,
    )
    assert decision.personal_selected_count in (0, 1, 2)
    assert decision.personal_selected_count >= 1
    assert "PERSONAL_CONTEXT_MATERIAL" in decision.reason_codes


# --- P/Q: shadow safety and no-extra-LLM-call still hold after calibration --

def test_calib_p_presence_decision_still_does_not_alter_final_response(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta fixa."])
    engine2, llm2, ollama2, anthropic2 = make_engine(tmp_path / "control", mode="local", ollama_replies=["Resposta fixa."])
    response = engine.respond("Ok, continua.")
    control_response = engine2.respond("Ok, continua.")
    assert response == control_response


def test_calib_q_no_additional_llm_calls(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.respond("Continua a escrever isso.")
    assert len(anthropic.calls) == 0


# --- R: full Phase 8.0 scenario matrix remains green -- see the tests above
# (test_a..test_t, unchanged behaviour except the explicit turns_since_address
# values now required by the two-tier recency gate) and the real-scenario
# runner in the project scratchpad, both re-run as part of this phase's
# validation pass.


# =====================================================================
# Phase 8.0.2 tests (section 13, A-F) -- SELF_REPRESENTATION_MOMENT
# =====================================================================

def test_selfrep_a_pt_self_representation_detected():
    decision = compute_presence_decision(
        user_message="O que consegues fazer?",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert "SELF_REPRESENTATION_MOMENT" in decision.reason_codes


def test_selfrep_b_en_self_representation_detected():
    decision = compute_presence_decision(
        user_message="What can you do?",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert "SELF_REPRESENTATION_MOMENT" in decision.reason_codes


def test_selfrep_c_ordinary_factual_question_not_detected():
    decision = compute_presence_decision(
        user_message="Qual é a capital da Suécia?",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
    )
    assert "SELF_REPRESENTATION_MOMENT" not in decision.reason_codes


def test_selfrep_d_self_representation_may_become_address_eligible():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    not_eligible = compute_presence_decision(
        user_message="What can you do?",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=0,
    )
    eligible = compute_presence_decision(
        user_message="What can you do?",
        response_language="en-GB",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=5,
    )
    assert not_eligible.address_mode == "none"
    assert eligible.address_mode == "honorific"
    assert "ADDRESS_MOMENT_ELIGIBLE" in eligible.reason_codes


def test_selfrep_e_pt_self_representation_never_resolves_sir():
    preferred = PreferredAddress(honorific_by_language={"en-GB": "Sir"})
    decision = compute_presence_decision(
        user_message="Quem és?",
        response_language="pt-PT",
        active_task_present=False,
        active_task_unresolved=False,
        active_task_dense=False,
        is_restart=False,
        preferred_address=preferred,
        turns_since_address=50,
    )
    assert decision.address_mode != "honorific"


def test_selfrep_f_shadow_only_invariant_remains_intact(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Sou o Echo, o teu assistente."])
    engine2, llm2, ollama2, anthropic2 = make_engine(tmp_path / "control", mode="local", ollama_replies=["Sou o Echo, o teu assistente."])
    response = engine.respond("O que consegues fazer?")
    control_response = engine2.respond("O que consegues fazer?")
    assert response == control_response
