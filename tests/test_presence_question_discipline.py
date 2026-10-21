"""Phase 8.1 (Presence Question Discipline Activation) tests.

The FIRST activated Presence behavior: when PresenceDecision.question_needed
is False on an eligible conversational composition path (SOCIAL_PATH,
GENERAL_CONVERSATION, GENERAL_KNOWLEDGE_QUERY), the composer's SYSTEM
PROMPT gets a compact "don't add an optional follow-up" instruction --
never post-processing of the generated text. Deterministic/task-required
questions never touch this code path at all.

Real-engine tests use tests.test_automatic_routing_regressions.make_engine
(mode="local", so no Anthropic call is ever possible). Because FakeProvider
returns a fixed canned reply regardless of prompt content, the SYSTEM
PROMPT actually sent is inspected directly via ollama.calls[-1]["messages"]
to confirm the instruction is (or isn't) present -- the final response text
is used only to confirm no corruption/regression.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import assistant.conversation as conversation_module
from assistant.conversation import ActiveTaskContext
from assistant.response_composer import ComposerRequest, ResponseComposer
from tests.test_automatic_routing_regressions import make_engine

DISCIPLINE_MARKER_PT = "Não acrescentes uma pergunta de seguimento"
DISCIPLINE_MARKER_EN = "Do not add a follow-up question"


def _last_system_prompt(provider) -> str:
    return provider.calls[-1]["messages"][0]["content"]


# --- A/B: acknowledgement/thanks suppress the optional follow-up --------

def test_a_acknowledgement_suppresses_optional_followup(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.respond("Ok, percebi, obrigado.")
    assert DISCIPLINE_MARKER_PT in _last_system_prompt(ollama)
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("presence_question_discipline_applied") is True


def test_b_thanks_suppresses_optional_followup(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.respond("Muito obrigado pela ajuda.")
    assert DISCIPLINE_MARKER_PT in _last_system_prompt(ollama)
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("presence_question_discipline_applied") is True


# --- C: direct factual answer suppresses the optional follow-up ---------

def test_c_factual_answer_suppresses_optional_followup(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.respond("Qual é a capital da Suécia?")
    assert DISCIPLINE_MARKER_PT in _last_system_prompt(ollama)
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("presence_question_discipline_applied") is True
    assert telemetry.get("selected_path") == "GENERAL_CONVERSATION"


# --- D: unblocked continuation suppresses the optional follow-up --------
# (this path is deterministic and never touches ResponseComposer at all --
# confirmed by asserting the response is identical whether the feature
# flag is on or off, i.e. nothing here could possibly regress it)

def test_d_unblocked_continuation_unaffected_by_flag(tmp_path: Path, monkeypatch):
    def build_and_run(flag: bool, path: Path):
        monkeypatch.setattr(conversation_module, "PRESENCE_QUESTION_DISCIPLINE_ENABLED", flag)
        engine, llm, ollama, anthropic = make_engine(path, mode="local", ollama_replies=["Resposta."])
        engine._active_task_context = ActiveTaskContext(
            id="t1", task_type="compose_email", status="working",
            entities={"recipient": "Joao"}, parameters={"purpose": "x", "topic": "y"},
            missing_information=[],
        )
        response = engine.respond("Continua.")
        telemetry = engine.get_last_turn_telemetry() or {}
        return response, telemetry

    response_on, telemetry_on = build_and_run(True, tmp_path / "on")
    response_off, telemetry_off = build_and_run(False, tmp_path / "off")
    assert response_on == response_off
    assert telemetry_on.get("selected_path") == telemetry_off.get("selected_path")
    assert telemetry_on.get("presence_question_discipline_applied") is None
    assert telemetry_off.get("presence_question_discipline_applied") is None


# --- E/F: required/genuinely-ambiguous questions survive identically ----

def test_e_blocked_task_required_question_survives(tmp_path: Path, monkeypatch):
    def build_and_run(flag: bool, path: Path):
        monkeypatch.setattr(conversation_module, "PRESENCE_QUESTION_DISCIPLINE_ENABLED", flag)
        engine, llm, ollama, anthropic = make_engine(path, mode="local", ollama_replies=["Resposta."])
        engine._active_task_context = ActiveTaskContext(
            id="t1", task_type="compose_email", status="collecting",
            entities={}, parameters={}, missing_information=["recipient"],
        )
        response = engine.respond("Continua.")
        return response

    response_on = build_and_run(True, tmp_path / "on")
    response_off = build_and_run(False, tmp_path / "off")
    assert response_on == response_off
    assert "?" in response_on


def test_f_genuine_ambiguity_retains_required_question(tmp_path: Path, monkeypatch):
    def build_and_run(flag: bool, path: Path):
        monkeypatch.setattr(conversation_module, "PRESENCE_QUESTION_DISCIPLINE_ENABLED", flag)
        engine, llm, ollama, anthropic = make_engine(path, mode="local", ollama_replies=["Resposta."])
        response = engine.respond("Planeia-me uma viagem.")
        telemetry = engine.get_last_turn_telemetry() or {}
        return response, telemetry

    response_on, telemetry_on = build_and_run(True, tmp_path / "on")
    response_off, telemetry_off = build_and_run(False, tmp_path / "off")
    assert response_on == response_off
    assert telemetry_on.get("selected_path") == "CLARIFYING_QUESTION"
    assert telemetry_on.get("presence_question_discipline_applied") is None
    assert telemetry_off.get("presence_question_discipline_applied") is None


# --- G/H: pt-PT and en-GB both work through the same localized instruction --

def test_g_pt_pt_behavior(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.respond("Sim, já percebi.")
    assert DISCIPLINE_MARKER_PT in _last_system_prompt(ollama)


def test_h_en_gb_behavior(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.response_language = "en-GB"
    engine.respond("Thanks, got it!")
    assert DISCIPLINE_MARKER_EN in _last_system_prompt(ollama)


# --- I: feature flag OFF preserves previous (pre-8.1) behavior ----------

def test_i_feature_flag_off_preserves_previous_behavior(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(conversation_module, "PRESENCE_QUESTION_DISCIPLINE_ENABLED", False)
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    response = engine.respond("Ok, percebi, obrigado.")
    system_prompt = _last_system_prompt(ollama)
    assert DISCIPLINE_MARKER_PT not in system_prompt
    assert response == "Resposta."
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("presence_question_discipline_applied") is None


# --- J: Presence computation failure preserves previous behavior --------

def test_j_presence_failure_preserves_previous_behavior(tmp_path: Path, monkeypatch):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])

    def _boom(*args, **kwargs):
        raise RuntimeError("forced presence failure")

    monkeypatch.setattr(conversation_module, "compute_presence_decision", _boom)
    response = engine.respond("Ok, percebi, obrigado.")
    assert response == "Resposta."
    system_prompt = _last_system_prompt(ollama)
    assert DISCIPLINE_MARKER_PT not in system_prompt
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry.get("presence_activation_error") == "forced presence failure"


# --- K/L: no routing or task-state change from activation ---------------

def test_k_no_routing_change(tmp_path: Path, monkeypatch):
    def build_and_run(flag: bool, path: Path):
        monkeypatch.setattr(conversation_module, "PRESENCE_QUESTION_DISCIPLINE_ENABLED", flag)
        engine, llm, ollama, anthropic = make_engine(path, mode="local", ollama_replies=["Resposta."])
        engine.respond("Qual é a capital da Suécia?")
        telemetry = engine.get_last_turn_telemetry() or {}
        return telemetry

    telemetry_on = build_and_run(True, tmp_path / "on")
    telemetry_off = build_and_run(False, tmp_path / "off")
    assert telemetry_on.get("selected_path") == telemetry_off.get("selected_path")
    assert telemetry_on.get("response_source") == telemetry_off.get("response_source")


def test_l_no_task_state_change(tmp_path: Path, monkeypatch):
    def build_and_run(flag: bool, path: Path):
        monkeypatch.setattr(conversation_module, "PRESENCE_QUESTION_DISCIPLINE_ENABLED", flag)
        engine, llm, ollama, anthropic = make_engine(path, mode="local", ollama_replies=["Resposta."])
        engine._active_task_context = ActiveTaskContext(
            id="t1", task_type="compose_email", status="collecting",
            entities={}, parameters={}, missing_information=["recipient"],
        )
        engine.respond("Continua.")
        task = engine._active_task_context
        return (task.status, list(task.missing_information)) if task is not None else None

    state_on = build_and_run(True, tmp_path / "on")
    state_off = build_and_run(False, tmp_path / "off")
    assert state_on == state_off


# --- M/N: no extra LLM call, no paid call --------------------------------

def test_m_no_extra_llm_call(tmp_path: Path, monkeypatch):
    def build_and_run(flag: bool, path: Path):
        monkeypatch.setattr(conversation_module, "PRESENCE_QUESTION_DISCIPLINE_ENABLED", flag)
        engine, llm, ollama, anthropic = make_engine(path, mode="local", ollama_replies=["Resposta."])
        engine.respond("Ok, percebi, obrigado.")
        return len(ollama.calls)

    calls_on = build_and_run(True, tmp_path / "on")
    calls_off = build_and_run(False, tmp_path / "off")
    assert calls_on == calls_off == 1


def test_n_no_paid_call(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta."])
    engine.respond("Ok, percebi, obrigado.")
    engine.respond("Qual é a capital da Suécia?")
    assert anthropic.calls == []


# --- O: final response remains valid text --------------------------------

def test_o_final_response_remains_valid_text(tmp_path: Path):
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta válida."])
    response = engine.respond("Ok, percebi, obrigado.")
    assert isinstance(response, str)
    assert response.strip() == "Resposta válida."


# --- P: quoted/user-content questions are never stripped -----------------
# (direct unit test on ResponseComposer.compose -- proves there is no post-
# processing step at all, regardless of question_discipline_instruction)

class _StubChatModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = []

    def chat(self, user_message, history=None, system_prompt=None, response_format=None,
              temperature=None, num_predict=None, timeout_seconds=None, task_profile=None):
        self.calls.append({"user_message": user_message, "system_prompt": system_prompt})
        return self.reply


def test_p_quoted_question_not_stripped_by_composer():
    quoted_reply = 'O cliente perguntou: "Isto já está pronto?" e eu respondi que sim.'
    stub = _StubChatModel(quoted_reply)
    composer = ResponseComposer(stub)
    request = ComposerRequest(
        intent="general_knowledge_query",
        user_message="Resume o que o cliente disse.",
        response_language="pt-PT",
        question_discipline_instruction="Não acrescentes uma pergunta de seguimento só para manter a conversa.",
    )
    result = composer.compose(request)
    assert result == quoted_reply
    assert "?" in result


def test_p2_question_discipline_instruction_reaches_system_prompt_only():
    stub = _StubChatModel("Resposta qualquer.")
    composer = ResponseComposer(stub)
    request = ComposerRequest(
        intent="general_knowledge_query",
        user_message="Pergunta qualquer com conteúdo?",
        response_language="pt-PT",
        question_discipline_instruction="Não acrescentes uma pergunta de seguimento só para manter a conversa.",
    )
    composer.compose(request)
    assert "Não acrescentes uma pergunta de seguimento" in stub.calls[-1]["system_prompt"]
    assert "Não acrescentes uma pergunta de seguimento" not in stub.calls[-1]["user_message"]
