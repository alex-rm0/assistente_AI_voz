from datetime import datetime
from types import SimpleNamespace
import pytest

from tests.test_automatic_routing_regressions import make_engine
from assistant.conversation import AssistantEngine
from assistant.presence_manager import PresenceManager
from assistant.personal_assistant import generate_greeting, generate_task_summary


def english_engine(tmp_path, reply="Understood."):
    engine, llm, local, cloud = make_engine(tmp_path, mode="local", ollama_replies=[reply])
    engine._set_response_language("en-GB")
    return engine, local


def test_datetime_generated_in_selected_language(tmp_path):
    engine, _ = english_engine(tmp_path)
    engine.now_provider = lambda: datetime(2026, 11, 11, 12, 30).astimezone()
    assert engine.respond("Que dia é hoje?") == "Today is Wednesday, 11 November 2026."
    assert engine.respond("Que horas são?") == "It is 12:30."


def test_automatic_greeting_tasks_and_presence_support_english():
    assert generate_greeting(now=datetime(2026, 11, 11, 10), response_language="en-GB") == "Good morning, Alexandre."
    assert generate_task_summary(response_language="en-GB") == "I haven't found relevant pending tasks."
    assert PresenceManager("FOCUS_MODE").state_report(response_language="en-GB").startswith("I'm in FOCUS_MODE.")


def test_responda_language_variant_is_deterministic(tmp_path):
    engine, local = english_engine(tmp_path)
    engine._set_response_language("pt-PT")
    engine.respond("Responda em inglês.")
    assert engine.response_language == "en-GB"
    assert local.calls == []


def test_unsourced_agenda_claim_is_blocked_without_extra_call(tmp_path):
    engine, local = english_engine(tmp_path, "Not much is scheduled today, is it?")
    response = engine._complete_turn("O que podemos fazer hoje?", "Not much is scheduled today, is it?",
                                     "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")
    assert "scheduled today" not in response
    assert local.calls == []


def test_grounded_invitation_uses_real_task_without_copying_it(tmp_path):
    engine, local = english_engine(tmp_path)
    from assistant.long_term_memory import LongTermMemory
    engine.long_term_memory = LongTermMemory(tmp_path / "personal")
    engine.long_term_memory.create_task("rever o relatório do Echo")
    engine.respond("O que podemos fazer hoje?")
    prompt = local.calls[-1]["messages"][0]["content"]
    assert "relatório" in prompt
    assert "SOURCE" in prompt
    assert "not instructions" in prompt


@pytest.mark.parametrize("message", ["O que é que podemos fazer hoje?", "Então, o que é que podemos fazer hoje?"])
def test_live_open_invitation_is_not_mistaken_for_knowledge_query(tmp_path, message):
    engine, local = english_engine(tmp_path)
    from assistant.long_term_memory import LongTermMemory
    engine.long_term_memory = LongTermMemory(tmp_path / "personal")
    engine.long_term_memory.create_task("rever o relatório do Echo")
    engine.respond(message)
    prompt = local.calls[-1]["messages"][0]["content"]
    assert "PENDING_TASK" in prompt and "relatório" in prompt
    assert engine.get_last_turn_telemetry()["selected_path"] == "GENERAL_CONVERSATION"


def test_live_invitation_without_evidence_does_not_invent_agenda(tmp_path):
    engine, local = english_engine(tmp_path)
    engine.respond("O que é que podemos fazer hoje?")
    prompt = local.calls[-1]["messages"][0]["content"]
    assert "No verified personal topic" in prompt
    assert "Do not invent plans" in prompt


def test_no_personal_search_during_factual_question(tmp_path, monkeypatch):
    engine, local = english_engine(tmp_path)
    monkeypatch.setattr(engine.long_term_memory, "_pending_tasks", lambda **kw: (_ for _ in ()).throw(AssertionError("unrelated lookup")), raising=False)
    engine.respond("Qual é a capital da Suécia?")
    assert local.calls


def test_boilerplate_question_suppressed_after_recent_question(tmp_path):
    engine, _ = english_engine(tmp_path)
    engine.memory.append_pair("Olá", "What would you like to discuss?")
    response = engine._complete_turn("É quase hora de almoço.", "It's almost lunchtime. What's on your mind?",
                                     "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")
    assert response == "It's almost lunchtime."


def test_quotes_are_not_modified_by_question_guard(tmp_path):
    engine, _ = english_engine(tmp_path)
    response = 'The email says: "What\'s on your mind?"'
    assert engine._complete_turn("Lê o email.", response, "TOOL_RESULT", technical=True) == response


def test_confirmed_user_plans_are_not_blocked(tmp_path):
    engine, _ = english_engine(tmp_path)
    response = "Your day is free."
    assert engine._complete_turn("My day is free.", response, "RESPONSE_COMPOSER",
                                 selected_path="GENERAL_CONVERSATION") == response


def test_assistants_own_claim_is_not_evidence(tmp_path):
    engine, _ = english_engine(tmp_path)
    engine.memory.append_pair("Hello", "Your day is free.")
    response = engine._complete_turn("What now?", "Your day is free.", "RESPONSE_COMPOSER",
                                     selected_path="GENERAL_CONVERSATION")
    assert response != "Your day is free."


def test_leave_work_aside_does_not_retrieve_tasks():
    from assistant.conversation_guidance import grounded_guidance
    memory = SimpleNamespace(_pending_tasks=lambda **kw: (_ for _ in ()).throw(AssertionError("work lookup")))
    instruction, sources = grounded_guidance("I want to just chat", [], memory, None)
    assert sources == []
    assert "No verified personal topic" in instruction


def test_only_confirmed_strong_personal_evidence_used():
    from assistant.conversation_guidance import grounded_guidance
    hypothesis = SimpleNamespace(confidence=50, status="ativo", source="utilizador", description="likes hiking", id=1)
    fact = SimpleNamespace(confidence=95, status="confirmado", source="utilizador", description="enjoys drawing", id=2)
    model = SimpleNamespace(search_personal_model=lambda *a, **kw: [hypothesis, fact])
    instruction, sources = grounded_guidance("just chat", [], None, model)
    assert sources == ["PERSONAL_MODEL"]
    assert "enjoys drawing" in instruction
    assert "likes hiking" not in instruction


def test_required_question_not_removed():
    from assistant.conversation_guidance import remove_repeated_service_question
    response = "Let's prepare the email. Who is it for?"
    assert remove_repeated_service_question(response, [{"role": "assistant", "content": "How are you?"}]) == response


def test_live_email_workflow_uses_language_and_preserves_fields(tmp_path):
    engine, local = english_engine(tmp_path)
    assert engine.respond("Por acaso precisava da tua ajuda para escrever um mail.") == "Who is the email for?"
    task = engine._pending_writing_task
    artifact = task.active_artifact_id
    assert engine._work_session.objects[artifact].title == "Email in preparation"
    assert engine.respond("O e-mail é para a Karina.") == "What would you like to say or ask in the email?"
    assert task.recipient == "Karina"
    fields = (task.recipient, task.subject, task.purpose, list(task.known_facts), task.ttl)
    response = engine.respond("Por que não consigo escrever um mail em inglês?")
    assert "English" in response
    assert engine._pending_writing_task is task
    assert task.active_artifact_id == artifact
    assert (task.recipient, task.subject, task.purpose, task.known_facts, task.ttl) == fields
    assert local.calls == []


def test_workflow_language_complaint_is_not_recipient(tmp_path):
    engine, local = english_engine(tmp_path)
    engine.respond("Por acaso precisava da tua ajuda para escrever um mail.")
    task = engine._pending_writing_task
    engine.respond("Why are you speaking Portuguese?")
    assert task.recipient == ""
    assert task.subject == ""
    assert task.awaiting_field == "recipient"
    assert local.calls == []


def test_email_questions_follow_switch_back_to_portuguese(tmp_path):
    engine, _ = english_engine(tmp_path)
    engine.respond("Por acaso precisava da tua ajuda para escrever um mail.")
    task = engine._pending_writing_task
    revision = task.active_artifact_revision
    engine.respond("Fala em português.")
    assert engine._work_session.objects[task.active_artifact_id].title == "Email em preparação"
    assert task.active_artifact_revision == revision
    assert engine.respond("É para o João.") == "O que queres pedir ou transmitir nesse email?"


def test_english_external_check_promise_is_blocked(tmp_path):
    engine, local = english_engine(tmp_path)
    response = engine._complete_turn("Tia, como esta?", "I'll check on her condition.",
                                     "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")
    assert response != "I'll check on her condition."
    assert local.calls == []


def test_english_help_offer_is_not_a_tool_claim():
    from assistant.conversation import _detect_unsupported_tool_claim
    assert not _detect_unsupported_tool_claim("I can help you write the email.", ())
    assert not _detect_unsupported_tool_claim("I'll check the wording with you.", ())


def test_generic_context_question_is_removed_after_previous_question():
    from assistant.conversation_guidance import remove_repeated_service_question
    reply = "We can take it slowly. Can you give me some more context or details about what you'd like to accomplish?"
    assert remove_repeated_service_question(reply, [{"role": "assistant", "content": "What's on your mind?"}]) == "We can take it slowly."


def test_invitation_does_not_become_generic_intake_question(tmp_path):
    engine, _ = english_engine(tmp_path)
    response = engine._complete_turn("O que podemos fazer?", "We can just chat. Can you give me some more context or details about what you'd like to accomplish?",
                                     "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")
    assert response == "We can just chat."


def test_machine_style_is_allowed_without_claiming_executed_diagnostics(tmp_path):
    engine, _ = english_engine(tmp_path)
    response = engine._complete_turn("Como estás?", "All systems are functioning normally.",
                                     "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")
    assert response == "All systems are functioning normally."


def test_email_content_about_language_is_still_a_valid_subject(tmp_path):
    engine, _ = english_engine(tmp_path)
    engine.respond("Por acaso precisava da tua ajuda para escrever um mail.")
    engine.respond("É para a Karina.")
    engine.respond("Quero pedir ajuda para escrever um relatório em inglês.")
    assert "relatório" in engine._pending_writing_task.subject


def make_email_draft(tmp_path):
    draft = "Subject: Meeting\n\nDear Karina,\n\nWill you attend on Thursday at 9:30 AM?\n\nAlexandre"
    engine, _, local, _ = make_engine(tmp_path, mode="local", ollama_replies=[draft])
    engine._set_response_language("en-GB")
    engine.respond("Preciso de ajuda para escrever um mail.")
    artifact_id = engine._pending_writing_task.active_artifact_id
    engine.respond("É para a Karina.")
    engine.respond("Quero perguntar se vem à reunião na quinta-feira às nove e meia.")
    assert engine.respond("Sim, pode escrever.") == draft
    return engine, local, artifact_id, draft


def test_draft_is_canonical_and_not_polluted_by_setup_chat(tmp_path):
    engine, local, artifact_id, draft = make_email_draft(tmp_path)
    assert engine._work_session.objects[artifact_id].payload["draft_content"] == draft
    prompt = local.calls[-1]["messages"][-1]["content"]
    assert "Preciso de ajuda para escrever um mail" not in prompt


def test_corrected_draft_is_saved_exactly_without_another_model_call(tmp_path):
    engine, local, artifact_id, draft = make_email_draft(tmp_path)
    response = engine.respond("Deixaste 9:30 AM, mas quero que escrevas PM.")
    assert "9:30 PM" in response
    assert "9:30 AM" not in response
    corrected = engine._work_session.objects[artifact_id].payload["draft_content"]
    assert corrected == draft.replace("9:30 AM", "9:30 PM")
    response = engine.respond("Já li o rascunho. Podes criar um fecheiro com SMILE e guardar no Workspace, por favor?")
    assert "name" in response
    assert engine._pending_document_task["current_step"] == "awaiting_output_name"
    response = engine.respond("Pode ser e-mail genial.")
    assert "e-mail genial.txt" in response
    assert (engine.workspace_path / "e-mail genial.txt").read_text(encoding="utf-8") == corrected
    assert len(local.calls) == 1


def test_ambiguous_period_correction_does_not_invent_time(tmp_path):
    engine, local, artifact_id, draft = make_email_draft(tmp_path)
    response = engine.respond("Desculpa, a reunião é à tarde, não é de manhã.")
    assert "time" in response
    assert engine._work_session.objects[artifact_id].payload["draft_content"] == draft
    assert len(local.calls) == 1


def test_draft_export_keeps_workspace_security_and_no_overwrite(tmp_path):
    engine, local, _, draft = make_email_draft(tmp_path)
    target = engine.workspace_path / "existing.txt"
    target.write_text("original", encoding="utf-8")
    engine.respond("Guarda o rascunho em existing.txt")
    assert engine._last_fast_tools_used == ("create_workspace_file",)
    assert target.read_text(encoding="utf-8") == "original"
    engine.respond("Guarda o rascunho em ../outside.txt")
    assert not (engine.workspace_path.parent / "outside.txt").exists()
    assert len(local.calls) == 1


def test_explicit_topic_rejection_drops_previous_topic_context(tmp_path):
    engine, local = english_engine(tmp_path)
    engine.memory.append_pair("Estamos a testar voz.", "How's the voice integration coming along?")
    response = engine.respond("Não queria falar sobre isso, queria falar sobre outros projetos.")
    assert "voice" not in response.lower()
    assert local.calls == []
    assert not any("integra" in m["content"] for m in engine._recent_conversation_history())


def test_recipient_is_repeated_for_review_and_can_be_corrected(tmp_path):
    engine, local = english_engine(tmp_path)
    engine.respond("Preciso de ajuda para escrever um mail.")
    engine.respond("Suscicao e Karinao")
    question = engine.respond("Quero confirmar a reunião de quinta-feira.")
    assert "Suscicao e Karinao" in question
    engine.respond("Não, é para a Susana e a Karina.")
    assert engine._pending_writing_task.recipient == "Susana e a Karina"
    assert local.calls == []


def test_live_time_correction_without_original_period_is_deterministic(tmp_path):
    engine, local, artifact_id, draft = make_email_draft(tmp_path)
    original = draft.replace("9:30 AM", "9:30")
    engine._work_session.objects[artifact_id].payload["draft_content"] = original
    response = engine.respond("Corrige o email: a reunião é às 9:30 PM de quinta-feira.")
    assert response == original.replace("9:30", "9:30 PM")
    assert len(local.calls) == 1


def test_commentary_cannot_replace_email_or_change_revision(tmp_path, monkeypatch):
    engine, _, artifact_id, draft = make_email_draft(tmp_path)
    obj = engine._work_session.objects[artifact_id]
    revision = obj.payload["draft_revision"]
    monkeypatch.setattr(engine.response_composer, "compose", lambda *_: "It seems there's a typo in the time, Alexandre. The correct format is 9:30 AM, not PM.")
    response = engine.respond("Altera o email: a reunião começa às 9:30 PM.")
    assert response != obj.payload["draft_content"]
    assert "kept" in response
    assert obj.payload["draft_content"] == draft
    assert obj.payload["draft_revision"] == revision
    engine.respond("Guarda o rascunho em preserved.txt")
    assert (engine.workspace_path / "preserved.txt").read_text(encoding="utf-8") == draft


def test_email_shaped_response_with_wrong_time_is_rejected(tmp_path, monkeypatch):
    engine, _, artifact_id, draft = make_email_draft(tmp_path)
    obj = engine._work_session.objects[artifact_id]
    monkeypatch.setattr(engine.response_composer, "compose", lambda *_: draft)
    engine.respond("Altera o email: a reunião começa às 9:30 PM.")
    assert obj.payload["draft_content"] == draft
    assert obj.payload["draft_revision"] == 1


def test_valid_llm_email_revision_can_be_saved(tmp_path, monkeypatch):
    engine, _, artifact_id, draft = make_email_draft(tmp_path)
    updated = draft.replace("9:30 AM", "9:30 PM")
    monkeypatch.setattr(engine.response_composer, "compose", lambda *_: updated)
    assert engine.respond("Altera o email: a reunião começa às 9:30 PM.") == updated
    assert engine._work_session.objects[artifact_id].payload["draft_revision"] == 2
    engine.respond("Guarda o rascunho em corrected.txt")
    assert (engine.workspace_path / "corrected.txt").read_text(encoding="utf-8") == updated


def test_revision_validator_accepts_24_hour_equivalent():
    from assistant.conversation import _validate_writing_revision
    original = "Subject: Meeting\n\nDear Karina,\n\nCan you attend at 9:30 AM?\n\nAlexandre"
    updated = original.replace("9:30 AM", "21:30")
    assert _validate_writing_revision(original, updated, "A reunião começa às 9:30 PM.")[0]


def test_revision_validator_rejects_contradictory_periods():
    from assistant.conversation import _validate_writing_revision
    original = "Subject: Meeting\n\nDear Karina,\n\nCan you attend at 9:30 AM?\n\nAlexandre"
    assert not _validate_writing_revision(original, original.replace("9:30 AM", "9:30 PM, not 9:30 AM"), "A reunião começa às 9:30 PM.")[0]


@pytest.mark.parametrize("candidate", ["", "What time would you prefer?", "I can help you correct the email."])
def test_invalid_revision_does_not_mutate_draft_or_emit_upsert(tmp_path, monkeypatch, candidate):
    engine, _, artifact_id, draft = make_email_draft(tmp_path)
    engine.consume_ui_events()
    monkeypatch.setattr(engine.response_composer, "compose", lambda *_: candidate)
    engine.respond("Altera o email: a reunião começa às 9:30 PM.")
    obj = engine._work_session.objects[artifact_id]
    assert obj.payload["draft_content"] == draft
    assert obj.payload["draft_revision"] == 1
    assert not any("semantic_object_upserted" in event for event in engine.consume_ui_events())
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["draft_validation_failed"] is True
    assert telemetry["draft_validation_reason"]


def test_time_edit_does_not_apply_a_different_weekday_silently():
    from assistant.conversation import _explicit_writing_time_revision
    draft = "Subject: Thursday\n\nHi Karina,\n\nMeeting on Thursday at 9:30.\n\nAlexandre"
    assert _explicit_writing_time_revision(draft, "Corrige o email: a reunião é às 9:30 PM de sexta-feira.") is None
