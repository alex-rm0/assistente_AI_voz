import pytest

from tests.test_live_writing_regressions import english_engine, collected_email


@pytest.mark.parametrize("language,reply", [
    ("pt-PT", "Sim, enviei."), ("pt-PT", "Sim, já o enviei."),
    ("pt-PT", "O email já foi enviado ao professor."),
    ("en-GB", "Yes, I sent it."), ("en-GB", "I've sent the email."),
    ("en-GB", "The email has been sent."),
])
def test_email_sending_cannot_be_claimed_without_execution(tmp_path, language, reply):
    engine, local = english_engine(tmp_path)
    engine._set_response_language(language)
    response = engine._complete_turn("Já enviaste o email ao professor?", reply,
                                    "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")
    assert response != reply
    assert engine._last_response_blocked
    assert "draft" in response if language == "en-GB" else "rascunho" in response
    assert not local.calls


@pytest.mark.parametrize("reply", [
    "Tu enviaste-me o ficheiro ontem.", "You sent me the file yesterday.",
    "I can help you write the email.", "I haven't sent the email.",
    "Não enviei o email.", 'You wrote: "I sent the email".',
])
def test_non_execution_send_mentions_are_preserved(tmp_path, reply):
    engine, _ = english_engine(tmp_path)
    assert engine._complete_turn("Explica isso.", reply, "RESPONSE_COMPOSER",
                                 selected_path="GENERAL_CONVERSATION") == reply


DRAFT = "Subject: Meeting\n\nHi Francisca and Karina,\n\nCan you attend this week at 21:30?\n\nAlexandre"


@pytest.mark.parametrize("generated", [False, True])
def test_recipient_letter_change_requires_real_correction(tmp_path, generated):
    engine, local = english_engine(tmp_path, [DRAFT])
    artifact = collected_email(engine)
    if generated:
        engine.respond("Podes escrever o primeiro rascunho.")
    calls = len(local.calls)
    response = engine.respond("Consegues trocar o nome da Karina, o K por um C?")
    assert "Carina" in response
    assert "Karina" in engine._work_session.objects[artifact].payload["recipient"]
    assert engine.get_last_turn_telemetry()["selected_path"] == "WRITING_TASK"
    engine.respond("Sim.")
    obj = engine._work_session.objects[artifact]
    assert obj.payload["recipient"] == "Francisca e Carina"
    if generated:
        assert "Carina" in obj.payload["draft_content"] and "Karina" not in obj.payload["draft_content"]
    assert len(local.calls) == calls


@pytest.mark.parametrize("answer,name", [("É Carina.", "Carina"), ("É para a Carina.", "Carina"), ("O nome é João.", "João")])
def test_wrong_name_in_generation_request_is_not_ignored(tmp_path, answer, name):
    engine, local = english_engine(tmp_path, [DRAFT])
    artifact = collected_email(engine)
    response = engine.respond("Podes escrever o primeiro rascunho, mas o nome da Karina está mal")
    assert "correct name" in response.lower()
    assert not local.calls
    assert not engine._work_session.objects[artifact].payload.get("draft_content")
    engine.respond(answer)
    assert engine._work_session.objects[artifact].payload["recipient"] == "Francisca e " + name


def test_unspecified_recipient_correction_asks_for_the_actual_name(tmp_path):
    engine, local = english_engine(tmp_path)
    artifact = collected_email(engine)
    response = engine.respond("Consegues corrigir o nome da Karina?")
    assert "correct name" in response.lower()
    assert not local.calls
    assert engine._work_session.objects[artifact].payload["recipient"] == "Francisca e Karina"


@pytest.mark.parametrize("reply", ["Here's the corrected draft:", "I've updated the draft.", "Já corrigi o rascunho."])
@pytest.mark.parametrize("tools", [(), ("get_active_window",)])
def test_unexecuted_draft_edit_is_blocked(tmp_path, reply, tools):
    engine, _ = english_engine(tmp_path)
    assert engine._complete_turn("Troca o nome.", reply, "RESPONSE_COMPOSER",
                                 selected_path="GENERAL_CONVERSATION", tools_used=tools) != reply
    assert engine._last_response_blocked


@pytest.mark.parametrize("generated", [False, True])
def test_natural_resume_keeps_the_email_in_work_session(tmp_path, generated):
    engine, local = english_engine(tmp_path, [DRAFT])
    artifact = collected_email(engine)
    if generated:
        engine.respond("Podes escrever o primeiro rascunho.")
    before = dict(engine._work_session.objects[artifact].payload)
    count = len(local.calls)
    response = engine.respond("Vamos continuar aqui a tarefa do mail, por favor.")
    assert "no email" not in response.lower()
    assert engine.get_last_turn_telemetry()["selected_path"] == "TASK_RESUME"
    assert engine._work_session.objects[artifact].payload == before
    assert len(local.calls) == count


def test_resume_paused_canonical_email_uses_same_task(tmp_path):
    from assistant.conversation import ActiveTaskContext
    engine, local = english_engine(tmp_path)
    email = ActiveTaskContext(id="mail-original", task_type="compose_email", status="paused")
    engine._paused_task_stack.append(email)
    engine._update_task_artifact_ui(email)
    engine.respond("Vamos continuar aqui a tarefa do mail, por favor.")
    assert engine._work_session.active_task is email
    assert not local.calls
