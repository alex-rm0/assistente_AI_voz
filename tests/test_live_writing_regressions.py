from __future__ import annotations

import pytest

from tests.test_automatic_routing_regressions import make_engine


def english_engine(tmp_path, replies=None):
    engine, _, local, _ = make_engine(tmp_path, mode="local", ollama_replies=replies)
    engine._set_response_language("en-GB")
    return engine, local


@pytest.mark.parametrize("message,reply", [
    ("Agora preciso que guardes isso num fecheiro, num workspace, por favor.", "I've saved the note in a workspace."),
    ("Qual é o nome do fecheiro?", 'It\'s saved in a workspace named "Reunião da Semana".'),
    ("Porquê que aparece ainda a task no ecrã?", "The task was saved, but it's still visible because it's been added to the current day's schedule."),
    ("Então acrescenta mais um e-mail para ver como fica no ecrã.", "I've added the new email to the workspace."),
])
def test_unexecuted_write_claim_is_not_shown(tmp_path, message, reply):
    engine, local = english_engine(tmp_path)
    response = engine._complete_turn(message, reply, "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")
    assert response != reply
    assert engine._last_response_blocked is True
    assert local.calls == []


@pytest.mark.parametrize("reply", [
    "I can help you save the email.",
    "You said you saved the file yesterday.",
    'The example says: "I have saved the note in a workspace."',
])
def test_nonexecuting_or_attributed_text_is_not_a_false_action(tmp_path, reply):
    engine, _ = english_engine(tmp_path)
    assert engine._complete_turn("Explica isso.", reply, "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION") == reply


def test_email_body_is_not_mistaken_for_assistant_execution(tmp_path):
    engine, _ = english_engine(tmp_path)
    email = "Subject: Update\n\nHi Ana,\n\nI've saved the note in a workspace.\n\nAlexandre"
    assert engine._complete_turn("Escreve o email.", email, "WRITING_WORKFLOW", technical=True, selected_path="WRITING_TASK") == email


def collected_email(engine):
    engine.respond("Por acaso preciso ajudar a escrever um mail?")
    artifact = engine._pending_writing_task.active_artifact_id
    engine.respond("Francisca e Karina")
    engine.respond("Queria perguntar se podem vir à reunião desta semana que é às 9h30 da noite.")
    return artifact


@pytest.mark.parametrize("message", [
    "Pois é, preciso que corrijas umas coisas no email. Consegues abrir a tarefa?",
    "Quero pegar na tarefa do mail. Abro uma tarefa com o mail.",
    "Vamos continuar aqui a tarefa do mail, por favor.",
])
def test_live_email_reference_resumes_same_workflow_without_tools(tmp_path, message):
    engine, local = english_engine(tmp_path)
    artifact = collected_email(engine)
    before = engine._pending_writing_task.recipient
    response = engine.respond(message)
    assert engine._pending_writing_task.active_artifact_id == artifact
    assert engine._pending_writing_task.recipient == before
    assert "gmail" not in response.lower()
    assert engine._work_session.attention.focus_object_id == artifact
    assert not engine.agent.has_pending_confirmation()
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0
    assert local.calls == []


@pytest.mark.parametrize("message", [
    "Consegues ler o mail, por favor?", "Consejo me ler o mail, por favor?",
    "Read the email draft, please.",
])
def test_live_read_email_uses_canonical_draft_not_filename(tmp_path, message):
    draft = "Subject: Meeting\n\nDear Francisca and Karina,\n\nCan you attend this week at 21:30?\n\nAlexandre"
    engine, local = english_engine(tmp_path, [draft])
    artifact = collected_email(engine)
    engine.respond("Guarda isso num ficheiro.")
    calls_before = len(local.calls)
    response = engine.respond(message)
    assert draft in response
    assert engine._work_session.attention.focus_object_id == artifact
    assert engine._work_session.objects[artifact].payload["draft_content"] == draft
    assert "workspace_search" not in engine.get_last_turn_telemetry()["tools_used"]
    assert len(local.calls) == calls_before


@pytest.mark.parametrize("message", [
    "abre o Gmail", "abre o mail no browser", "lê o ficheiro email_resumo_novo.txt",
    "não abras a tarefa do email", "Don't read the email draft",
])
def test_email_reference_does_not_capture_apps_files_or_negation(message):
    from assistant.writing_continuity import is_email_resume_request, is_email_read_request
    assert not is_email_resume_request(message)
    assert not is_email_read_request(message)


@pytest.mark.parametrize("resume_first", [False, True])
def test_saved_email_survives_trip_and_can_resume_and_read_without_model(tmp_path, resume_first):
    draft = "Subject: Meeting\n\nDear Francisca and Karina,\n\nCan you attend this week at 21:30?\n\nAlexandre"
    engine, local = english_engine(tmp_path, [draft])
    artifact = collected_email(engine)
    engine.respond("Guarda isso num ficheiro.")
    saved = engine._work_session.objects[artifact].payload["saved_file"]
    engine.respond("Vamos também preparar umas férias em Estocolmo.")
    trip = engine._work_session.active_task
    trip_fields = (dict(trip.entities), dict(trip.parameters), list(trip.collected_facts))
    calls = len(local.calls)
    if resume_first:
        response = engine.respond("volta ao email")
        assert engine.get_last_turn_telemetry()["llm_calls"] == 0
        assert engine._work_session.attention.focus_object_id == artifact
        assert "trip" not in response.lower()
    for message in ("Lê o mail, por favor.", "Consegues ler o mail, por favor?"):
        assert draft in engine.respond(message)
        assert engine.get_last_turn_telemetry()["llm_calls"] == 0
    assert len(local.calls) == calls
    assert engine._work_session.objects[artifact].payload["saved_file"] == saved
    assert (dict(trip.entities), dict(trip.parameters), list(trip.collected_facts)) == trip_fields


@pytest.mark.parametrize("target", ["gmail", "www.youtube.com", "https://www.google.com"])
def test_external_site_with_existing_workspace_objects_still_requires_confirmation(tmp_path, target):
    engine, local = english_engine(tmp_path)
    collected_email(engine)
    engine.respond("Vamos também preparar umas férias em Estocolmo.")
    response = engine.respond("abre " + target)
    assert engine.agent.has_pending_confirmation()
    assert response.startswith("Would you like me to open")
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0
    assert "cancelled" in engine.respond("Não.")
    assert not engine.agent.has_pending_confirmation()
    assert local.calls == []


def test_name_spelling_correction_stays_in_workflow(tmp_path):
    engine, local = english_engine(tmp_path)
    artifact = collected_email(engine)
    response = engine.respond("Escreveste mal Karina. Karina é com C.")
    assert "Carina" in response
    assert engine._pending_writing_task.recipient == "Francisca e Karina"
    response = engine.respond("Exatamente. Quero que corrijas, por favor.")
    assert engine._pending_writing_task.recipient == "Francisca e Carina"
    assert engine._pending_writing_task.active_artifact_id == artifact
    assert engine._work_session.objects[artifact].payload["recipient"] == "Francisca e Carina"
    assert engine.get_last_turn_telemetry()["selected_path"] == "WRITING_TASK"
    assert local.calls == []


def test_real_log_flow_saves_canonical_email_and_adds_another(tmp_path):
    draft = "Subject: Meeting this week\n\nDear Francisca and Carina,\n\nCan you attend the meeting this week at 9:30 PM?\n\nBest regards,\nAlexandre"
    engine, local = english_engine(tmp_path, [draft])
    artifact = collected_email(engine)
    engine.respond("Escreveste mal Karina. Karina é com C.")
    engine.respond("Exatamente. Quero que corrijas, por favor.")
    response = engine.respond("Agora preciso que guardes isso num fecheiro, num workspace, por favor.")
    assert "Saved" in response
    assert engine._work_session.objects[artifact].payload["draft_content"] == draft
    assert len(local.calls) == 1
    saved_name = engine._work_session.objects[artifact].payload["saved_file"]
    assert (engine.workspace_path / saved_name.removeprefix("workspace/")).read_text(encoding="utf-8") == draft
    assert engine._last_fast_tools_used == ("create_workspace_file",)
    response = engine.respond("Qual é o nome do fecheiro?")
    assert saved_name in response
    response = engine.respond("Porquê que aparece ainda a task no ecrã? Obrigado.")
    assert "open" in response
    assert "schedule" not in response
    engine.respond("Então acrescenta mais um e-mail para ver como é que fica no ecrã.")
    assert engine._pending_writing_task.active_artifact_id != artifact
    assert len([obj for obj in engine._work_session.objects.values() if obj.kind == "writing_task"]) == 2
    assert engine._work_session.objects[artifact].payload["draft_content"] == draft
    assert len(local.calls) == 1


def test_failed_draft_generation_never_writes_a_file(tmp_path):
    engine, _ = english_engine(tmp_path, ["I've saved the email in the workspace."])
    artifact = collected_email(engine)
    response = engine.respond("Guarda isso num ficheiro chamado teste.txt")
    assert "couldn't" in response
    assert not (engine.workspace_path / "teste.txt").exists()
    assert engine._pending_writing_task.active_artifact_id == artifact
    assert not engine._work_session.objects[artifact].payload.get("draft_content")


def test_recipient_spelling_proposal_can_be_declined(tmp_path):
    engine, local = english_engine(tmp_path)
    artifact = collected_email(engine)
    engine.respond("Karina é com C.")
    engine.respond("Não.")
    assert engine._pending_writing_task.recipient == "Francisca e Karina"
    assert engine._pending_writing_task.proposed_recipient_correction is None
    assert engine._pending_writing_task.active_artifact_id == artifact
    assert local.calls == []


def test_failed_save_has_no_success_receipt(tmp_path):
    draft = "Subject: Meeting\n\nHi Francisca and Karina,\n\nCan you attend this week at 21:30?\n\nAlexandre"
    engine, _ = english_engine(tmp_path, [draft])
    artifact = collected_email(engine)
    engine.respond("Guarda isso num ficheiro.")
    engine.respond("Guarda o rascunho em existing.txt")
    before = (engine.workspace_path / "existing.txt").read_text(encoding="utf-8")
    engine.respond("Guarda o rascunho em existing.txt")
    assert (engine.workspace_path / "existing.txt").read_text(encoding="utf-8") == before
    assert engine._work_session.objects[artifact].payload["saved_file"] == "workspace/existing.txt"


def test_machine_footer_does_not_become_email_content(tmp_path):
    draft = "Subject: Meeting\n\nHi Francisca and Karina,\n\nCan you attend?\n\nAlexandre\n\nSystem checks: All circuits awake."
    engine, _ = english_engine(tmp_path, [draft])
    artifact = collected_email(engine)
    engine.respond("Guarda isso num ficheiro.")
    assert engine._pending_writing_task is not None
    assert not engine._work_session.objects[artifact].payload.get("draft_content")


@pytest.mark.parametrize("instruction,expected", [
    ("às 9h30 da noite", 21 * 60 + 30),
    ("às 9h30 da manhã", 9 * 60 + 30),
    ("at 9:30 PM", 21 * 60 + 30),
    ("às 3h15 da tarde", 15 * 60 + 15),
])
def test_explicit_time_period_is_preserved(instruction, expected):
    from assistant.conversation import _requested_writing_clock
    assert _requested_writing_clock(instruction) == expected


def test_initial_email_with_missing_period_cannot_be_exported(tmp_path):
    draft = "Subject: Meeting\n\nDear Francisca and Karina,\n\nCan you attend this week at 9:30?\n\nAlexandre"
    engine, local = english_engine(tmp_path, [draft])
    artifact = collected_email(engine)
    response = engine.respond("Guarda o rascunho em meeting.txt")
    assert "couldn't" in response
    assert not engine._work_session.objects[artifact].payload.get("draft_content")
    assert not (engine.workspace_path / "meeting.txt").exists()
    assert "21:30" in str(local.calls[0]["messages"])


def test_initial_email_accepts_equivalent_twenty_four_hour_time(tmp_path):
    draft = "Subject: Meeting\n\nDear Francisca and Karina,\n\nCan you attend this week at 21:30?\n\nAlexandre"
    engine, local = english_engine(tmp_path, [draft])
    artifact = collected_email(engine)
    engine.respond("Guarda o rascunho em meeting.txt")
    assert engine._work_session.objects[artifact].payload["draft_content"] == draft
    assert len(local.calls) == 1


def test_automatic_names_do_not_overwrite_existing_files(tmp_path):
    from assistant.conversation import _automatic_export_filename
    (tmp_path / "ferias_estocolmo.txt").write_text("original", encoding="utf-8")
    assert _automatic_export_filename(tmp_path, "Férias: Estocolmo") == "ferias_estocolmo_2.txt"
    assert (tmp_path / "ferias_estocolmo.txt").read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("title", ["../../C:\\Windows\\notes", "CON", "...", "A" * 250])
def test_automatic_export_name_is_a_safe_local_text_filename(tmp_path, title):
    from assistant.conversation import _automatic_export_filename
    filename = _automatic_export_filename(tmp_path, title)
    assert filename.endswith(".txt") and len(filename) < 100
    assert "/" not in filename and "\\" not in filename and ":" not in filename
    assert filename.split(".")[0].upper() != "CON"
