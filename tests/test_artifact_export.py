import pytest

from assistant.conversation import ActiveTaskContext
from tests.test_automatic_routing_regressions import make_engine


@pytest.mark.parametrize("language", ["pt-PT", "en-GB"])
@pytest.mark.parametrize("task_type", ["plan_trip", "generic_comparison", "make_decision", "study_topic", "research_topic", "generic_activity", "writing"])
def test_save_current_task_uses_real_content_without_a_filename_or_llm(tmp_path, language, task_type):
    engine, _, local, cloud = make_engine(tmp_path, mode="local")
    engine._set_response_language(language)
    task = ActiveTaskContext(id="task-export", task_type=task_type, goal="organizar Projeto Aurora", parameters={"topic": "Projeto Aurora", "destination": "Estocolmo"})
    engine._active_task_context = task
    engine._update_task_artifact_ui(task)
    before = list(task.source_turns)
    response = engine.respond("Guarda isso num ficheiro na workspace.")
    files = list(engine.workspace_path.glob("*.txt"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "Estocolmo" in content or "Aurora" in content
    assert "source_turns" not in content and "confidence=" not in content
    assert ("Saved" if language == "en-GB" else "Guardei") in response
    assert files[0].name in response
    assert task.source_turns == before
    assert local.calls == [] and cloud.calls == []
    assert engine.get_last_turn_telemetry()["selected_path"] == "ARTIFACT_EXPORT"
    assert engine.get_last_turn_telemetry()["output_file_created"]
    payload = engine._work_session.objects[task.active_artifact_id].payload
    assert payload["saved_file"] == "workspace/" + files[0].name
    engine._update_task_artifact_ui(task)
    assert engine._work_session.objects[task.active_artifact_id].payload["saved_file"] == payload["saved_file"]
    assert files[0].name in engine.respond("Qual é o nome do ficheiro?")
    assert local.calls == []


def test_real_travel_flow_exports_known_plan_not_invented_itinerary(tmp_path):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    engine.respond("Vamos preparar umas férias em Estocolmo.")
    engine.respond("Vamos dois, de 12 a 15 de junho, e queremos visitar museus e passear.")
    task = engine._active_task_context
    response = engine.respond("Guarda o itinerário num ficheiro chamado ferias.txt.")
    target = engine.workspace_path / "ferias.txt"
    assert target.exists()
    assert "Estocolmo" in target.read_text(encoding="utf-8")
    assert "ferias.txt" in response
    assert engine._active_task_context is task
    assert local.calls == []


def test_prepared_task_draft_is_exported_exactly(tmp_path):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    draft = "Itinerário de Estocolmo\n\nDia 1: passeio definido pelo utilizador.\nDia 2: museus."
    task = ActiveTaskContext(id="trip", task_type="plan_trip", parameters={"generated_draft": draft, "destination": "Estocolmo"})
    engine._active_task_context = task
    engine._update_task_artifact_ui(task)
    engine.respond("Guarda o itinerário na workspace.")
    assert next(engine.workspace_path.glob("*.txt")).read_text(encoding="utf-8") == draft
    assert local.calls == []


def test_ambiguous_tasks_do_not_export_arbitrary_content(tmp_path):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    first = ActiveTaskContext(id="trip-one", task_type="plan_trip", parameters={"destination": "Coimbra"})
    second = ActiveTaskContext(id="trip-two", task_type="plan_trip", parameters={"destination": "Estocolmo"})
    engine._active_task_context = first
    engine._paused_task_stack = [second]
    engine._update_task_artifact_ui(first)
    engine._update_task_artifact_ui(second)
    response = engine.respond("Guarda o itinerário num ficheiro.")
    assert "?" in response
    assert list(engine.workspace_path.glob("*.txt")) == []
    assert local.calls == []


@pytest.mark.parametrize("message", ["Guarda o plano em ../outside.txt", "Save the plan to C:\\outside.txt"])
def test_task_export_does_not_escape_workspace(tmp_path, message):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    engine.respond("Vamos preparar umas férias em Estocolmo.")
    engine.respond(message)
    assert not (tmp_path / "outside.txt").exists()
    assert list(engine.workspace_path.glob("*.txt")) == []
    assert local.calls == []


def test_export_does_not_overwrite_explicit_existing_filename(tmp_path):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    engine.respond("Compara Samsung com Apple")
    path = engine.workspace_path / "comparison.txt"
    path.write_text("original", encoding="utf-8")
    engine.respond("Guarda a comparação em comparison.txt")
    assert path.read_text(encoding="utf-8") == "original"
    task = engine._active_task_context
    assert not engine._work_session.objects[task.active_artifact_id].payload.get("saved_file")
    assert local.calls == []


@pytest.mark.parametrize("language", ["pt-PT", "en-GB"])
def test_document_draft_export_chooses_name_and_keeps_original(tmp_path, language):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    engine._set_response_language(language)
    source = engine.workspace_path / "notas.txt"
    source.write_text("Texto original.", encoding="utf-8")
    engine.respond("Lê o ficheiro notas.txt")
    draft = "Prepared revised text.\nSecond paragraph."
    engine._active_document_context.draft_content = draft
    engine._active_document_context.draft_revision = 1
    response = engine.respond("Guarda esta versão num ficheiro na workspace.")
    outputs = [p for p in engine.workspace_path.glob("*.txt") if p != source]
    assert len(outputs) == 1
    assert outputs[0].read_text(encoding="utf-8") == draft
    assert source.read_text(encoding="utf-8") == "Texto original."
    assert ("Saved" if language == "en-GB" else "Guardei") in response
    assert local.calls == []


def test_document_summary_export_reuses_prepared_output_not_source(tmp_path):
    summary = "Summary: a short meeting on Thursday."
    engine, _, local, _ = make_engine(tmp_path, mode="local", ollama_replies=[summary])
    engine._set_response_language("en-GB")
    (engine.workspace_path / "notas.txt").write_text("A reunião de quinta-feira serve para rever o relatório e distribuir tarefas.", encoding="utf-8")
    engine.respond("Resume o ficheiro notas.txt.")
    calls = len(local.calls)
    response = engine.respond("Guarda esse resumo num ficheiro.")
    outputs = [p for p in engine.workspace_path.glob("*.txt") if p.name != "notas.txt"]
    assert len(outputs) == 1
    assert outputs[0].read_text(encoding="utf-8") == summary
    assert "Saved" in response
    assert len(local.calls) == calls


def test_inline_text_summary_is_saved_without_copying_other_chat(tmp_path):
    summary = "- First point.\n- Second point."
    engine, _, local, _ = make_engine(tmp_path, mode="local", ollama_replies=[summary])
    engine._set_response_language("en-GB")
    engine.respond("Resume este texto em dois pontos: A reunião definiu o relatório e as tarefas da semana.")
    calls = len(local.calls)
    response = engine.respond("Guarda esse resumo num ficheiro.")
    assert next(engine.workspace_path.glob("*.txt")).read_text(encoding="utf-8") == summary
    assert "Saved" in response
    assert len(local.calls) == calls
    engine.clear_conversation()
    assert engine._work_session.prepared_output == {}


def test_missing_document_draft_does_not_save_the_original_by_mistake(tmp_path):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    source = engine.workspace_path / "notas.txt"
    source.write_text("Original.", encoding="utf-8")
    engine.respond("Lê o ficheiro notas.txt")
    response = engine.respond("Guarda o rascunho num ficheiro.")
    assert list(engine.workspace_path.glob("*.txt")) == [source]
    assert "guardar" in response or "rascunho" in response
    assert local.calls == []


@pytest.mark.parametrize("quoted", [False, True])
def test_explicit_workspace_read_preempts_research_content_resolution(tmp_path, quoted):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    engine._set_response_language("en-GB")
    engine.respond("Compara Samsung com Apple")
    (engine.workspace_path / "notes.txt").write_text("Canonical document content.", encoding="utf-8")
    response = engine.respond('Lê o ficheiro "notes.txt".' if quoted else "Lê o ficheiro notes.txt.")
    assert response.startswith("Here")
    assert "Canonical document content." in response
    assert engine.get_last_turn_telemetry()["selected_path"] == "DOCUMENT_TASK"
    assert local.calls == []


@pytest.mark.parametrize("message,prefix", [
    ("Mostra o original.", "Here"),
    ("Mostra o rascunho.", "Here"),
    ("Descarta o rascunho.", "Discarded"),
    ("Volta à versão anterior.", "Restored"),
])
def test_document_controls_keep_configured_language(tmp_path, message, prefix):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    engine._set_response_language("en-GB")
    engine.respond("Compara Samsung com Apple")
    (engine.workspace_path / "notes.txt").write_text("Original text.", encoding="utf-8")
    engine.respond("Lê o ficheiro notes.txt.")
    context = engine._active_document_context
    context.draft_content = "Canonical draft."
    context.draft_revision = 2
    context.previous_draft_content = "Earlier draft."
    context.previous_draft_revision = 1
    response = engine.respond(message)
    assert response.startswith(prefix)
    assert local.calls == []


def test_followup_summary_replaces_previous_prepared_summary(tmp_path):
    summary = "Latest prepared summary."
    engine, _, local, _ = make_engine(tmp_path, mode="local", ollama_replies=[summary])
    engine._set_response_language("en-GB")
    (engine.workspace_path / "notes.txt").write_text("Meeting on Thursday, report on Friday.", encoding="utf-8")
    engine.respond("Lê o ficheiro notes.txt.")
    engine._active_document_context.prepared_content = "Old summary."
    engine.respond("Faz um resumo.")
    engine.respond("Guarda esse resumo num ficheiro.")
    exported = [path for path in engine.workspace_path.glob("*.txt") if path.name != "notes.txt"]
    assert exported[0].read_text(encoding="utf-8") == summary
    assert len(local.calls) == 1


def test_generate_and_save_document_output_does_not_ask_for_a_name(tmp_path):
    summary = "Prepared summary of the real document."
    engine, _, local, _ = make_engine(tmp_path, mode="local", ollama_replies=[summary])
    engine._set_response_language("en-GB")
    (engine.workspace_path / "notes.txt").write_text("Meeting on Thursday; report on Friday.", encoding="utf-8")
    response = engine.respond("Resume o ficheiro notes.txt e guarda o resumo num ficheiro txt.")
    outputs = [p for p in engine.workspace_path.glob("*.txt") if p.name != "notes.txt"]
    assert len(outputs) == 1 and outputs[0].read_text(encoding="utf-8") == summary
    assert "Saved" in response
    assert len(local.calls) == 1


def test_legacy_prepared_email_uses_same_export_contract(tmp_path):
    draft = "Subject: Review\n\nHi Ana,\n\nPlease review the report.\n\nAlexandre"
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    task = ActiveTaskContext(id="legacy-email", task_type="compose_email", entities={"recipient": "Ana"}, parameters={"generated_draft": draft})
    engine._active_task_context = task
    engine._update_task_artifact_ui(task)
    engine.respond("Guarda esse email.")
    assert next(engine.workspace_path.glob("*.txt")).read_text(encoding="utf-8") == draft
    assert engine._work_session.objects[task.active_artifact_id].payload["saved_file"]
    assert local.calls == []


@pytest.mark.parametrize("preamble", ["Here is the rewritten document:", "Here's the revised version:"])
def test_english_rewrite_preamble_is_not_part_of_document(preamble):
    from assistant.conversation import _strip_rewrite_preamble
    assert _strip_rewrite_preamble(preamble + "\n\nCanonical document.") == "Canonical document."


@pytest.mark.parametrize("source,expected", [
    ("Criei o ficheiro 'note.txt' na pasta workspace.", "Created the file 'note.txt' in the workspace."),
    ("Nao posso criar ficheiros fora da pasta workspace.", "I can't create files outside the workspace."),
    ("So posso criar ficheiros com extensao .txt.", "I can only create .txt files."),
    ("Preciso do nome do ficheiro da workspace que queres usar.", "Which file in the workspace would you like me to use?"),
    ("Não encontrei esse objeto no workspace.", "I couldn't find that object in the workspace."),
])
def test_fixed_file_responses_use_selected_language(source, expected):
    from assistant.response_language import localize_deterministic_response
    assert localize_deterministic_response(source, "en-GB") == expected
    assert localize_deterministic_response(source, "pt-PT") == source


def test_document_summary_is_also_the_content_sent_to_its_card(tmp_path):
    import json
    summary = "Prepared summary of the document."
    engine, _, _, _ = make_engine(tmp_path, mode="local", ollama_replies=[summary])
    engine._set_response_language("en-GB")
    (engine.workspace_path / "notes.txt").write_text("Meeting on Thursday, report on Friday.", encoding="utf-8")
    engine.respond("Resume o ficheiro notes.txt.")
    events = [json.loads(event) for event in engine.consume_ui_events()]
    assert any(event["type"] == "document_draft_ready" and event.get("content") == summary for event in events)


@pytest.mark.parametrize("other_task", [False, True])
def test_email_prepared_from_document_is_not_confused_with_an_older_email(tmp_path, other_task):
    from tests.test_conversation_language_and_initiative import make_email_draft
    engine, local, older_id, older_draft = make_email_draft(tmp_path)
    if other_task:
        engine.respond("Compara Samsung com Apple")
    prepared = "Subject: Notes\n\nHi Ana,\n\nThe report is due on Friday.\n\nAlexandre"
    (engine.workspace_path / "notes.txt").write_text("The report is due on Friday.", encoding="utf-8")
    local.replies.append(prepared)
    response = engine.respond("Lê o ficheiro notes.txt e escreve um email para a Ana com o resumo.")
    assert prepared == response
    calls = len(local.calls)
    engine.respond("Guarda esse email num ficheiro.")
    files = list(engine.workspace_path.glob("*.txt"))
    exported = [p for p in files if p.name != "notes.txt"]
    assert len(exported) == 1 and exported[0].read_text(encoding="utf-8") == prepared
    assert engine._work_session.objects[older_id].payload["draft_content"] == older_draft
    assert len(local.calls) == calls
