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
