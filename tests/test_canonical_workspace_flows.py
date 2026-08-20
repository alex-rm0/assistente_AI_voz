from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant.attention import ACTIVE, FOCUS, MINIMIZED
from assistant.representation import SemanticObject, register_object
from assistant.text_matching import normalize_for_matching
from tests.test_automatic_routing_regressions import make_engine


def _telemetry(engine) -> dict:
    return engine.get_last_turn_telemetry() or {}


def test_unrelated_social_turn_is_not_captured_by_active_task_notes(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["Percebo. Descansar um pouco também faz parte."],
    )

    engine.respond("Vamos planear umas férias.")
    task = engine._work_session.active_task
    assert task is not None
    facts_before = list(task.collected_facts)

    response = engine.respond("Hoje estou cansado.")
    telemetry = _telemetry(engine)

    assert response
    assert telemetry["canonical_interpretation_primary_route"] == "GENERAL_CONVERSATION"
    assert telemetry["selected_path"] == "GENERAL_CONVERSATION"
    assert telemetry["task_continuation_rejected"] is True
    assert telemetry["active_task_continuation_rejected_reason"] == "no_positive_task_evidence"
    assert task.collected_facts == facts_before
    assert telemetry["workspace_command_detected"] is False
    assert telemetry["representation_command_detected"] is False
    assert telemetry["attention_command_detected"] is False


def test_equivalent_email_phrasings_use_the_canonical_writing_workflow(tmp_path: Path) -> None:
    draft = (
        "Olá João,\n\n"
        "Escrevo-te sobre a reunião de amanhã.\n\n"
        "Um abraço,\n"
        "Alexandre"
    )
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[draft])

    engine.respond("Preciso de escrever um email.")
    task = engine._work_session.writing_context
    telemetry = _telemetry(engine)
    assert task is not None
    task_id = id(task)
    artifact_id = task.active_artifact_id
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert engine._work_session.active_task is None

    engine.respond("É para o João.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert id(engine._work_session.writing_context) == task_id
    assert engine._work_session.writing_context.active_artifact_id == artifact_id

    engine.respond("Sobre a reunião de amanhã.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert id(engine._work_session.writing_context) == task_id
    assert engine._work_session.writing_context.active_artifact_id == artifact_id

    engine.respond("Podes escrever.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert telemetry["llm_calls"] >= 1
    assert len(ollama.calls) == telemetry["llm_calls"]


def test_writing_workflow_consumes_slot_aware_followups(tmp_path: Path) -> None:
    draft = (
        "Assunto: Reunião de amanhã\n\n"
        "Olá João,\n\n"
        "Escrevo-te sobre a reunião de amanhã.\n\n"
        "Um abraço,\n"
        "Alexandre"
    )
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[draft])

    response = engine.respond("Ajuda-me a escrever um email.")
    task = engine._work_session.writing_context
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert task is not None
    task_id = id(task)
    assert task.awaiting_field == "recipient"
    assert telemetry["writing_awaiting_field"] == "recipient"
    assert "Para quem é o email?" in response

    response = engine.respond("É para o João.")
    task = engine._work_session.writing_context
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert task is not None
    assert id(task) == task_id
    assert task.recipient == "João"
    assert task.awaiting_field != "recipient"
    assert telemetry["writing_recipient"] == "João"
    assert "Para quem é o email?" not in response

    engine.respond("Sobre a reunião de amanhã.")
    task = engine._work_session.writing_context
    telemetry = _telemetry(engine)
    assert task is not None
    assert id(task) == task_id
    assert "reunião de amanhã" in task.subject
    assert "reunião de amanhã" in telemetry["writing_subject"]

    engine.respond("Podes escrever.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert telemetry["llm_calls"] == 1
    assert engine._work_session.writing_context is None
    assert len(ollama.calls) == 1


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Para o João.", "João"),
        ("João.", "João"),
        ("É para a Marta.", "Marta"),
        ("Para a equipa de marketing.", "Equipa de marketing"),
    ],
)
def test_writing_workflow_recipient_paraphrases(tmp_path: Path, message: str, expected: str) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("Ajuda-me a escrever um email.")
    response = engine.respond(message)
    task = engine._work_session.writing_context
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WRITING_TASK"
    assert task is not None
    assert task.recipient == expected
    assert task.awaiting_field != "recipient"
    assert "Para quem é o email?" not in response
    assert ollama.calls == []


def test_explicit_second_activity_starts_writing_workflow_and_preserves_visual_targets(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("Vamos planear umas férias.")
    plan_task = engine._work_session.active_task
    assert plan_task is not None
    plan_task_id = plan_task.id
    plan_artifact_id = plan_task.active_artifact_id

    response = engine.respond("Agora, ao mesmo tempo, ajuda-me num email.")
    telemetry = _telemetry(engine)
    writing_task = engine._work_session.writing_context
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert telemetry["llm_calls"] == 0
    assert writing_task is not None
    assert writing_task.active_artifact_id
    assert writing_task.active_artifact_id in engine._work_session.objects
    assert engine._work_session.active_task is None
    assert any(task.id == plan_task_id for task in engine._work_session.paused_task_stack)
    assert "Para quem é o email?" in response

    response = engine.respond("É para o João.")
    telemetry = _telemetry(engine)
    writing_task = engine._work_session.writing_context
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert telemetry["llm_calls"] == 0
    assert writing_task is not None
    assert writing_task.recipient == "João"
    assert "Para quem é o email?" not in response

    engine.respond("Abre o planeamento de férias.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert telemetry["attention_command_target_ids"] == [plan_artifact_id]
    assert engine._work_session.attention.focus_object_id == plan_artifact_id
    assert engine._work_session.writing_context is writing_task

    engine.respond("Volta ao mail.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert telemetry["attention_target_resolution_source"] == "writing_context_alias"
    assert telemetry["attention_command_target_ids"] == [writing_task.active_artifact_id]
    assert engine._work_session.attention.focus_object_id == writing_task.active_artifact_id
    assert engine._work_session.writing_context is writing_task
    assert ollama.calls == []


def test_second_task_and_visual_return_preserve_distinct_foci(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("Vamos planear umas férias.")
    plan_task = engine._work_session.active_task
    assert plan_task is not None
    plan_artifact_id = plan_task.active_artifact_id

    engine.respond("Agora preciso de escrever um email.")
    email_task = engine._work_session.writing_context
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "WRITING_TASK"
    assert email_task is not None
    assert email_task.task_type == "compose_email"
    assert email_task.active_artifact_id
    assert any(task.id == plan_task.id for task in engine._work_session.paused_task_stack)

    engine.respond("Abre o planeamento de férias.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert engine._work_session.attention.focus_object_id == plan_artifact_id
    assert engine._work_session.writing_context is email_task

    engine.respond("Volta ao email.")
    telemetry = _telemetry(engine)
    assert telemetry["llm_calls"] == 0
    assert engine._work_session.attention.focus_object_id == email_task.active_artifact_id
    assert ollama.calls == []


def test_generic_comparison_activity_starts_with_owned_options(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("Vamos comparar Samsung com iPhone.")
    telemetry = _telemetry(engine)
    activity = engine._work_session.active_task

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert activity is not None
    assert activity.task_type == "generic_comparison"
    assert activity.activity_family == "comparison"
    assert activity.active_artifact_id in engine._work_session.objects
    assert "Samsung" in activity.parameters["options"]
    assert "iPhone" in activity.parameters["options"]
    option_objects = {obj.title: obj for obj in engine._work_session.objects.values() if obj.kind == "option"}
    assert set(option_objects) == {"Samsung", "iPhone"}
    assert all(obj.task_id == activity.id for obj in option_objects.values())
    assert "Samsung" in response and "iPhone" in response
    assert ollama.calls == []


def test_generic_comparison_start_materializes_lowercase_subjects_with_reason_clause(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("vamos comparar samsung com apple, preciso de um telefone novo")
    telemetry = _telemetry(engine)
    activity = engine._work_session.active_task

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert activity is not None
    assert activity.task_type == "generic_comparison"
    option_objects = {obj.title: obj for obj in engine._work_session.objects.values() if obj.kind == "option"}
    assert set(option_objects) == {"Samsung", "Apple"}
    assert option_objects["Samsung"].task_id == activity.id
    assert option_objects["Apple"].task_id == activity.id
    assert "telefone novo" in activity.source_turns[0]
    assert "Samsung" in response and "Apple" in response
    assert ollama.calls == []


def test_generic_comparison_can_start_while_email_is_preserved(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("Vamos escrever um mail.")
    email = engine._work_session.active_task
    assert email is not None
    email_artifact_id = email.active_artifact_id

    engine.respond(
        "O João é o meu professor de Cálculo, e quero enviar um mail a perguntar quando é que está marcada a entrega do trabalho 1."
    )
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "TASK_CONTINUATION"
    assert engine._work_session.active_task is email
    assert email.entities["recipient"] == "João"
    assert email.entities["recipientRole"] == "professor"
    assert "Cálculo" in email.parameters["course"]
    assert "entrega do trabalho 1" in email.parameters["topic"]
    assert email.status == "ready"

    engine.respond("Enquanto isso, estou a pensar comprar um telefone novo, vamos comparar Samsung com iPhone.")
    telemetry = _telemetry(engine)
    comparison = engine._work_session.active_task
    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert comparison is not None
    assert comparison.task_type == "generic_comparison"
    assert any(task.id == email.id for task in engine._work_session.paused_task_stack)
    assert email.active_artifact_id == email_artifact_id
    option_objects = {obj.title: obj for obj in engine._work_session.objects.values() if obj.kind == "option"}
    assert {"Samsung", "iPhone"}.issubset(option_objects)
    assert option_objects["Samsung"].task_id == comparison.id
    assert option_objects["iPhone"].task_id == comparison.id

    engine.respond("Compara bateria e câmara.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "TASK_CONTINUATION"
    assert engine._work_session.active_task.id == comparison.id
    assert any("bateria" in item and "câmara" in item for item in comparison.requested_outputs)

    engine.respond("Adiciona o Pixel.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert engine._work_session.objects["option-pixel"].task_id == comparison.id

    engine.respond("Volta ao email.")
    telemetry = _telemetry(engine)
    assert telemetry["selected_path"] == "TASK_RESUME"
    assert telemetry["task_resumed"] is True
    assert telemetry["llm_calls"] == 0
    assert engine._work_session.attention.focus_object_id == email_artifact_id
    assert any(task.id == comparison.id for task in engine._work_session.paused_task_stack)
    assert engine._work_session.active_task.id == email.id
    assert ollama.calls == []


def test_compound_workspace_command_shows_email_and_minimizes_owned_comparison_options(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos escrever um mail.")
    email_task = engine._work_session.active_task
    assert email_task is not None
    email_artifact_id = email_task.active_artifact_id

    engine.respond("Enquanto isso, estou a pensar comprar um telefone novo, vamos comparar Samsung com iPhone.")
    comparison_task = engine._work_session.active_task
    engine.respond("Adiciona o Pixel.")
    assert comparison_task is not None
    option_ids = {
        obj.title: object_id
        for object_id, obj in engine._work_session.objects.items()
        if obj.kind == "option" and obj.task_id == comparison_task.id
    }
    assert {"Samsung", "iPhone", "Pixel"} == set(option_ids)

    response = engine.respond("preciso que abras o mail para ver, minimiza os telefones")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["compound_workspace_intent"] is True
    assert telemetry["compound_workspace_intent_count"] == 2
    assert telemetry["executed_operations"] == 2
    assert telemetry["failed_operations"] == 0
    assert telemetry["partial_execution"] is False
    assert telemetry["llm_calls"] == 0
    assert "email" in response.lower()
    assert "opções" in response.lower()
    assert engine._work_session.active_task.id == comparison_task.id
    assert engine._work_session.attention.object_states[email_artifact_id] == FOCUS
    for option_id in option_ids.values():
        assert engine._work_session.attention.object_states[option_id] == MINIMIZED
    assert engine._work_session.attention.object_states[email_artifact_id] != MINIMIZED
    assert ollama.calls == []


def test_collection_resolution_does_not_grab_unrelated_task_artifact(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos escrever um mail.")
    email_task = engine._work_session.active_task
    engine.respond("Enquanto isso, estou a pensar comprar um telefone novo, vamos comparar Samsung com iPhone.")
    comparison_task = engine._work_session.active_task
    engine.respond("Adiciona o Pixel.")
    assert email_task is not None and comparison_task is not None

    engine.respond("minimiza os telefones")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    minimized_ids = {
        object_id
        for object_id, level in engine._work_session.attention.object_states.items()
        if level == MINIMIZED
    }
    assert email_task.active_artifact_id not in minimized_ids
    assert comparison_task.active_artifact_id not in minimized_ids
    assert {"option-samsung", "option-iphone", "option-pixel"}.issubset(minimized_ids)
    assert ollama.calls == []


def test_add_to_collection_uses_target_owner_not_active_email_task(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Estou a pensar comprar um telefone novo, vamos comparar Samsung com iPhone.")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None

    engine.respond("Vamos escrever um mail.")
    email_task = engine._work_session.active_task
    assert email_task is not None and email_task.id != comparison_task.id

    response = engine.respond("acrescenta Pixel aos telefones")
    telemetry = _telemetry(engine)

    assert response == "Acrescentei Pixel."
    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert "option-pixel" in engine._work_session.objects
    assert "option-pixel-aos-telefones" not in engine._work_session.objects
    pixel = engine._work_session.objects["option-pixel"]
    assert pixel.title == "Pixel"
    assert pixel.task_id == comparison_task.id
    assert engine._work_session.active_task.id == email_task.id
    assert ollama.calls == []


@pytest.mark.parametrize("message", ["acrescenta Samsung aos telefones", "adiciona Samsung e Apple às opções"])
def test_active_generic_comparison_cannot_swallow_add_to_collection(tmp_path: Path, message: str) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None
    facts_before = list(comparison_task.collected_facts)

    response = engine.respond(message)
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert comparison_task.collected_facts == facts_before
    assert "já fazem parte" in response
    assert len([obj for obj in engine._work_session.objects.values() if obj.kind == "option" and obj.title == "Samsung"]) == 1
    assert ollama.calls == []


def test_multi_add_to_collection_creates_multiple_objects_before_active_task_notes(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("estou a pensar comprar um telefone novo, vamos comparar Pixel com Nothing")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None
    facts_before = list(comparison_task.collected_facts)

    response = engine.respond("acrescenta samsung e apple aos telefones")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert "Acrescentei Samsung e Apple" in response
    owned_titles = {
        obj.title
        for obj in engine._work_session.objects.values()
        if obj.kind == "option" and obj.task_id == comparison_task.id
    }
    assert owned_titles == {"Pixel", "Nothing", "Samsung", "Apple"}
    assert comparison_task.collected_facts == facts_before
    assert ollama.calls == []


def test_collection_membership_invariant_after_start_and_add(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None
    engine.respond("acrescenta Pixel aos telefones")

    owned = {
        obj.title: object_id
        for object_id, obj in engine._work_session.objects.items()
        if obj.kind == "option" and obj.task_id == comparison_task.id
    }
    assert set(owned) == {"Samsung", "Apple", "Pixel"}

    engine.respond("minimiza os telefones")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["llm_calls"] == 0
    for object_id in owned.values():
        assert engine._work_session.attention.object_states[object_id] == MINIMIZED
    assert ollama.calls == []


def test_qt_flow_add_collection_minimize_expand_then_resume_writing(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Estou a pensar comprar um telefone novo, vamos comparar Samsung com iPhone.")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None

    engine.respond("Vamos escrever um mail.")
    email_task = engine._work_session.active_task
    assert email_task is not None
    assert email_task.missing_information == ["recipient"]

    engine.respond("acrescenta Pixel aos telefones")
    assert engine._work_session.objects["option-pixel"].task_id == comparison_task.id
    assert engine._work_session.active_task.id == email_task.id

    engine.respond("preciso que abras o mail para ver, minimiza os telefones")
    telemetry_b = _telemetry(engine)
    assert telemetry_b["executed_operations"] == 2
    assert telemetry_b["failed_operations"] == 0
    assert telemetry_b["llm_calls"] == 0
    for object_id in ("option-samsung", "option-iphone", "option-pixel"):
        assert engine._work_session.attention.object_states[object_id] == MINIMIZED
    assert engine._work_session.attention.object_states[email_task.active_artifact_id] == FOCUS

    response_c = engine.respond("aumenta o mail")
    telemetry_c = _telemetry(engine)
    assert telemetry_c["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry_c["llm_calls"] == 0
    assert response_c in ("Claro.", "Feito.", "Aumentei o email.")
    assert "Para quem" not in response_c
    assert engine._work_session.representation.major_object_ids == [email_task.active_artifact_id]
    assert engine._work_session.active_task.id == email_task.id
    assert engine._work_session.active_task.missing_information == ["recipient"]

    response_d = engine.respond("é para o João")
    telemetry_d = _telemetry(engine)
    assert telemetry_d["selected_path"] == "TASK_CONTINUATION"
    assert engine._work_session.active_task.entities["recipient"] == "João"
    assert "Para quem" not in response_d
    assert ollama.calls == []


def test_real_scenario_compare_add_email_then_compound_minimize_collection(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None
    assert {
        obj.title for obj in engine._work_session.objects.values()
        if obj.kind == "option" and obj.task_id == comparison_task.id
    } == {"Samsung", "Apple"}

    engine.respond("acrescenta Pixel aos telefones")
    assert engine._work_session.objects["option-pixel"].task_id == comparison_task.id

    engine.respond("vamos escrever um mail")
    email_task = engine._work_session.active_task
    assert email_task is not None
    assert email_task.task_type == "compose_email"
    assert email_task.active_artifact_id in engine._work_session.objects

    response = engine.respond("preciso que abras o mail para ver, minimiza os telefones")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["executed_operations"] == 2
    assert telemetry["failed_operations"] == 0
    assert telemetry["llm_calls"] == 0
    assert "opções" in response
    for object_id in ("option-samsung", "option-apple", "option-pixel"):
        assert engine._work_session.attention.object_states[object_id] == MINIMIZED
    assert engine._work_session.attention.object_states[email_task.active_artifact_id] == FOCUS
    assert engine._work_session.attention.object_states[email_task.active_artifact_id] != MINIMIZED
    assert ollama.calls == []


def test_compound_show_collection_then_adds_comparison_criteria_to_referenced_owner(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None
    engine.respond("acrescenta Pixel aos telefones")
    engine.respond("vamos escrever um mail")
    email_task = engine._work_session.active_task
    assert email_task is not None and email_task.id != comparison_task.id

    response = engine.respond("mostra os telefones, vamos comparar bateria e armazenamento")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["executed_operations"] == 2
    assert telemetry["failed_operations"] == 0
    assert telemetry["llm_calls"] == 0
    assert "critérios" in response
    option_titles = {
        obj.title
        for obj in engine._work_session.objects.values()
        if obj.kind == "option" and obj.task_id == comparison_task.id
    }
    criterion_titles = {
        obj.title
        for obj in engine._work_session.objects.values()
        if obj.kind == "criterion" and obj.task_id == comparison_task.id
    }
    assert option_titles == {"Samsung", "Apple", "Pixel"}
    assert criterion_titles == {"Bateria", "Armazenamento"}
    assert not [
        obj for obj in engine._work_session.objects.values()
        if obj.task_id == email_task.id and obj.kind in {"option", "criterion"}
    ]
    assert engine._work_session.active_task.id == email_task.id
    assert ollama.calls == []


def test_comparison_criteria_are_distinct_from_options(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None

    engine.respond("mostra os telefones, vamos comparar preço e consumo")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert {
        obj.title for obj in engine._work_session.objects.values()
        if obj.kind == "option" and obj.task_id == comparison_task.id
    } == {"Samsung", "Apple"}
    assert {
        normalize_for_matching(obj.title) for obj in engine._work_session.objects.values()
        if obj.kind == "criterion" and obj.task_id == comparison_task.id
    } == {"preco", "consumo"}
    assert ollama.calls == []


def test_relative_all_except_preserves_anchor_and_minimizes_rest(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")
    engine.respond("acrescenta Pixel aos telefones")
    engine.respond("vamos escrever um mail")
    email_task = engine._work_session.active_task
    assert email_task is not None

    response = engine.respond("abre o mail para ver, minimiza tudo o resto")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["executed_operations"] == 2
    assert telemetry["failed_operations"] == 0
    assert telemetry["llm_calls"] == 0
    assert "resto" in response
    assert engine._work_session.attention.object_states[email_task.active_artifact_id] == FOCUS
    for object_id in ("option-samsung", "option-apple", "option-pixel"):
        assert engine._work_session.attention.object_states[object_id] == MINIMIZED
    assert engine._work_session.attention.object_states[email_task.active_artifact_id] != MINIMIZED
    assert ollama.calls == []


def test_relative_all_except_without_anchor_asks_clarification_without_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos comparar Samsung com Apple")

    response = engine.respond("minimiza tudo o resto")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["workspace_command_executed"] is False
    assert telemetry["llm_calls"] == 0
    assert "queres manter aberto" in response
    assert ollama.calls == []


def test_relative_all_except_respects_do_not_minimize_constraint(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")
    engine.respond("Não minimizes Samsung.")
    engine.respond("vamos escrever um mail")
    email_task = engine._work_session.active_task
    assert email_task is not None

    engine.respond("abre o mail para ver, minimiza tudo o resto")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert engine._work_session.attention.object_states[email_task.active_artifact_id] == FOCUS
    assert engine._work_session.attention.object_states["option-samsung"] != MINIMIZED
    assert engine._work_session.attention.object_states["option-apple"] == MINIMIZED
    assert ollama.calls == []


def test_associate_to_collection_mutates_orphan_criteria_without_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None
    engine.respond("vamos escrever um mail")
    email_task = engine._work_session.active_task
    assert email_task is not None
    register_object(
        engine._work_session,
        SemanticObject(id="criterion-bateria", kind="criterion", title="Bateria", task_id=email_task.id),
    )

    response = engine.respond("consegues associar aos telefones?")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert engine._work_session.objects["criterion-bateria"].task_id == comparison_task.id
    assert "Associei Bateria" in response
    assert ollama.calls == []


def test_associate_to_collection_does_not_fake_mutation_without_candidates(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")

    response = engine.respond("consegues associar aos telefones?")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert "Não encontrei" in response
    assert ollama.calls == []


@pytest.mark.parametrize(
    "message",
    [
        "Qual é a capital de França?",
        "Quanto é 5x8?",
        "O que achas disto?",
        "Estou cansado.",
        "Quem inventou o telefone?",
    ],
)
def test_non_continuable_messages_do_not_start_generic_activity(tmp_path: Path, message: str) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["Resposta curta."],
    )

    engine.respond(message)
    telemetry = _telemetry(engine)

    assert engine._work_session.active_task is None
    assert engine._work_session.writing_context is None
    assert telemetry["selected_path"] == "GENERAL_CONVERSATION"


def test_representation_and_spatial_commands_bypass_active_task_and_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("Vamos planear umas férias.")
    active_task_id = engine._work_session.active_task.id

    engine.respond("Vamos comparar Suécia, Itália e Açores.")
    telemetry = _telemetry(engine)
    assert telemetry["canonical_interpretation_primary_route"] == "REPRESENTATION_COMMAND"
    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert engine._work_session.active_task.id == active_task_id
    option_titles = {obj.title for obj in engine._work_session.objects.values() if obj.id.startswith("option-")}
    assert option_titles == {"Suécia", "Itália", "Açores"}

    object_ids_before = set(engine._work_session.objects)
    engine.respond("Mostra em lista.")
    telemetry = _telemetry(engine)
    assert telemetry["canonical_interpretation_primary_route"] == "REPRESENTATION_COMMAND"
    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert set(engine._work_session.objects) == object_ids_before
    assert engine._work_session.representation.style == "list"

    engine.respond("Põe Suécia e Itália lado a lado.")
    telemetry = _telemetry(engine)
    assert telemetry["canonical_interpretation_primary_route"] == "WORKSPACE_COMMAND"
    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert set(engine._work_session.objects) == object_ids_before
    assert ollama.calls == []


def test_attention_uses_semantic_targets_and_preserves_constraints(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos comparar Suécia, Itália e Açores.")
    engine.consume_ui_events()

    engine.respond("Foca na Suécia.")
    telemetry = _telemetry(engine)
    assert telemetry["canonical_interpretation_primary_route"] == "ATTENTION_COMMAND"
    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["attention_target_resolution_source"] in ("mentioned_object_ids", "workspace_reference")
    suecia_id = telemetry["attention_command_target_ids"][0]
    assert engine._work_session.objects[suecia_id].title == "Suécia"
    assert engine._work_session.attention.object_states[suecia_id] == FOCUS
    assert engine._work_session.attention.focus_object_id == suecia_id

    engine.respond("Minimiza Itália.")
    telemetry = _telemetry(engine)
    assert telemetry["canonical_interpretation_primary_route"] == "ATTENTION_COMMAND"
    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    italia_id = telemetry["attention_command_target_ids"][0]
    assert engine._work_session.objects[italia_id].title == "Itália"
    assert engine._work_session.attention.object_states[italia_id] == MINIMIZED

    engine.respond("Mostra Itália outra vez.")
    telemetry = _telemetry(engine)
    assert telemetry["canonical_interpretation_primary_route"] == "ATTENTION_COMMAND"
    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["attention_target_resolution_source"] in ("mentioned_object_ids", "workspace_reference", "single_hidden_or_minimized")
    assert engine._work_session.attention.object_states[italia_id] == ACTIVE

    engine.respond("Não minimizes a Suécia.")
    telemetry = _telemetry(engine)
    assert telemetry["canonical_interpretation_primary_route"] == "ATTENTION_COMMAND"
    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert engine._work_session.attention.object_states[suecia_id] == FOCUS
    assert engine._work_session.attention.focus_object_id == suecia_id
    assert suecia_id in engine._work_session.constraints.do_not_minimize_ids
    assert ollama.calls == []


def test_maximize_all_global_workspace_command_restores_visible_attention_without_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos comparar Samsung com iPhone.")
    engine.respond("Adiciona o Pixel.")
    pixel_id = "option-pixel"

    engine.respond("Minimiza Pixel.")
    assert engine._work_session.attention.object_states[pixel_id] == MINIMIZED

    engine.respond("maximiza todos")
    telemetry = _telemetry(engine)

    assert telemetry["canonical_interpretation_primary_route"] == "WORKSPACE_COMMAND"
    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["workspace_command_scope"] == "all"
    assert telemetry["workspace_command_action"] == "expand"
    assert telemetry["global_workspace_action"] is True
    assert telemetry["llm_calls"] == 0
    assert engine._work_session.attention.object_states[pixel_id] == ACTIVE
    assert engine._work_session.representation.hidden_object_ids == []
    assert ollama.calls == []


def test_expand_email_artifact_is_workspace_command_not_task_continuation(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos escrever um mail.")
    email_task = engine._work_session.active_task
    assert email_task is not None

    response = engine.respond("aumenta o mail")
    telemetry = _telemetry(engine)
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    command_events = [event for event in payloads if event["type"] == "workspace_command"]
    representation_events = [event for event in payloads if event["type"] == "representation_changed"]

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["workspace_command_action"] == "expand"
    assert telemetry["workspace_command_executed"] is True
    assert telemetry["llm_calls"] == 0
    assert command_events[-1]["target_ref"] == email_task.active_artifact_id
    assert representation_events[-1]["major_object_ids"] == [email_task.active_artifact_id]
    assert engine._work_session.attention.focus_object_id == email_task.active_artifact_id
    assert response != "Certo."
    assert ollama.calls == []


@pytest.mark.parametrize("message", ["aumenta o mail", "aumenta o email", "expande o email", "maximiza o email", "faz o email maior"])
def test_expand_email_variants_resolve_existing_task_artifact(tmp_path: Path, message: str) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos escrever um mail.")
    email_task = engine._work_session.active_task
    assert email_task is not None

    engine.respond(message)
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["workspace_command_action"] == "expand"
    assert telemetry["llm_calls"] == 0
    assert engine._work_session.representation.major_object_ids == [email_task.active_artifact_id]
    assert ollama.calls == []


@pytest.mark.parametrize("message", ["maximiza tudo", "abre todos", "mostra todos", "expande todos", "traz tudo para cima"])
def test_global_workspace_command_variants_do_not_use_llm(tmp_path: Path, message: str) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos comparar Samsung com iPhone.")

    engine.respond(message)
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["workspace_command_scope"] == "all"
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []


def test_visual_focus_command_does_not_resume_paused_activity(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos escrever um mail.")
    engine.respond("É para o João.")
    email_task = engine._work_session.active_task

    engine.respond("Enquanto isso, vamos comparar Samsung com iPhone.")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None
    assert email_task is not None and email_task.id != comparison_task.id

    engine.respond("Mostra o email.")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["attention_command_action"] == "open_workspace_reference"
    assert engine._work_session.active_task.id == comparison_task.id
    assert any(task.id == email_task.id for task in engine._work_session.paused_task_stack)
    assert ollama.calls == []


def test_resume_activity_command_switches_active_task_and_next_turn_continues_email(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos escrever um mail.")
    engine.respond("É para o João.")
    email_task = engine._work_session.active_task
    assert email_task is not None

    engine.respond("Enquanto isso, vamos comparar Samsung com iPhone.")
    comparison_task = engine._work_session.active_task
    assert comparison_task is not None

    engine.respond("Volta ao email.")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "TASK_RESUME"
    assert telemetry["task_resumed"] is True
    assert engine._work_session.active_task.id == email_task.id
    assert any(task.id == comparison_task.id for task in engine._work_session.paused_task_stack)
    assert engine._work_session.attention.focus_object_id == email_task.active_artifact_id

    engine.respond("O João é o meu professor de Cálculo e quero perguntar quando é a entrega do trabalho 1.")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "TASK_CONTINUATION"
    assert engine._work_session.active_task.id == email_task.id
    assert engine._work_session.active_task.parameters["course"] == "Cálculo"
    assert engine._work_session.active_task.parameters["purpose"] == "request_clarification"
    assert "trabalho 1" in engine._work_session.active_task.parameters["topic"]
    assert ollama.calls == []


def test_equivalent_generic_comparison_is_resumed_not_duplicated(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Estou a pensar comprar um telefone novo, vamos comparar Samsung com iPhone.")
    comparison_task = engine._work_session.active_task
    comparison_artifact_id = comparison_task.active_artifact_id

    engine.respond("Vamos escrever um mail.")
    email_task = engine._work_session.active_task
    assert email_task is not None and email_task.id != comparison_task.id

    engine.respond("Enquanto isso, vamos comparar Samsung com iPhone.")
    telemetry = _telemetry(engine)

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["task_resumed"] is True
    assert telemetry["generic_activity_reused"] is True
    assert engine._work_session.active_task.id == comparison_task.id
    assert engine._work_session.active_task.active_artifact_id == comparison_artifact_id
    comparison_artifacts = [
        obj for obj in engine._work_session.objects.values()
        if obj.kind == "task_artifact" and obj.payload.get("task_type") == "generic_comparison"
    ]
    assert len(comparison_artifacts) == 1
    assert any(task.id == email_task.id for task in engine._work_session.paused_task_stack)
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []


def test_simple_add_is_not_partial_execution(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos comparar Samsung com iPhone.")

    response = engine.respond("Adiciona o Pixel.")
    telemetry = _telemetry(engine)

    assert response == "Acrescentei Pixel."
    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["partial_execution"] is False
    assert telemetry["requested_intent_count"] == 1
    assert telemetry["executed_intent_count"] == 1
    assert telemetry["failed_intent_count"] == 0
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []


@pytest.mark.parametrize("message", ["mantém Açores visível", "deixa Açores aberto", "não escondas Açores"])
def test_attention_keep_visible_paraphrases_share_the_same_route(tmp_path: Path, message: str) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("Vamos comparar Suécia, Itália e Açores.")
    engine.consume_ui_events()

    engine.respond(message)
    telemetry = _telemetry(engine)

    assert telemetry["canonical_interpretation_primary_route"] == "ATTENTION_COMMAND"
    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["llm_calls"] == 0
    target_id = telemetry["attention_command_target_ids"][0]
    assert engine._work_session.objects[target_id].title == "Açores"
    assert target_id in engine._work_session.constraints.keep_visible_ids
    assert ollama.calls == []


def test_interaction_interpreter_does_not_import_conversation() -> None:
    source = Path("assistant/interaction_interpreter.py").read_text(encoding="utf-8")

    assert "from assistant import conversation" not in source
    assert "_conv." not in source
    assert "assistant.conversation" not in source
