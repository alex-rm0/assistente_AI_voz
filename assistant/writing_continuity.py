"""Resume email work from WorkSession, never from an absence assumption."""
import re

from assistant.attention import FOCUS, set_attention
from assistant.response_language import localize
from assistant.text_matching import normalize_for_matching
from assistant.writing_corrections import current_email_object, email_objects


def is_email_resume_request(message):
    text = normalize_for_matching(message)
    return bool(re.search(r"\b(?:continuar|continua|continuamos|retoma|retomar|volta|voltar|resume|continue|pick up|back to)\b", text)
                and re.search(r"\b(?:e-?mail|mail)\b", text))


def writing_workflow_question(task, response_language="pt-PT"):
    missing = set(task.missing_fields)
    if {"project", "purpose"}.issubset(missing):
        return localize("writing.ask_project_purpose", response_language)
    for field in ("recipient", "project", "purpose"):
        if field in missing:
            return localize("writing.ask_" + field, response_language)
    if task.recipient:
        return localize("writing.confirm_recipient_draft", response_language, recipient=task.recipient)
    return localize("writing.ask_draft", response_language)


def try_resume_written_email(engine):
    session = engine._work_session
    contexts = [item for item in [session.active_task, *session.paused_task_stack]
                if getattr(item, "task_type", "") == "compose_email" and item.status not in {"completed", "cancelled"}]
    # Canonical tasks retain their existing resume executor and capacity checks.
    if contexts and session.writing_context is None:
        return None
    obj = current_email_object(session)
    if obj is None:
        if email_objects(session):
            return localize("writing.which_draft", engine.response_language)
        document = session.document_context
        if document is not None and document.prepared_action == "compose_email":
            return "Let's continue that email draft." if engine.response_language == "en-GB" else "Vamos continuar esse rascunho."
        if session.writing_context is not None:
            return writing_workflow_question(session.writing_context, engine.response_language)
        return localize("planner.resume_no_email", engine.response_language)
    if session.active_task is not None and session.active_task.task_type != "compose_email":
        engine._pause_active_task()
    attention = set_attention(session, obj.id, FOCUS, source="voice_command")
    engine._queue_ui_event("attention_changed", {"object_states": dict(attention.object_states),
                                               "focus_object_id": attention.focus_object_id, "source": attention.source})
    engine._queue_semantic_object_sync(obj)
    session.export_focus = ("writing", obj.id)
    if session.writing_context is not None:
        engine._record_writing_workflow_trace(session.writing_context)
        return writing_workflow_question(session.writing_context, engine.response_language)
    return ("The email draft is still here. What would you like to change?" if engine.response_language == "en-GB"
            else "O rascunho do email continua aqui. O que queres alterar?")
