"""Recipient intake and confirmed corrections against the canonical email."""
import re

from assistant.response_language import localize
from assistant.text_matching import normalize_for_matching


def extract_contextual_writing_recipient(message):
    raw = str(message or "").strip(" \t\r\n.,!?;:")
    if not raw:
        return ""
    normalized = normalize_for_matching(raw)
    if any(marker in normalized for marker in ("professor", "patrocinador", "cliente", "colega")):
        return ""
    spelling = re.fullmatch(
        r"(.+?)[,.;!?\s]+([\wÀ-ÿ-]+)\s+(?:é|e)\s+com\s+(?:a letra\s+)?([A-Za-z])", raw, re.I)
    if spelling:
        head, name, letter = spelling.groups()
        recipient = extract_contextual_writing_recipient(head)
        if not recipient or not re.search(r"\b" + re.escape(name) + r"\b", recipient, re.I):
            return ""
        return re.sub(r"\b" + re.escape(name) + r"\b", lambda _: letter.upper() + name[1:], recipient, flags=re.I)
    match = re.search(r"^(?:(?:o\s+)?e-?mail\s+)?(?:é\s+|e\s+)?para\s+(?:o\s+|a\s+)?(.+)$", raw, flags=re.I)
    candidate = match.group(1).strip(" \t\r\n.,!?;:") if match else raw
    if not candidate or normalize_for_matching(candidate) in {"um", "uma", "o", "a", "isso", "isto", "email", "mail"}:
        return ""
    if len(candidate.split()) > 6:
        return ""
    # Names/addresses, not a second sentence of instructions. Initials remain valid.
    if not re.fullmatch(r"[\wÀ-ÿ'’@+,-]+(?:\.[\wÀ-ÿ@+-]+)*\.?", candidate.replace(" ", "")):
        return ""
    if re.search(r"\b\w{2,}\.[\s]+\w", candidate):
        return ""
    return candidate[0].upper() + candidate[1:] if candidate[0].islower() else candidate


def email_objects(session):
    return [obj for obj in session.objects.values()
            if obj.kind == "writing_task" or obj.payload.get("task_type") == "compose_email"]


def current_email_object(session):
    for context in (session.writing_context, session.active_task):
        obj = session.objects.get(getattr(context, "active_artifact_id", ""))
        if obj is not None and obj in email_objects(session):
            return obj
    focus = getattr(session.attention, "focus_object_id", "")
    emails = email_objects(session)
    return next((obj for obj in emails if obj.id == focus), emails[0] if len(emails) == 1 else None)


def try_recipient_correction(engine, message):
    session = engine._work_session
    obj = current_email_object(session)
    if obj is None:
        return None
    task = session.writing_context
    if task is not None and task.active_artifact_id != obj.id:
        task = None
    context = next((item for item in [session.active_task, *session.paused_task_stack]
                    if getattr(item, "active_artifact_id", "") == obj.id), None)
    recipient = task.recipient if task is not None else str(obj.payload.get("recipient") or getattr(context, "entities", {}).get("recipient", ""))
    text = normalize_for_matching(message).strip(" .!?,")
    names = [name for name in re.findall(r"[\wÀ-ÿ-]+", recipient) if name.casefold() not in {"e", "and", "o", "a"}]
    mentioned = [name for name in names if re.search(r"\b" + re.escape(normalize_for_matching(name)) + r"\b", text)]
    spelling = re.search(r"\b([\wÀ-ÿ-]+)\s+(?:é|e)\s+com\s+(?:a letra\s+)?([A-Za-z])\b", message, re.I)
    exchange = re.search(r"\b([a-z])\s+por\s+(?:(?:um|uma|a letra)\s+)?([a-z])\b", text)
    proposal = obj.payload.get("recipient_correction")
    if spelling and any(name.casefold() == spelling[1].casefold() for name in names):
        old = next(name for name in names if name.casefold() == spelling[1].casefold())
        proposal = [old, spelling[2].upper() + old[1:]]
    elif exchange and len(mentioned) == 1 and mentioned[0][0].casefold() == exchange[1]:
        proposal = [mentioned[0], exchange[2].upper() + mentioned[0][1:]]
    if proposal:
        obj.payload["recipient_correction"] = proposal
        if task is not None:
            task.proposed_recipient_correction = tuple(proposal)
        if re.fullmatch(r"(?:sim|exatamente|isso|certo|correto|yes|ok)(?:[\W]+(?:quero que corrijas|corrige|por favor))*", text):
            return _apply(engine, obj, task, recipient, *proposal)
        if text in {"nao", "no", "cancela", "cancel"}:
            obj.payload.pop("recipient_correction", None)
            if task is not None:
                task.proposed_recipient_correction = None
            return "I'll keep the original name." if engine.response_language == "en-GB" else "Mantenho o nome original."
        return localize("writing.confirm_spelling", engine.response_language, old=proposal[0], new=proposal[1])
    name_reference = re.search(r"\b(?:nome|name|destinatario|recipient)\b", text)
    complaint = bool(name_reference and (
        re.search(r"\b(?:mal|errado|errada|wrong|incorrect)\b", text)
        or re.search(r"\b(?:corrige|corrigir|troca|trocar|altera|alterar|change|correct)\b", text)))
    if complaint:
        obj.payload["recipient_name_needed"] = mentioned[0] if len(mentioned) == 1 else recipient
    old = obj.payload.get("recipient_name_needed")
    if old:
        replacement = re.fullmatch(
            r"(?:[ée]|o nome [ée]|it is|it's|the name is|n[aã]o,?\s+[ée]\s+para|the recipient is)\s+"
            r"(?:para\s+)?(?:(?:o|a)\s+)?([\wÀ-ÿ-]+(?:\s+[\wÀ-ÿ-]+){0,2})",
            message.strip(" .,!?:;"), re.I)
        if replacement and not complaint:
            new = " ".join(word[0].upper() + word[1:] for word in replacement[1].split())
            return _apply(engine, obj, task, recipient, old, new)
        return "What's the correct name?" if engine.response_language == "en-GB" else "Qual é o nome correto?"
    return None


def _apply(engine, obj, task, recipient, old, new):
    updated = re.sub(r"\b" + re.escape(old) + r"\b", lambda _: new, recipient, flags=re.I)
    obj.payload["recipient"] = updated
    for field in ("recipient_correction", "recipient_name_needed"):
        obj.payload.pop(field, None)
    if task is not None:
        task.recipient = updated
        task.proposed_recipient_correction = None
        if engine._turn_trace is not None:
            engine._record_writing_workflow_trace(task)
    for context in [engine._work_session.active_task, *engine._work_session.paused_task_stack]:
        if getattr(context, "active_artifact_id", "") == obj.id:
            context.entities["recipient"] = updated
            if context.parameters.get("generated_draft"):
                context.parameters["generated_draft"] = re.sub(r"\b" + re.escape(old) + r"\b", lambda _: new, context.parameters["generated_draft"], flags=re.I)
            engine._update_task_artifact_ui(context)
    if obj.payload.get("draft_content"):
        obj.payload["draft_content"] = re.sub(r"\b" + re.escape(old) + r"\b", lambda _: new, obj.payload["draft_content"], flags=re.I)
        obj.payload["draft_revision"] = int(obj.payload.get("draft_revision", 0)) + 1
    obj.title = localize("writing.recipient_title", engine.response_language, recipient=updated)
    engine._queue_semantic_object_sync(obj)
    return (f"I've changed {old} to {new}." if engine.response_language == "en-GB" else f"Troquei {old} por {new}.")
