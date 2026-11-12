"""Email and conversational presentation; canonical recipient data stays intact."""
import re

from assistant.text_matching import normalize_for_matching


def recipient_display(value, language):
    return re.sub(r"\s+e\s+", " and ", str(value)) if language == "en-GB" else str(value)


def clean_new_email_subject(draft, task):
    explicit_reply = re.match(r"\s*re\s*:", task.subject, re.I) or re.search(
        r"\b(?:reply to|respond to|responder (?:ao|a um|a))\s+(?:the |an? |o |um )?(?:e-?mail|mail)\b",
        normalize_for_matching(task.purpose))
    if explicit_reply:
        return draft
    return re.sub(r"(?im)^(\s*(?:subject|assunto):\s*)(?:re\s*:\s*)+", r"\1", draft)


def remove_unsolicited_greeting(response, history, user_message, language):
    if not history or normalize_for_matching(user_message).strip(" .!?,") in {"bom dia", "good morning"}:
        return response
    match = re.match(r"\s*(?:good morning|bom dia)[.!,:\s]+", response, re.I)
    if match:
        return response[match.end():].lstrip() or ("Hi." if language == "en-GB" else "Olá.")
    return response


def is_writing_language_feedback(message):
    text = normalize_for_matching(message).strip(" .,!?:;")
    return bool(
        re.match(r"^(?:por\s*que|porque|why|how come)\b", text)
        and re.search(r"\b(?:ingles|portugues|english|portuguese)\b", text)
        and re.search(r"\b(?:escrever|escreves|responder|respondes|falar|falas|write|writing|speaking|responding|replying)\b", text)
    )
