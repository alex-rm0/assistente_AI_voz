"""Guards for execution claims, distinct from drafts and quoted content."""
import re

from assistant.text_matching import contains_any_phrase, normalize_for_matching


def unquoted_text(response):
    return normalize_for_matching(re.sub(r'"[^"\n]*"|“[^”\n]*”|`[^`]*`', "", response)).replace("’", "'")


def unsupported_send_claim(response, tools_used=()):
    if any(name in {"send_email", "send_mail"} for name in tools_used):
        return ""
    text = unquoted_text(response)
    for sentence in re.split(r"[.!?;\n]", text):
        if re.search(r"\b(?:nao|nunca|not|never|haven't|hasn't|didn't|don't)\b", sentence):
            continue
        match = re.search(
            r"\b(?:enviei|mandei|i(?:'ve| have)?\s+(?:already\s+)?(?:sent|emailed|forwarded)|"
            r"(?:o (?:email|mail)|a mensagem)\s+(?:ja\s+)?(?:foi|esta)\s+enviad[oa]|"
            r"(?:the (?:email|mail|message)|it)\s+(?:has been|was|is)\s+(?:already\s+)?sent)\b", sentence)
        if match:
            return match[0]
    return ""


def send_not_executed_response(language):
    return ("I haven't sent the email. I can prepare the draft, but I can't send emails."
            if language == "en-GB" else
            "Não enviei o email. Posso preparar o rascunho, mas não consigo enviar emails.")


_REQUESTS = ("abre", "abrir", "mostra", "mostrar", "foca", "minimiza", "compacta", "esconde", "oculta")
_SUCCESS = ("i opened", "esta a abrir", "abri", "ja abri", "minimizei", "minimizado", "escondi", "foquei", "esta em foco")
_OPEN_REQUEST_RE = re.compile(r"^(?:please )?open\s|\b(?:can|could|would) you (?:please )?open\b|\bplease open\b")


def detect_ungrounded_workspace_operation_claim(user_message, response):
    write_claim = re.search(
        r"\b(?:i(?:'ve| have)?\s+(?:saved|stored|created|added|written)|"
        r"(?:it|the (?:task|file|email|note|draft))\s*(?:'s|is|was|has been)\s+(?:saved|stored|created|added))\b",
        unquoted_text(response),
    )
    if write_claim:
        return write_claim[0]
    message = normalize_for_matching(user_message)
    if not (contains_any_phrase(message, _REQUESTS) or _OPEN_REQUEST_RE.search(message.strip())):
        return ""
    text = normalize_for_matching(response)
    return next((marker for marker in _SUCCESS if marker in text), "")
