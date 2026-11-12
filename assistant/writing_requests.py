"""Narrow writing entry detection, independent of conversation execution."""
import re


def incomplete_writing_request(normalized_message, batch_count=1):
    text = normalized_message.replace("-", " ").strip(" .,!?:;")
    if re.search(r"\b(?:nao|not|don't|do not)\s+(?:me\s+)?(?:quer[oa]|queria|gostava|ajud\w*|want|help)\b", text):
        return False
    if re.search(r"\b(?:acrescenta|adiciona|cria)\s+(?:mais\s+um|outro)\s+(?:e mail|email|mail)\b", text):
        return True
    if not re.search(r"\b(?:preciso|quero|queria|gostava|ajuda|ajudas|podes|consegues|help me|i need to|i want to)\b", text):
        return False
    if batch_count > 1 and re.search(r"\b(?:escrever|redigir|preparar)\b", text) and re.search(r"\b(?:emails?|e mails?|mails?|mensagens?)\b", text):
        return True
    if re.search(r"\b(?:(?:escrever|redigir|preparar)\s+(?:um|o)\s+(?:email|e mail|mail)|escrever uma mensagem|write (?:an?|the) (?:email|e mail|mail))\b", text):
        return True
    return bool(re.search(
        r"\b(?:ajuda|ajuda me|preciso|quero|podes|consegues)\b.*\b(?:escrever|redigir|preparar|mandar|enviar|compor|criar|fazer|responder)\b.*\b(?:email|mail|mensagem)\b|\bajuda(?: me)? (?:num|no|com um|com o|com uma) (?:email|mail|mensagem)\b", text))
