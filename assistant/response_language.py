from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


PORTUGUESE_PT = "pt-PT"
ENGLISH_GB = "en-GB"
SUPPORTED_RESPONSE_LANGUAGES = {PORTUGUESE_PT, ENGLISH_GB}


@dataclass(frozen=True)
class ResponseLanguageCommand:
    handled: bool
    language: str = ""
    confirmation: str = ""


def normalize_response_language(value: object) -> str:
    text = str(value or "").strip()
    if text in SUPPORTED_RESPONSE_LANGUAGES:
        return text
    clean = _normalize(text)
    if clean in {"en", "english", "en-gb", "en gb", "ingles", "inglês"}:
        return ENGLISH_GB
    if clean in {"pt", "pt-pt", "pt pt", "portugues", "português", "portuguese", "portuguese portugal"}:
        return PORTUGUESE_PT
    return ""


def legacy_language_value(language: str) -> str:
    return "en" if normalize_response_language(language) == ENGLISH_GB else PORTUGUESE_PT


def response_language_instruction(language: str) -> str:
    normalized = normalize_response_language(language) or PORTUGUESE_PT
    if normalized == ENGLISH_GB:
        return (
            "High-priority response language:\n"
            "- Respond in English.\n"
            "- Prefer natural British English spelling and phrasing.\n"
            "- Do not mix Portuguese into the answer unless the user explicitly asks for it.\n"
            "- If the user asks to switch back to Portuguese, obey that language command immediately."
        )
    return (
        "Preferências de idioma:\n"
        "- idioma_base = pt-PT.\n"
        "- response_language = pt-PT.\n"
        "- Responde em português de Portugal.\n"
        "- Trata o Alexandre por tu.\n"
        "- Vocabulário pt-PT obrigatório quando responderes em português: "
        "aplicações, ecrã, acompanhar/observar, ficheiros, aceder.\n"
        "- Evita português do Brasil: aplicativos, tela, assistindo, arquivos, acessar."
    )


def response_language_status(language: str) -> str:
    return "English" if normalize_response_language(language) == ENGLISH_GB else "Português"


def detect_response_language_command(message: str) -> ResponseLanguageCommand:
    text = _normalize(message).strip(" .,!?:;")
    if not text:
        return ResponseLanguageCommand(False)
    if _asks_for_english(text):
        return ResponseLanguageCommand(True, ENGLISH_GB, "Sure. I'll respond in English.")
    if _asks_for_portuguese(text):
        return ResponseLanguageCommand(True, PORTUGUESE_PT, "Claro. Vou responder em português.")
    return ResponseLanguageCommand(False)


def simple_social_response(message: str, language: str) -> str | None:
    text = _normalize(message).strip(" .,!?:;")
    text = re.sub(r"[,.!?:;]+", " ", text)
    text = " ".join(text.split())
    if not text:
        return None
    english = normalize_response_language(language) == ENGLISH_GB
    if text in {"ola", "olá", "hello", "hi", "hey"}:
        return "Hi. How are you?" if english else "Olá! Como estás?"
    if text in {"viva tudo bem contigo", "ola tudo bem contigo", "olá tudo bem contigo", "tudo bem contigo"}:
        return "All good. How about you?" if english else "Tudo bem. E contigo?"
    if text in {
        "ola como estas",
        "olá como estás",
        "como estas",
        "como estás",
        "tudo bem",
        "how are you",
        "how are you doing",
    }:
        return "I'm good. How about you?" if english else "Estou bem. E tu?"
    if text in {"tambem estou bem", "também estou bem", "tambem estou bem obrigado", "também estou bem obrigado"}:
        return "Good to hear." if english else "Ainda bem."
    if text in {"bom dia", "good morning"}:
        return "Good morning." if english else "Bom dia."
    if text in {"boa tarde", "good afternoon"}:
        return "Good afternoon." if english else "Boa tarde."
    if text in {"boa noite", "good evening", "good night"}:
        return "Good evening." if english else "Boa noite."
    if text in {"obrigado", "obrigada", "muito obrigado", "muito obrigada", "thanks", "thank you"}:
        return "You're welcome." if english else "De nada."
    if text in {"ok obrigado", "ok obrigada", "esta bem obrigado", "está bem obrigado"}:
        return "You're welcome." if english else "De nada."
    if text in {"estas ai", "estás ai", "estas aí", "estás aí", "are you there"}:
        return "Yes, I'm here." if english else "Sim, estou aqui."
    if text in {"como te chamas", "qual e o teu nome", "qual é o teu nome", "what is your name"}:
        return "I'm Echo." if english else "Chamo-me Echo."
    return None


def looks_mixed_for_language(response: str, language: str) -> bool:
    text = _normalize(response)
    if not text:
        return False
    if normalize_response_language(language) == ENGLISH_GB:
        pt_markers = ("estou ", "estas ", "estás ", "contigo", "portugues", "ficheiro", "aplicacoes", "ecrã", "teu ")
        en_markers = (" i ", " you ", " the ", " sure ", " how ", " english ")
        return sum(marker in f" {text} " for marker in pt_markers) >= 2 and not any(marker in f" {text} " for marker in en_markers)
    en_markers = (" i'll ", " sure ", " how about you", " i can ", " english ")
    return any(marker in f" {text} " for marker in en_markers) and any(marker in text for marker in ("estou", "vou", "posso", "contigo"))


def localize_deterministic_response(response: str, language: str) -> str:
    if normalize_response_language(language) != ENGLISH_GB:
        return response
    text = str(response or "").strip()
    normalized = _normalize(text)
    if not text:
        return text
    exact = {
        "claro. o que se passa?": "Sure. What's going on?",
        "feito.": "Done.",
        "de nada.": "You're welcome.",
        "ainda nao consegui fazer essa pesquisa — nao tenho essa ferramenta ligada a esta conversa.": (
            "I still can't do that research. I don't have that tool connected to this conversation."
        ),
        "nao tenho essa informacao guardada sobre isso.": "I don't have that information saved.",
        "nao tenho memoria suficiente para confirmar essa duracao.": "I don't have enough memory to confirm that duration.",
    }
    if normalized in exact:
        return exact[normalized]
    if normalized.startswith("boa, temos ") and " como opcoes" in normalized:
        body = re.sub(r"^boa, temos\s+", "", text, flags=re.IGNORECASE)
        body = re.sub(r"\s+como opções.*$", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s+como opcoes.*$", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s+e\s+([^,]+)$", r" and \1", body)
        return f"Good, we have {body} as options. They're already in the workspace."
    if normalized.startswith("certo. temos a comparacao entre "):
        body = re.sub(r"^Certo\.\s*Temos a comparação entre\s+", "", text)
        body = re.sub(r"\s+ativa\.$", "", body)
        return f"Right. The comparison between {body} is active."
    if normalized.startswith("ainda falta perceber"):
        return "I still need to understand one or two things before making a useful suggestion."
    if normalized.startswith("ainda nao tenho fontes suficientes"):
        return "I don't have enough reliable sources to answer that safely yet."
    if (
        normalized.startswith("nao tenho ainda uma ferramenta de pesquisa")
        or normalized.startswith("ainda nao tenho uma ferramenta de pesquisa")
        or normalized.startswith("nao tenho uma ferramenta de pesquisa")
    ):
        return "I don't have a real research tool connected yet."
    return text


def _asks_for_english(text: str) -> bool:
    if text in {"in english", "english", "english please"}:
        return True
    return any(
        phrase in text
        for phrase in (
            "prefiro ingles",
            "fala em ingles",
            "fala comigo em ingles",
            "responde em ingles",
            "resposta em ingles",
            "quero que fales ingles",
            "a partir de agora fala ingles",
            "muda o idioma para ingles",
            "speak english",
            "answer in english",
            "switch to english",
            "respond in english",
        )
    )


def _asks_for_portuguese(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "prefiro portugues",
            "fala portugues",
            "fala em portugues",
            "responde em portugues",
            "resposta em portugues",
            "volta para portugues",
            "volta ao portugues",
            "muda o idioma para portugues",
            "switch to portuguese",
            "speak portuguese",
            "answer in portuguese",
            "respond in portuguese",
        )
    )


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(without_marks.split())
