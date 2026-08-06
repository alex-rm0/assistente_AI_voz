from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from assistant.model_provider import ProviderConfigurationError, ProviderTimeoutError
from assistant.voice_critic import VoiceCritic


class ChatModel(Protocol):
    def chat(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str: ...


NO_CONTEXT_FALLBACK = "Tenho algum contexto sobre isso, mas ainda não o suficiente para te responder bem."

_SYSTEM_PROMPT = """
Tu és o Echo, um companheiro digital do Alexandre.

Compreende a mensagem atual através do histórico e responde ao facto concreto.

Fala em português de Portugal e trata o Alexandre por tu.

Sê natural e breve. Uma ou duas frases bastam na maioria das vezes.

Não inventes factos, emoções, causas ou experiências.
Não fales como se tivesses vivido o que aconteceu ao Alexandre.
Não invertas nem minimizes más notícias.
Não transformes automaticamente uma partilha numa lição, plano ou discurso motivacional.
Não assumas que o Alexandre quer celebrar, planear ou receber conselhos, salvo se ele o pedir.
Não uses entusiasmo exagerado nem fecha a conversa com "precisas de mais alguma coisa?".

Faz no máximo uma pergunta. Quando a mensagem for apenas uma partilha, podes responder sem pergunta.

Quando o Alexandre pede uma explicação sobre um tema, começa por explicar o tema.
Não fales sobre o teu próprio nível de conhecimento.
""".strip()

_REGENERATION_INSTRUCTION = (
    "A resposta anterior foi semanticamente incompatível com a mensagem do Alexandre.\n"
    "Responde novamente em uma ou duas frases curtas, sem inverter a emoção da mensagem, "
    "sem falar de ti como se tivesses vivido a experiência do Alexandre, e com no máximo uma pergunta."
)


@dataclass
class ComposerRequest:
    """Input needed to direct a response in Echo's voice."""

    intent: str
    user_message: str
    history: list[dict[str, str]] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    fallback: str = NO_CONTEXT_FALLBACK
    show_technical: bool = False
    technical_text: str = ""
    next_goal: str = ""
    context: str = ""
    intent_instruction: str = ""
    # Only set by callers that need tighter, more deterministic generation
    # (currently just document rewrite) — every other intent leaves these
    # None and the provider's own defaults apply, unchanged.
    temperature: float | None = None
    num_predict: int | None = None
    # Only set by callers with their own deadline budget (currently just
    # document rewrite). When set, a ProviderTimeoutError from the LLM
    # propagates instead of being swallowed into a fallback reply, so the
    # caller can react to it specifically (see compose() below).
    timeout_seconds: float | None = None
    language_instruction: str = (
        "Preferências de idioma:\n"
        "- idioma_base = pt-PT.\n"
        "- idioma_atual = pt-PT.\n"
        "- Vocabulário pt-PT obrigatório quando responderes em português: "
        "aplicações, ecrã, acompanhar/observar, ficheiros, aceder."
    )


class ResponseComposer:
    """Directs the final response instead of repairing text after generation."""

    def __init__(self, llm: ChatModel, voice_critic: VoiceCritic | None = None) -> None:
        self.llm = llm
        self.voice_critic = voice_critic or VoiceCritic(llm)

    def compose(self, request: ComposerRequest) -> str:
        if request.show_technical:
            return request.technical_text or request.fallback

        social_response = _simple_social_response(request)
        if social_response is not None:
            return social_response

        if not request.user_message.strip() and not request.facts and not request.context and not request.history:
            return _simple_fallback(request)

        # Only passed through at all when a caller actually set them (currently
        # just document rewrite) — every other intent's chat-model stub in
        # tests never had to know about these kwargs, and still doesn't.
        generation_kwargs: dict[str, object] = {}
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        if request.num_predict is not None:
            generation_kwargs["num_predict"] = request.num_predict
        if request.timeout_seconds is not None:
            generation_kwargs["timeout_seconds"] = request.timeout_seconds

        try:
            _mark_llm_source(self.llm, "RESPONSE_COMPOSER")
            reply = self.llm.chat(
                self._user_prompt(request),
                history=request.history,
                system_prompt=_system_prompt_for(request),
                **generation_kwargs,
            )
        except ProviderConfigurationError:
            raise
        except ProviderTimeoutError:
            # Only propagates for callers that set their own timeout_seconds
            # (currently just document rewrite) -- everyone else keeps the
            # pre-existing behavior of falling back to a safe reply, exactly
            # as any other provider error already did.
            if request.timeout_seconds is not None:
                raise
            return _simple_fallback(request)
        except Exception:
            return _simple_fallback(request)

        reply = (reply or "").strip()
        if not reply:
            return _simple_fallback(request)

        if _looks_like_copied_evidence(reply, request.facts):
            return _simple_fallback(request)

        return reply

    def regenerate(self, user_message: str, history: list[dict[str, str]] | None, reason: str) -> str:
        """One extra LLM call to correct a response with a severe semantic conflict."""
        instruction = (
            "A resposta anterior ofereceu ajuda em vez de executar um pedido de escrita completo.\n"
            "Executa agora o pedido diretamente. Não perguntes se o Alexandre quer ajuda."
            if reason == "writing_request_help_offer"
            else _REGENERATION_INSTRUCTION
        )
        try:
            _mark_llm_source(self.llm, "RESPONSE_COMPOSER_REGENERATION")
            reply = self.llm.chat(
                f"{instruction}\n\nMensagem do Alexandre:\n{user_message.strip()}\n\n"
                f"Motivo da correção: {reason}",
                history=history or [],
                system_prompt=_SYSTEM_PROMPT,
            )
        except ProviderConfigurationError:
            raise
        except Exception:
            return ""
        return (reply or "").strip()

    def local_safe_fallback(self, user_message: str, fallback: str = "") -> str:
        return _local_safe_fallback(user_message) or fallback or NO_CONTEXT_FALLBACK

    @staticmethod
    def _user_prompt(request: ComposerRequest) -> str:
        parts = [
            f"Mensagem do Alexandre:\n{request.user_message.strip()}",
            f"Intenção:\n{request.intent.strip() or 'conversa'}",
        ]
        if request.context.strip():
            parts.append(f"Contexto relevante:\n{request.context.strip()}")
        if request.facts:
            facts = "\n".join(f"- {fact}" for fact in request.facts if str(fact).strip())
            if facts:
                parts.append(f"Factos relevantes:\n{facts}")
        if request.next_goal.strip():
            parts.append(f"Próximo objetivo da conversa:\n{request.next_goal.strip()}")
        parts.append("Continua naturalmente a conversa como Echo.")
        return "\n\n".join(parts)


def _system_prompt_for(request: ComposerRequest) -> str:
    parts = [_SYSTEM_PROMPT]
    if request.intent_instruction.strip():
        parts.append(request.intent_instruction.strip())
    if request.language_instruction.strip():
        parts.append(request.language_instruction.strip())
    return "\n\n".join(parts)


def _mark_llm_source(llm: ChatModel, source: str) -> None:
    marker = getattr(llm, "mark_next_call_source", None)
    if callable(marker):
        marker(source)
    else:
        try:
            setattr(llm, "_next_call_source", source)
        except Exception:
            pass


def _simple_social_response(request: ComposerRequest) -> str | None:
    text = _normalize(request.user_message).strip(" .,!?:;")
    if not text:
        return None
    if _has_relevant_content_after_greeting(text):
        return None
    if text in {"ola", "olá"}:
        return "Olá! Como estás?"
    if text in {"ola como estas", "olá como estás", "como estas", "como estás", "tudo bem"}:
        return "Estou bem. E tu?"
    if text in {"bom dia"}:
        return "Bom dia."
    if text in {"boa tarde"}:
        return "Boa tarde."
    if text in {"boa noite"}:
        return "Boa noite."
    if text in {"obrigado", "obrigada"}:
        return "De nada."
    return None


def _simple_fallback(request: ComposerRequest) -> str:
    if request.intent == "clarifying_question":
        questions = _questions_from_facts(request.facts)
        if questions:
            return " ".join(questions[:2])
    if request.fallback and request.fallback != NO_CONTEXT_FALLBACK:
        return request.fallback
    return NO_CONTEXT_FALLBACK


def _questions_from_facts(facts: list[str]) -> list[str]:
    questions: list[str] = []
    for fact in facts:
        text = str(fact or "").strip()
        if not text:
            continue
        text = re.sub(r"^Perguntas necessárias:\s*", "", text, flags=re.IGNORECASE)
        if "?" in text:
            questions.append(text)
    return questions


def _has_relevant_content_after_greeting(text: str) -> bool:
    greeting_prefixes = ("ola ", "olá ", "bom dia ", "boa tarde ", "boa noite ")
    if not text.startswith(greeting_prefixes):
        return False
    return any(
        word in text
        for word in (
            "preocupado",
            "preocupada",
            "exausto",
            "exausta",
            "nervoso",
            "nervosa",
            "preciso",
            "quero",
            "problema",
            "coisa",
        )
    )


def _local_safe_fallback(user_message: str) -> str:
    """Small, content-aware safe replies used only when composition and regeneration both fail."""
    message = _normalize(user_message)
    if any(phrase in message for phrase in ("chumbei", "falhei", "correu mal", "corri mal", "resultado negativo")):
        return "Lamento que tenha corrido mal."
    if "nao me sinto motivado" in message or "nao me sinto motivada" in message or "desmotivado" in message or "desmotivada" in message:
        return "Percebo que isso te tenha deixado desmotivado."
    if "medo" in message or "assustado" in message or "assustada" in message:
        return "Percebo que isso te esteja a pesar."
    if "triste" in message or "frustrado" in message or "frustrada" in message:
        return "Lamento. Deve ser difícil."
    return ""


def _looks_like_copied_evidence(reply: str, facts: list[str]) -> bool:
    normalized_reply = _normalize(reply)
    if not normalized_reply:
        return True
    for fact in facts:
        normalized_fact = _normalize(str(fact))
        if not normalized_fact:
            continue
        if normalized_reply == normalized_fact:
            return True
        if len(normalized_fact) > 28 and normalized_fact in normalized_reply:
            return True
    return False


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_marks.split())
