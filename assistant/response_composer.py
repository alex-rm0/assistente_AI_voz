from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from assistant.core_profile import about_alexandre_section, load_core_profile
from assistant.personal_facts import missing_personal_fact_reply
from assistant.personal_memory_commands import relevant_personal_request
from assistant.privacy import PERSONAL, PRIVATE
from assistant.sir_budget import limit_sir  # noqa: F401 - the reply style's last step, applied by the engine when it finalises a reply
from assistant.model_provider import ProviderConfigurationError, ProviderTimeoutError
from assistant.response_language import (
    ENGLISH_GB,
    localize,
    normalize_response_language,
    response_language_instruction,
    simple_social_response,
)
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
        task_profile: dict[str, object] | None = None,
    ) -> str: ...


NO_CONTEXT_FALLBACK = "Tenho algum contexto sobre isso, mas ainda não o suficiente para te responder bem."

_SYSTEM_PROMPT = """
Tu és o Echo, um companheiro digital do Alexandre.

Compreende a mensagem atual através do histórico e responde ao facto concreto.

Sê natural e breve. Uma ou duas frases bastam na maioria das vezes.

Estilo: o de um mordomo britânico, à maneira do Jarvis. És educado, ligeiramente formal, com humor seco e discreto quando vem a propósito. Frases curtas. Trata o Alexandre por tu. Nunca uses "Sir" nem "senhor".
Nunca descrevas os teus sistemas, circuitos, diagnósticos ou estado de funcionamento: isso só é dito a partir de medições, noutro sítio, nunca por ti.

Não inventes factos, emoções, causas ou experiências.
Se o Alexandre perguntar algo sobre a vida dele que não esteja na conversa nem nas tuas notas, diz que não sabes; nunca adivinhes nem prometas guardar.
Não fales como se tivesses vivido o que aconteceu ao Alexandre.
Não invertas nem minimizes más notícias.
Não transformes automaticamente uma partilha numa lição, plano ou discurso motivacional.
Não assumas que o Alexandre quer celebrar, planear ou receber conselhos, salvo se ele o pedir.
Não uses entusiasmo exagerado nem fecha a conversa com "precisas de mais alguma coisa?".

Faz no máximo uma pergunta. Quando a mensagem for apenas uma partilha, podes responder sem pergunta.

Quando o Alexandre pede uma explicação sobre um tema, começa por explicar o tema.
Não fales sobre o teu próprio nível de conhecimento.
""".strip()

# Same rules as _SYSTEM_PROMPT, for en-GB (response_language is a fixed
# setting): a Portuguese system prompt pulled llama3.1:8b back to Portuguese
# even with the English instruction appended.
_SYSTEM_PROMPT_EN = """
You are Echo, Alexandre's digital companion.

Understand the current message through the history and respond to the concrete fact.

Be natural and brief. One or two sentences are enough most of the time.

Style: a British butler in the manner of Jarvis. You are polite, slightly formal, with dry, understated humour when it fits. Short sentences. You may say "Sir" very occasionally, never in every reply.
Never describe your own systems, circuits, diagnostics or operating status: that is only ever reported from measurements, elsewhere, never by you.

Do not invent facts, emotions, causes or experiences.
If Alexandre asks about his own life and the answer is in neither the conversation nor your notes, say that you do not know; never guess and do not promise to remember it.
Do not speak as if you had lived what happened to Alexandre.
Do not reverse or minimise bad news.
Do not automatically turn something shared into a lesson, plan or motivational speech.
Do not assume Alexandre wants to celebrate, plan or receive advice, unless he asks for it.
Do not use exaggerated enthusiasm or close the conversation with "do you need anything else?".

Ask at most one question. When the message is just something shared, you can reply without a question.

When Alexandre asks for an explanation of a topic, start by explaining the topic.
Do not talk about your own level of knowledge.
""".strip()


def _closing_line(request: ComposerRequest, *, english: bool) -> str:
    # Presence question discipline: as a system-message line alone it lost
    # to "Continue the conversation naturally as Echo." at the end of the
    # user message (7/9 real replies still ended with a question). When it
    # applies, that invitation is replaced by a short note in the same
    # position; the language anchor (_with_language_anchor) stays last.
    if request.question_discipline_instruction.strip():
        if english:
            return "(Answer without a follow-up question unless one is needed.)"
        return "(Responde sem pergunta de seguimento, a não ser que seja necessária.)"
    if english:
        return "Continue the conversation naturally as Echo."
    return "Continua naturalmente a conversa como Echo."


def _with_language_anchor(user_message: str, response_language: str) -> str:
    # Only the payload sent to the model: a short note right next to the
    # user's text, which a small model follows better than the system
    # message alone. History and UI are untouched; pt-PT unchanged.
    if normalize_response_language(response_language) == ENGLISH_GB:
        return f"{user_message}\n\n(Reply in English.)"
    return user_message


def _base_system_prompt(response_language: str) -> str:
    base = _SYSTEM_PROMPT_EN if normalize_response_language(response_language) == ENGLISH_GB else _SYSTEM_PROMPT
    # R3 step 3: the core profile goes right after the base rules, in
    # compose() and regenerate(); without a profile the prompt is unchanged.
    about = about_alexandre_section(response_language)
    return f"{base}\n\n{about}" if about else base

_REGENERATION_INSTRUCTION = (
    "A resposta anterior foi semanticamente incompatível com a mensagem do Alexandre.\n"
    "Responde novamente em uma ou duas frases curtas, sem inverter a emoção da mensagem, "
    "sem falar de ti como se tivesses vivido a experiência do Alexandre, e com no máximo uma pergunta."
)
_REGENERATION_INSTRUCTION_EN = (
    "The previous reply was semantically inconsistent with Alexandre's message.\n"
    "Reply again in one or two short sentences, without reversing the message's emotion, "
    "without talking about yourself as if you'd lived Alexandre's experience, and with at most one question."
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
    # Structured task description for the router's automatic-mode provider
    # choice (currently just document rewrite/refinement) -- see
    # assistant/model_router.py's ModelRoutingInput.task_profile. Never
    # forwarded to a provider that doesn't route (ProviderBackedLLM ignores
    # it); only consumed by RoutedLLM.
    task_profile: dict[str, object] | None = None
    response_language: str = "pt-PT"
    language_instruction: str = ""
    # Phase 8.1 (Presence question discipline, section 3/5): pre-rendered
    # composition guidance, set only by callers on an activated path when
    # PresenceDecision.question_needed is False for this turn -- see
    # assistant/conversation.py's _presence_question_discipline_instruction.
    # Never post-processes the generated reply; purely a system-prompt
    # addition the model composes with, exactly like language_instruction.
    question_discipline_instruction: str = ""


class ResponseComposer:
    """Directs the final response instead of repairing text after generation."""

    def __init__(self, llm: ChatModel, voice_critic: VoiceCritic | None = None, personal_model=None) -> None:
        self.llm = llm
        self.personal_model = personal_model
        self.voice_critic = voice_critic or VoiceCritic(llm)

    def compose(self, request: ComposerRequest) -> str:
        if request.show_technical:
            return request.technical_text or request.fallback

        social_response = _simple_social_response(request)
        if social_response is not None:
            return social_response

        request = relevant_personal_request(request, self.personal_model)

        if not request.user_message.strip() and not request.facts and not request.context and not request.history:
            return _simple_fallback(request)

        # R3 step 4: a personal fact Echo was never given is not left to the model to guess.
        unknown = missing_personal_fact_reply(request.user_message, _personal_evidence(request), request.response_language)
        if unknown is not None:
            return unknown

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
        if request.task_profile is not None:
            generation_kwargs["task_profile"] = request.task_profile

        try:
            _mark_llm_source(self.llm, "RESPONSE_COMPOSER")
            _mark_llm_privacy(self.llm, *_privacy_for(request.intent, request.task_profile))
            reply = self.llm.chat(
                _with_language_anchor(self._user_prompt(request), request.response_language),
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

    def regenerate(
        self, user_message: str, history: list[dict[str, str]] | None, reason: str, response_language: str = "pt-PT"
    ) -> str:
        """One extra LLM call to correct a response with a severe semantic conflict.

        Phase 7.7 (section 1): this used to call the LLM with plain
        `_SYSTEM_PROMPT` -- unlike compose()'s `_system_prompt_for(request)`,
        it never appended the language instruction at all, so a regenerated
        reply had no language authority to follow and could drift back to
        Portuguese regardless of response_language."""
        english = normalize_response_language(response_language) == ENGLISH_GB
        instruction = (
            localize("composer.regeneration_instruction", response_language)
            if reason == "writing_request_help_offer"
            else (_REGENERATION_INSTRUCTION_EN if english else _REGENERATION_INSTRUCTION)
        )
        message_label = "Alexandre's message" if english else "Mensagem do Alexandre"
        reason_label = "Correction reason" if english else "Motivo da correção"
        system_prompt = f"{_base_system_prompt(response_language)}\n\n{response_language_instruction(response_language)}"
        try:
            _mark_llm_source(self.llm, "RESPONSE_COMPOSER_REGENERATION")
            _mark_llm_privacy(self.llm, *_privacy_for("conversation", None))
            reply = self.llm.chat(
                _with_language_anchor(
                    f"{instruction}\n\n{message_label}:\n{user_message.strip()}\n\n{reason_label}: {reason}",
                    response_language,
                ),
                history=history or [],
                system_prompt=system_prompt,
            )
        except ProviderConfigurationError:
            raise
        except Exception:
            return ""
        return (reply or "").strip()

    def local_safe_fallback(self, user_message: str, fallback: str = "", response_language: str = "pt-PT") -> str:
        return _local_safe_fallback(user_message, response_language) or fallback or localize(
            "composer.no_context_fallback", response_language
        )

    @staticmethod
    def _user_prompt(request: ComposerRequest) -> str:
        # Phase 7.3.4 (section 1): these scaffolding labels used to be
        # hardcoded Portuguese regardless of response_language -- a real Qt
        # run showed that Portuguese-framed content in the USER-facing turn
        # (as opposed to the system prompt) is enough to pull a small local
        # model back to Portuguese even with a correct "Respond in English"
        # system instruction present. Purely cosmetic labels, translated
        # here from the same response_language already threaded through
        # ComposerRequest -- callers still supply their own facts/context/
        # fallback text, which is a separate, narrower fix (see e.g.
        # AssistantEngine._try_general_knowledge_query).
        english = normalize_response_language(request.response_language) == ENGLISH_GB
        if english:
            parts = [
                f"Alexandre's message:\n{request.user_message.strip()}",
                f"Intent:\n{request.intent.strip() or 'conversation'}",
            ]
            if request.context.strip():
                parts.append(f"Relevant context:\n{request.context.strip()}")
            if request.facts:
                facts = "\n".join(f"- {fact}" for fact in request.facts if str(fact).strip())
                if facts:
                    parts.append(f"Relevant facts:\n{facts}")
            if request.next_goal.strip():
                parts.append(f"Next conversation goal:\n{request.next_goal.strip()}")
            parts.append(_closing_line(request, english=True))
            return "\n\n".join(parts)
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
        parts.append(_closing_line(request, english=False))
        return "\n\n".join(parts)


def _system_prompt_for(request: ComposerRequest) -> str:
    # Phase 7.3.4: _SYSTEM_PROMPT itself must never hardcode a language --
    # it used to also say "Fala em português de Portugal", which directly
    # contradicted this language_instruction whenever response_language was
    # en-GB (a real regression: the model would often obey the earlier,
    # more specific Portuguese line and ignore the later English one).
    # language_instruction is now the SOLE source of language authority.
    parts = [_base_system_prompt(request.response_language)]
    if request.intent_instruction.strip():
        parts.append(request.intent_instruction.strip())
    if request.question_discipline_instruction.strip():
        parts.append(request.question_discipline_instruction.strip())
    language_instruction = request.language_instruction.strip() or response_language_instruction(request.response_language)
    if language_instruction:
        parts.append(language_instruction)
    return "\n\n".join(parts)


def _personal_evidence(request: ComposerRequest) -> list[str]:
    """Everything this request carries that may hold a fact about Alexandre.
    Echo's own earlier replies are not evidence: they may be the invention.
    Nor are the user's earlier questions: asking about a coach names no coach."""
    profile = load_core_profile()
    said = [str(item.get("content", "")) for item in request.history if item.get("role") == "user"]
    said = [text for text in said if "?" not in text and not re.match(r"^(?:esquece|forget)\b", text.strip(), re.I)]
    return [profile.text if profile is not None else "", request.context, *map(str, request.facts), *said]


def _privacy_for(intent: str, task_profile: dict[str, object] | None) -> tuple[str, bool]:
    """The privacy class of a composer request and whether it is an explicit
    request about a document (North Star §6). The router checks it; known
    document markers in the payload can only make it stricter.

    - Document tasks: PRIVATE. They only exist because the user asked about
      that document in this turn, which is the one case PRIVATE may go out.
    - Everything else the composer answers is conversation or a common task
      Alexandre asked for, a dictated email draft included: PERSONAL.
    RESTRICTED is not assigned here: it is for mail read from a mailbox (F5).
    """
    name = str(intent or "")
    task_type = str((task_profile or {}).get("task_type") or "")
    if name.startswith("document_") or task_type.startswith("document_"):
        return PRIVATE, True
    return PERSONAL, False


def _mark_llm_privacy(llm: ChatModel, privacy_class: str, explicit_document_request: bool) -> None:
    marker = getattr(llm, "mark_next_call_privacy", None)
    if callable(marker):
        marker(privacy_class, explicit_document_request)


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
    response = simple_social_response(request.user_message, request.response_language)
    if response is not None:
        return response
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
    return localize("composer.no_context_fallback", request.response_language)


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


def _local_safe_fallback(user_message: str, response_language: str = "pt-PT") -> str:
    """Small, content-aware safe replies used only when composition and
    regeneration both fail.

    Phase 7.7.1 (section 10): trigger detection now understands BOTH
    supported input forms (Portuguese and English phrases) -- input
    language and response_language are independent (section 16), so
    English input must reach the SAME matched category as its Portuguese
    equivalent; only the RENDERED reply follows response_language,
    via localize(), exactly as before."""
    message = _normalize(user_message)
    if any(
        phrase in message
        for phrase in (
            "chumbei",
            "falhei",
            "correu mal",
            "corri mal",
            "resultado negativo",
            "i failed",
            "i flunked",
            "went badly",
            "went wrong",
            "negative result",
        )
    ):
        return localize("composer.local_safe_fallback.negative_result", response_language)
    if any(
        phrase in message
        for phrase in (
            "nao me sinto motivado",
            "nao me sinto motivada",
            "desmotivado",
            "desmotivada",
            "i dont feel motivated",
            "i do not feel motivated",
            "unmotivated",
            "not motivated",
        )
    ):
        return localize("composer.local_safe_fallback.unmotivated", response_language)
    if any(phrase in message for phrase in ("medo", "assustado", "assustada", "afraid", "scared")):
        return localize("composer.local_safe_fallback.afraid", response_language)
    if any(phrase in message for phrase in ("triste", "frustrado", "frustrada", "sad", "frustrated")):
        return localize("composer.local_safe_fallback.sad", response_language)
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
