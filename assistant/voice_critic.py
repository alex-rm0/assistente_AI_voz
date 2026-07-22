from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from assistant.response_composer import ComposerRequest


class ChatModel(Protocol):
    def chat(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        response_format: str | None = None,
    ) -> str: ...


_VOICE_CRITIC_PROMPT = """
Revê apenas erros objetivos desta resposta.

Não reescrevas a resposta.
Não alteres o significado.
Não acrescentes ou retires informação.
Não acrescentes conselhos, perguntas, emoções, interpretações ou planos.
Não mudes o sujeito nem a perspetiva.

Corrige apenas:
- português do Brasil para português de Portugal;
- mistura entre tu e você;
- colocação pronominal claramente incorreta;
- concordância;
- erro gramatical grave;
- mais de uma pergunta, mantendo apenas a mais relevante.

Se não existir um erro objetivo, devolve exatamente o original.

Devolve apenas a resposta final.
""".strip()


@dataclass(frozen=True)
class VoiceCriticTrace:
    original_response: str
    reviewed_response: str
    final_response: str
    review_changed: bool
    review_accepted: bool
    final_response_changed: bool
    rejection_reason: str = ""
    review_trigger: str = ""

    @property
    def changed(self) -> bool:
        return self.final_response_changed

    @property
    def accepted(self) -> bool:
        return self.review_accepted


class VoiceCritic:
    """Semantic style review with strict fidelity checks."""

    def __init__(self, llm: ChatModel, enabled: bool = True) -> None:
        self.llm = llm
        self.enabled = enabled
        self.last_trace = VoiceCriticTrace("", "", "", False, True, False, "", "not_run")

    def review(
        self,
        reply: str,
        request: ComposerRequest,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        return self.review_with_trace(reply, request, history=history).final_response

    def review_with_trace(
        self,
        reply: str,
        request: ComposerRequest,
        history: list[dict[str, str]] | None = None,
    ) -> VoiceCriticTrace:
        clean = (reply or "").strip()
        trigger = _review_trigger(clean, request)
        if not clean or not self.enabled or not trigger:
            trace = VoiceCriticTrace(clean, clean, clean, False, True, False, "", trigger or "not_needed")
            self.last_trace = trace
            return trace

        try:
            _mark_llm_source(self.llm, "VOICE_CRITIC")
            reviewed = self.llm.chat(
                _review_prompt(clean, request),
                history=history or [],
                system_prompt=_VOICE_CRITIC_PROMPT,
            )
        except Exception:
            trace = VoiceCriticTrace(clean, clean, clean, False, False, False, "erro na revisão", trigger)
            self.last_trace = trace
            return trace

        reviewed = (reviewed or "").strip()
        if not reviewed:
            trace = VoiceCriticTrace(clean, reviewed, clean, reviewed != clean, False, False, "revisão vazia", trigger)
            self.last_trace = trace
            _print_voice_critic_trace(trace, self.llm)
            return trace

        accepted, reason = _revision_is_faithful(clean, reviewed)
        final = reviewed if accepted else clean
        trace = VoiceCriticTrace(
            original_response=clean,
            reviewed_response=reviewed,
            final_response=final,
            review_changed=reviewed != clean,
            review_accepted=accepted,
            final_response_changed=final != clean,
            rejection_reason="" if accepted else reason,
            review_trigger=trigger,
        )
        self.last_trace = trace
        _print_voice_critic_trace(trace, self.llm)
        return trace


def _review_prompt(reply: str, request: ComposerRequest) -> str:
    return (
        f"Mensagem atual do Alexandre:\n{request.user_message.strip()}\n\n"
        f"Intencao:\n{request.intent.strip() or 'conversa'}\n\n"
        f"Resposta a rever (apenas forma, nao conteudo):\n{reply}\n\n"
        "Devolve apenas a resposta revista."
    )


def _mark_llm_source(llm: ChatModel, source: str) -> None:
    marker = getattr(llm, "mark_next_call_source", None)
    if callable(marker):
        marker(source)
    else:
        try:
            setattr(llm, "_next_call_source", source)
        except Exception:
            pass


_NEGATIVE_MESSAGE_MARKERS = (
    "chumbei",
    "correu mal",
    "corri mal",
    "falhei",
    "perdi",
    "estou frustrado",
    "estou frustrada",
    "estou triste",
    "estou desmotivado",
    "estou desmotivada",
    "nao me sinto motivado",
    "nao me sinto motivada",
    "tenho medo",
    "resultado negativo",
    "correu mesmo mal",
)

_INVERSION_REPLY_MARKERS = (
    "que alivio",
    "que bom",
    "fico contente",
    "estou contente",
    "parabens",
    "felicidades",
)


def detect_semantic_conflict(user_message: str, reply: str) -> str:
    """Detecta uma resposta emocionalmente invertida face a uma mensagem negativa."""
    message = _normalize(user_message)
    if not any(marker in message for marker in _NEGATIVE_MESSAGE_MARKERS):
        return ""
    reply_normalized = _normalize(reply)
    for marker in _INVERSION_REPLY_MARKERS:
        if marker in reply_normalized:
            return f"mensagem negativa seguida de resposta invertida ('{marker}')"
    return ""


_SUBJECT_SWAP_MARKERS = (
    "eu passei",
    "eu vivi",
    "quando encontrei",
    "quando fiz o exame",
    "o meu antigo",
    "fiquei surpreendido",
    "fiquei surpreendida",
    "surpreendido com o resultado",
    "surpreendida com o resultado",
    "estou mais assustado do que",
    "estou mais assustada do que",
    "espero poder melhorar",
    "estou contente por ter sido convidado",
    "estou contente por ter sido convidada",
    "passei por isso",
)


_SUBJECT_SWAP_EMOTION_PATTERN = re.compile(
    r"\b(fiquei|senti-me|sinto-me)\b[^.!?]{0,20}\b"
    r"(triste|contente|feliz|surpreendid[oa]|assustad[oa]|chateado|chateada|"
    r"zangad[oa]|orgulhos[oa]|deprimid[oa]|ansios[oa])\b"
)


def detect_subject_swap(reply: str) -> str:
    """Deteta o Echo a falar como se tivesse vivido a experiência do utilizador."""
    normalized_reply = _normalize(reply)
    for marker in _SUBJECT_SWAP_MARKERS:
        if marker in normalized_reply:
            return "mudou a experiência do utilizador para o Echo"
    if _SUBJECT_SWAP_EMOTION_PATTERN.search(normalized_reply):
        return "mudou a experiência do utilizador para o Echo"
    return ""


def _review_trigger(reply: str, request: ComposerRequest) -> str:
    """Only objective, clear-cut problems trigger the Voice Critic.

    Length, formality, generic motivation, or "not natural enough" are never
    reasons to call it — the Composer's own output is trusted by default, and
    small imperfections are tolerated rather than chased with style markers.
    """
    conflict = detect_semantic_conflict(request.user_message, reply)
    if conflict:
        return f"conflito_semantico:{conflict}"
    swap = detect_subject_swap(reply)
    if swap:
        return f"troca_de_sujeito:{swap}"
    normalized = _normalize(reply)
    issue = _voice_issue_trigger(normalized)
    if issue:
        return issue
    if not _allows_long_answer(request) and _has_too_many_questions_for_casual(reply, request):
        return "perguntas_a_mais"
    return ""


def _needs_review(reply: str, request: ComposerRequest) -> bool:
    return bool(_review_trigger(reply, request))


def _has_voice_issue(normalized_reply: str) -> bool:
    return bool(_voice_issue_trigger(normalized_reply))


def has_voice_issue(reply: str) -> str:
    """Public check for brasileirismos/style markers, reused to validate regenerated text."""
    return _voice_issue_trigger(_normalize(reply))


def _voice_issue_trigger(normalized_reply: str) -> str:
    """Objective, clear-cut language problems only: explicit Brazilian vocabulary
    and construction, or tu/você mixing. Deliberately excludes anything about
    tone, formality, or "sounding natural" — those are tolerated imperfections,
    not reasons to call the Voice Critic (see _review_trigger)."""
    markers = (
        "voce",
        "voces",
        "arquivo",
        "arquivos",
        "aplicativo",
        "aplicativos",
        "tela",
        "acessar",
        "usuario",
        "seus",
        "sua",
        "suas",
        "compartilhar",
        "secao",
        "estressado",
        "estressada",
        "estressante",
        "estresse",
        "esporte",
        "se sinta",
        "se sente",
        "te deu errado",
        "relacionado ao",
        "me diga",
        "me conta",
        "vai te",
        "te ajudar",
        "podes me",
    )
    for marker in markers:
        if _contains_voice_marker(normalized_reply, marker):
            return marker
    return ""


def _contains_voice_marker(text: str, marker: str) -> bool:
    if re.fullmatch(r"[a-z0-9]+", marker):
        return bool(re.search(rf"\b{re.escape(marker)}\b", text))
    return marker in text


def _has_too_many_questions_for_casual(reply: str, request: ComposerRequest) -> bool:
    if reply.count("?") <= 1:
        return False
    intent = _normalize(request.intent)
    message = _normalize(request.user_message)
    if intent in {
        "general_conversation",
        "social_path",
        "social_conversation",
        "conversa_normal",
        "normal_conversation",
    }:
        return True
    return any(
        phrase in message
        for phrase in (
            "fim de semana",
            "praia",
            "amigos",
            "estou bem",
            "correu tudo bem",
            "foi pacifica",
            "foi pacifico",
        )
    )


_MOTIVATION_MARKERS = (
    "vais conseguir",
    "podes conseguir",
    "tenta ver isto como",
    "o importante e",
    "nao desistas",
    "vai correr melhor",
    "confia em ti",
    "acredita em ti",
    "e uma oportunidade",
    "ponto de inflexao",
    "encontrar um ponto de partida",
    "deves tentar",
    "tenta novamente",
    "tenta de novo",
    "convem tentar",
    "experimenta outra vez",
)

_NEW_EMOTION_MARKERS = (
    "triste",
    "deprimido",
    "deprimida",
    "ansioso",
    "ansiosa",
    "zangado",
    "zangada",
    "chateado",
    "chateada",
    "orgulhoso",
    "orgulhosa",
    "feliz",
    "radiante",
)


def _revision_is_faithful(original: str, reviewed: str) -> tuple[bool, str]:
    """Fidelity check for a purely linguistic reviewer: protects meaning, not exact wording.

    Legitimate corrections (você->tu, idiom swaps, pronoun placement, synonyms) must be
    accepted even when they share no literal words with the original, so this does not
    check for generic keyword overlap. It only rejects changes that plausibly add new
    content: new people, events, emotions, advice, questions, or a change of who
    experienced something.
    """
    if len(reviewed.split()) > max(12, int(len(original.split()) * 1.75)):
        return False, "aumentou demasiado o comprimento"

    if reviewed.count("?") > original.count("?"):
        return False, "revisão acrescentou uma pergunta nova"
    if reviewed.count("?") > 1 and reviewed.count("?") >= original.count("?"):
        return False, "revisão manteve mais de uma pergunta"

    swap_reason = detect_subject_swap(reviewed)
    if swap_reason and not detect_subject_swap(original):
        return False, swap_reason

    original_names = _proper_names(original)
    reviewed_names = _proper_names(reviewed)
    new_names = reviewed_names - original_names
    if new_names:
        return False, "introduziu nomes próprios novos: " + ", ".join(sorted(new_names))

    original_norm = _normalize(original)
    reviewed_norm = _normalize(reviewed)
    for marker in ("chefe", "reencontro", "trabalho", "exame", "dia mau", "encontrei", "aconteceu porque"):
        if marker in reviewed_norm and marker not in original_norm:
            return False, f"introduziu acontecimento ou causa nova: {marker}"

    for marker in _MOTIVATION_MARKERS:
        if marker in reviewed_norm and marker not in original_norm:
            return False, f"introduziu motivação ou conselho não pedido: {marker}"

    for marker in _NEW_EMOTION_MARKERS:
        if marker in reviewed_norm and marker not in original_norm:
            return False, f"introduziu emoção nova: {marker}"

    if original.strip().endswith("?") and not reviewed.strip().endswith("?"):
        return False, "mudou pergunta para afirmação"
    if not original.strip().endswith("?") and reviewed.strip().endswith("?"):
        return False, "mudou afirmação para pergunta"

    conflict_reason = detect_semantic_conflict(original, reviewed)
    if conflict_reason:
        return False, conflict_reason

    return True, ""


def _proper_names(text: str) -> set[str]:
    ignored = {
        "Sim",
        "Não",
        "Nao",
        "Entendo",
        "Percebo",
        "Talvez",
        "Acho",
        "Claro",
        "Bem",
        "Então",
        "Entao",
        "Peço",
        "Peco",
        "Lamento",
        "Podes",
        "Queres",
        "Estás",
        "Estas",
        "Foi",
        "Tu",
        "Eu",
        "O",
        "A",
    }
    names: set[str] = set()
    for match in re.finditer(r"\b[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-zàáâãéêíóôõúç]{2,}\b", text):
        word = match.group(0)
        if word in ignored:
            continue
        prefix = text[: match.start()].rstrip()
        if not prefix or prefix[-1] in ".!?;:":
            continue
        names.add(word)
    return names


def _allows_long_answer(request: ComposerRequest) -> bool:
    text = _normalize(request.user_message)
    return any(
        phrase in text
        for phrase in (
            "explica em detalhe",
            "explica detalhadamente",
            "resposta detalhada",
            "faz um plano",
            "cria um plano",
            "relatorio",
            "relatório",
            "desenvolve",
            "passo a passo",
        )
    )


def _print_voice_critic_trace(trace: VoiceCriticTrace, llm: ChatModel) -> None:
    settings = getattr(llm, "settings", None)
    if not bool(getattr(settings, "debug_ollama_payload", False)):
        return
    print(
        "[VOICE CRITIC TRACE]\n"
        f"trigger={trace.review_trigger}\n"
        f"original={trace.original_response}\n"
        f"reviewed={trace.reviewed_response}\n"
        f"review_changed={'true' if trace.review_changed else 'false'}\n"
        f"review_accepted={'true' if trace.review_accepted else 'false'}\n"
        f"final_response_changed={'true' if trace.final_response_changed else 'false'}\n"
        f"reason={trace.rejection_reason}"
    )


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_marks.split())
