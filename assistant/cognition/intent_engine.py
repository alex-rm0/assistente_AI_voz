from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from assistant.text_matching import contains_phrase, find_evidence_span


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float
    goal: str
    reason: str
    # The literal marker phrase that justified this intent, "" for the
    # conversa_normal default (there is no positive evidence for that one —
    # it's what's left when nothing else matched). See Falha 2 of the
    # ferro/erro follow-up: AGENT/PROBLEM_SOLVING must never be selected
    # without a real, word-boundary-safe evidence span.
    evidence_span: str = ""


class IntentEngine:
    """Infers what Alexandre is really trying to do.

    Every branch below is a real word/phrase match (see assistant.text_matching),
    never a raw substring check — "erro" must never fire on "ferro", "que"
    must never fire on "Quero". A branch with no matching evidence_span is a
    bug, not a feature: it means some intent got selected from mere
    word-fragment co-occurrence instead of real evidence (see Falha 2 of the
    ferro/erro follow-up task, where "negociar um protocolo... braço de
    ferro" was misclassified as resolver_problema_tecnico this way).
    """

    def analyse(self, user_message: str) -> IntentResult:
        text = _normalize(user_message)

        span = find_evidence_span(text, ("onde ficamos", "onde ficámos", "ultima sessao", "última sessão"))
        if span:
            return IntentResult(
                intent="retomar_contexto",
                confidence=0.92,
                goal="reconstruir continuidade da sessão anterior",
                reason="o pedido procura saber onde o trabalho ficou interrompido",
                evidence_span=span,
            )

        span = find_evidence_span(text, ("o que sabes sobre mim", "o que sabes de mim", "quem sou eu"))
        if span:
            return IntentResult(
                intent="explorar_identidade",
                confidence=0.94,
                goal="interpretar o Personal Model sem listar memórias",
                reason="o utilizador pergunta pelo conhecimento acumulado sobre si",
                evidence_span=span,
            )

        span = find_evidence_span(
            text,
            (
                "ajuda-me a estudar",
                "ajuda me a estudar",
                "preciso de estudar",
                "quero estudar",
                "tenho um exame",
                "tenho exame",
                "exame de",
                "nervoso para um exame",
                "nervoso para o exame",
                "nervosa para um exame",
                "nervosa para o exame",
            ),
        )
        if span:
            return IntentResult(
                intent="planear_estudo",
                confidence=0.88,
                goal="preparar um plano de estudo ajustado ao contexto real",
                reason="o pedido está relacionado com estudo ou exame e ainda precisa de contexto concreto",
                evidence_span=span,
            )

        span = find_evidence_span(text, ("ferias", "férias", "viagem", "viajar"))
        if span and _contains_any(text, "planear", "planeia", "organizar", "ajuda", "preparar", "queria", "gostava"):
            return IntentResult(
                intent="tomar_decisao",
                confidence=0.86,
                goal="ajudar a planear uma viagem ou férias",
                reason="o pedido envolve uma decisão com restrições ainda desconhecidas",
                evidence_span=span,
            )

        span = find_evidence_span(text, ("praia", "ida a praia", "ida à praia"))
        if span and _contains_any(text, "ajuda", "organizar", "planear", "preparar"):
            return IntentResult(
                intent="tomar_decisao",
                confidence=0.82,
                goal="ajudar a organizar uma ida com contexto suficiente",
                reason="existe um pedido explícito de planeamento, não apenas conversa casual",
                evidence_span=span,
            )

        span = find_evidence_span(
            text, ("portatil", "portátil", "laptop", "casa", "apartamento", "carro", "automovel", "automóvel")
        )
        if span and _contains_any(text, "escolher", "comprar", "ajuda", "procurar", "queria", "gostava", "preciso"):
            return IntentResult(
                intent="tomar_decisao",
                confidence=0.86,
                goal="ajudar a tomar uma decisão personalizada",
                reason="o pedido envolve uma escolha que depende de preferências pessoais",
                evidence_span=span,
            )

        span = find_evidence_span(
            text,
            (
                "programar",
                "python",
                "erro",
                "bug",
                "codigo",
                "código",
                "refatorar",
                "arquitetura",
                "architecture",
                "assistente",
            ),
        )
        if span:
            return IntentResult(
                intent="resolver_problema_tecnico",
                confidence=0.84,
                goal="ajudar a compreender e resolver um problema técnico",
                reason="o pedido menciona programação, erro ou trabalho técnico",
                evidence_span=span,
            )

        span = find_evidence_span(text, ("tarefa", "tarefas", "lembrete", "lembra-me", "tenho de"))
        if span:
            return IntentResult(
                intent="gerir_tarefas",
                confidence=0.86,
                goal="criar, consultar ou alterar tarefas com base na fonte de verdade",
                reason="o pedido está relacionado com tarefas ou lembretes",
                evidence_span=span,
            )

        action_match = re.search(r"\b(abre|abrir|executa|corre|apaga|move)\b", text)
        if action_match:
            return IntentResult(
                intent="acao_operacional",
                confidence=0.78,
                goal="avaliar se existe uma ação segura e confirmável",
                reason="o pedido parece envolver uma ação local ou externa",
                evidence_span=action_match.group(0),
            )

        return IntentResult(
            intent="conversa_normal",
            confidence=0.55,
            goal="compreender o pedido e responder apenas se houver contexto suficiente",
            reason="não existe uma intenção especializada clara",
        )


def _contains_any(text: str, *phrases: str) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_marks.split())
