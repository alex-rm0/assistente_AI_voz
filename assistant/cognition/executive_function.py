from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from assistant.cognition.intent_engine import IntentResult
from assistant.text_matching import find_evidence_span, find_prefix_evidence_span


class ConversationCategory(str, Enum):
    SOCIAL_CONVERSATION = "SOCIAL_CONVERSATION"
    PERSONAL_MODEL = "PERSONAL_MODEL"
    SESSION_CONTINUITY = "SESSION_CONTINUITY"
    PLANNING = "PLANNING"
    PROBLEM_SOLVING = "PROBLEM_SOLVING"
    GENERAL_INFORMATION = "GENERAL_INFORMATION"
    OPERATIONAL_ACTION = "OPERATIONAL_ACTION"
    TASK_MANAGEMENT = "TASK_MANAGEMENT"


@dataclass(frozen=True)
class CognitiveStrategy:
    category: ConversationCategory
    mode: str
    reason: str
    # The literal marker phrase (or the upstream IntentResult's own
    # evidence_span) that justified this category — "" only for the
    # GENERAL_INFORMATION default, which by definition has no positive
    # evidence. Exposed as agent_route_evidence_span telemetry so a routing
    # decision can always be traced back to real text, never to mere
    # word-fragment co-occurrence (see Falha 2 of the ferro/erro follow-up).
    evidence_span: str = ""
    use_context_manager: bool = False
    use_context_builder: bool = False
    use_personal_model: bool = False
    use_session: bool = False
    use_long_term_memory: bool = False
    use_tasks: bool = False
    use_observed_context: bool = False
    use_reflection: bool = False
    use_reasoning: bool = False
    allow_clarifying_questions: bool = False
    allowed_question_topics: tuple[str, ...] = ()
    blocked_question_topics: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_system_1(self) -> bool:
        return self.mode == "SYSTEM_1"

    @property
    def enabled_sources(self) -> set[str]:
        sources: set[str] = set()
        if self.use_personal_model:
            sources.add("personal_model")
        if self.use_session:
            sources.add("session")
        if self.use_long_term_memory:
            sources.add("long_term_memory")
        if self.use_tasks:
            sources.add("tasks")
        if self.use_observed_context:
            sources.add("observed_context")
        return sources


class ExecutiveFunction:
    """Chooses which cognitive modules should participate.

    Conceptually this is Echo's System 1 / System 2 gate. It does not answer
    Alexandre and it does not produce prose; it only returns a strategy.
    """

    def choose(self, user_message: str, intent: IntentResult) -> CognitiveStrategy:
        text = _normalize(user_message)

        if _is_social_conversation(text):
            return CognitiveStrategy(
                category=ConversationCategory.SOCIAL_CONVERSATION,
                mode="SYSTEM_1",
                reason="interação social simples; memória e reflexão profunda não ajudam",
                notes=("responder de forma natural e breve",),
            )

        personal_model_span = find_prefix_evidence_span(
            text, ("lembra-te que", "lembra que", "nao te esquecas que", "guarda isto", "esquece", "corrige isto")
        )
        if intent.intent == "explorar_identidade" or personal_model_span:
            return CognitiveStrategy(
                category=ConversationCategory.PERSONAL_MODEL,
                mode="SYSTEM_2",
                reason="pedido relacionado com conhecimento explícito sobre o Alexandre",
                evidence_span=personal_model_span or intent.evidence_span,
                use_context_builder=True,
                use_personal_model=True,
                use_reflection=True,
                use_reasoning=True,
                allow_clarifying_questions=False,
            )

        session_span = find_evidence_span(
            text, ("resume a ultima sessao", "o que fizemos ontem", "o que fizemos hoje", "proximo passo")
        )
        if intent.intent == "retomar_contexto" or session_span:
            return CognitiveStrategy(
                category=ConversationCategory.SESSION_CONTINUITY,
                mode="SYSTEM_2",
                reason="pedido de continuidade; Session Reflection é a fonte principal",
                evidence_span=session_span or intent.evidence_span,
                use_context_builder=True,
                use_session=True,
                use_tasks=True,
                use_reflection=True,
                use_reasoning=True,
                allow_clarifying_questions=False,
            )

        planning_span = find_evidence_span(text, ("planear", "organizar um projeto", "organizar projeto"))
        if intent.intent in {"planear_estudo", "tomar_decisao"} or planning_span:
            return CognitiveStrategy(
                category=ConversationCategory.PLANNING,
                mode="SYSTEM_2",
                reason="pedido de planeamento; precisa apenas de contexto diretamente útil ao objetivo",
                evidence_span=planning_span or intent.evidence_span,
                use_context_manager=True,
                use_context_builder=True,
                use_long_term_memory=True,
                use_tasks=True,
                use_reflection=True,
                use_reasoning=True,
                allow_clarifying_questions=True,
                allowed_question_topics=("objetivo atual", "datas", "tempo", "orçamento", "restrições", "preferências da tarefa"),
                blocked_question_topics=("hobbies genéricos", "trabalho", "estudos gerais", "perfil pessoal sem ligação à tarefa"),
            )

        if intent.intent == "resolver_problema_tecnico":
            return CognitiveStrategy(
                category=ConversationCategory.PROBLEM_SOLVING,
                mode="SYSTEM_2",
                reason="pedido de resolução; projeto, ferramentas e workspace podem ser relevantes",
                evidence_span=intent.evidence_span,
                use_context_manager=True,
                use_context_builder=True,
                use_personal_model=False,
                use_session=True,
                use_long_term_memory=True,
                use_observed_context=True,
                use_reflection=True,
                use_reasoning=True,
                allow_clarifying_questions=True,
                allowed_question_topics=("erro", "ficheiro", "objetivo técnico", "passos para reproduzir"),
            )

        if intent.intent == "gerir_tarefas":
            return CognitiveStrategy(
                category=ConversationCategory.TASK_MANAGEMENT,
                mode="SYSTEM_2",
                reason="pedido de tarefas; usar a fonte de verdade das tarefas",
                evidence_span=intent.evidence_span,
                use_context_builder=True,
                use_tasks=True,
                use_reflection=False,
                use_reasoning=False,
            )

        if intent.intent == "acao_operacional":
            return CognitiveStrategy(
                category=ConversationCategory.OPERATIONAL_ACTION,
                mode="SYSTEM_1",
                reason="ação operacional deve seguir router, segurança e confirmação sem reflexão profunda",
                evidence_span=intent.evidence_span,
            )

        return CognitiveStrategy(
            category=ConversationCategory.GENERAL_INFORMATION,
            mode="SYSTEM_2",
            reason="conversa normal; usar reflexão leve para identificar o centro da mensagem sem carregar memória",
            use_context_builder=True,
            use_reflection=True,
            use_reasoning=True,
            allow_clarifying_questions=True,
            allowed_question_topics=("foco da mensagem", "próximo passo imediato"),
            notes=("não consultar memória nem ferramentas",),
        )


def _is_social_conversation(text: str) -> bool:
    stripped = text.strip(" .,!?:;")
    if stripped in {
        "ola",
        "olá",
        "bom dia",
        "boa tarde",
        "boa noite",
        "obrigado",
        "obrigada",
        "muito obrigado",
        "muito obrigada",
        "ate ja",
        "até ja",
        "ate logo",
        "até logo",
    }:
        return True
    if stripped in {"como estas", "como estás", "tudo bem", "estas bem", "estás bem"}:
        return True
    return stripped.startswith("ola") and any(phrase in stripped for phrase in ("como estas", "tudo bem"))


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_marks.split())
