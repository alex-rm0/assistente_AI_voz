from __future__ import annotations

from dataclasses import dataclass, field

from assistant.cognition.context_builder import ContextBundle
from assistant.cognition.preference_builder import UserPreferenceBuilder


@dataclass(frozen=True)
class ConversationalFocus:
    conversational_focus: str = ""
    supporting_context: str = ""
    implied_tension: str = ""
    response_goal: str = ""
    should_ask: bool = False
    question_goal: str = ""
    question_budget: int = 0
    primary_move: str = ""

    @property
    def has_focus(self) -> bool:
        return bool(self.conversational_focus or self.implied_tension or self.response_goal)


@dataclass(frozen=True)
class ReflectionResult:
    insights: list[str] = field(default_factory=list)
    ignored_context: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    conversational_focus: ConversationalFocus = field(default_factory=ConversationalFocus)

    @property
    def needs_more_context(self) -> bool:
        return bool(self.questions)


class ReflectionEngine:
    """Turns gathered context into insights and uncertainty.

    It never writes the final answer. If the context is insufficient, it
    produces questions that the Response Composer can turn into natural text.
    """

    def reflect(self, context: ContextBundle) -> ReflectionResult:
        intent_name = context.intent.intent
        facts = context.relevant_facts()
        focus = _conversational_focus_for(context)
        insights = self._insights_for(context)
        missing = self._missing_context_for(context)
        questions = self._questions_for(intent_name, missing, focus)
        ignored = self._ignored_context_for(context)

        confidence = context.intent.confidence
        if questions:
            confidence = min(confidence, 0.58)
        elif facts:
            confidence = min(0.95, max(confidence, 0.72))

        return ReflectionResult(
            insights=insights,
            ignored_context=ignored,
            missing_context=missing,
            questions=questions,
            confidence=confidence,
            conversational_focus=focus,
        )

    def _insights_for(self, context: ContextBundle) -> list[str]:
        intent_name = context.intent.intent
        insights: list[str] = []

        if context.personal_facts:
            insights.append("Há informação do Personal Model que pode ajudar, mas deve ser interpretada e não copiada.")

        if context.session_facts:
            insights.append("Existe continuidade de sessão que pode ajudar a reconstruir onde o trabalho ficou.")

        if context.pending_tasks:
            insights.append("Há tarefas pendentes que podem influenciar a resposta.")

        if intent_name == "planear_estudo":
            insights.append("O pedido é demasiado amplo para um plano útil sem saber disciplina, objetivo e tempo.")

        if intent_name == "tomar_decisao":
            insights.append("A decisão depende de restrições pessoais que ainda não estão claras.")

        if intent_name == "resolver_problema_tecnico" and not context.memory_facts and not context.session_facts:
            insights.append("Falta o erro concreto, o ficheiro ou o contexto técnico a analisar.")

        return insights

    def _missing_context_for(self, context: ContextBundle) -> list[str]:
        text = context.user_message.lower()
        intent_name = context.intent.intent
        missing: list[str] = []

        if intent_name == "planear_estudo":
            if not _mentions_study_subject(text):
                if "exame" in text:
                    missing.append("exame ou disciplina")
                else:
                    missing.append("disciplina ou tema de estudo")
            if not any(word in text for word in ("hoje", "amanhã", "hora", "horas", "semana", "dia ", "data", "quando")):
                missing.append("tempo disponível ou prazo")
            if _mentions_study_subject(text) and not any(
                word in text for word in ("apontamentos", "slides", "material", "resumos", "exercícios", "exercicios")
            ):
                missing.append("materiais de estudo disponíveis")

        if intent_name == "tomar_decisao":
            assessment = UserPreferenceBuilder().assess(context.user_message, "\n".join(context.memory_facts))
            if not assessment.enough_for_recommendation and assessment.next_question:
                missing.append(f"preferencias:{assessment.next_question}")

        if intent_name == "resolver_problema_tecnico":
            if _technical_request_needs_concrete_error(text):
                missing.append("erro concreto ou parte do código")

        return missing

    def _questions_for(self, intent_name: str, missing: list[str], focus: ConversationalFocus) -> list[str]:
        if focus.should_ask and focus.question_goal:
            return [focus.question_goal]

        if not missing:
            return []

        if intent_name == "planear_estudo":
            if "exame ou disciplina" in missing:
                return [
                    "Que exame é?",
                ]
            if "disciplina ou tema de estudo" in missing:
                return [
                    "É para que disciplina ou tema?",
                ]
            return [
                "Quando é exatamente o exame?",
                "Já tens apontamentos ou vais estudar pelos slides?",
            ]

        if intent_name == "tomar_decisao":
            preference_question = _preference_question(missing)
            if preference_question:
                return [preference_question]
            return [
                "Qual é o destino ou quais são as opções em cima da mesa?",
                "Tens datas ou orçamento aproximado?",
            ]

        if intent_name == "resolver_problema_tecnico":
            return [
                "Qual é o erro concreto ou que parte do código queres analisar primeiro?",
            ]

        return [f"Falta-me perceber melhor: {', '.join(missing)}."]

    def _ignored_context_for(self, context: ContextBundle) -> list[str]:
        ignored: list[str] = []
        if context.observed_context and context.intent.intent not in {"retomar_contexto", "resolver_problema_tecnico"}:
            ignored.append("contexto observado do computador por não ser central para este pedido")
        return ignored


def _mentions_study_subject(text: str) -> bool:
    if "exame de " in text:
        return True
    return any(
        word in text
        for word in (
            "disciplina",
            "cadeira",
            "matemática",
            "programação",
            "algorítmicas",
            "algoritmicas",
            "algoritmos",
        )
    )


def _technical_request_needs_concrete_error(text: str) -> bool:
    if any(word in text for word in ("arquitetura", "architecture", "projeto", "projecto", "assistente")):
        return False
    if any(word in text for word in ("erro", "traceback", "ficheiro", "código", "codigo", "bug")):
        return False
    return any(phrase in text for phrase in ("ajuda-me a programar", "ajuda me a programar", "ajuda-me a resolver", "ajuda me a resolver"))


def _preference_question(missing: list[str]) -> str:
    for item in missing:
        if item.startswith("preferencias:"):
            return item.split(":", 1)[1]
    return ""


def _conversational_focus_for(context: ContextBundle) -> ConversationalFocus:
    if context.intent.intent not in {"conversa_normal", "tomar_decisao", "planear_estudo"}:
        return ConversationalFocus()

    text = _normalize_text(context.user_message)

    if _mentions_resistance_after_arriving_home(text):
        return ConversationalFocus(
            conversational_focus="dúvida sobre ter energia ou vontade para começar a trabalhar",
            supporting_context="acabou de chegar a casa",
            implied_tension="quer ser produtivo, mas sente resistência",
            response_goal="ajudar a distinguir cansaço de saturação ou falta de vontade",
            should_ask=True,
            question_goal="O que te está a travar: cansaço ou falta de vontade?",
            question_budget=1,
            primary_move="perguntar",
        )

    if _mentions_document_restart_resistance(text):
        return ConversationalFocus(
            conversational_focus="resistência a voltar a pegar num documento quase terminado",
            supporting_context="o documento já está perto de concluído",
            implied_tension="o bloqueio parece estar no recomeço, não no conteúdo",
            response_goal="identificar o obstáculo real sem oferecer ferramentas prematuramente",
            should_ask=False,
            question_budget=0,
            primary_move="interpretar",
        )

    if _mentions_cognitive_overload(text):
        return ConversationalFocus(
            conversational_focus="cansaço mental depois de demasiadas atividades no mesmo dia",
            supporting_context="reuniões, treino e estudo aparecem como contexto de carga",
            implied_tension="quer continuar a render, mas já não tem cabeça",
            response_goal="apontar a sobrecarga como centro da mensagem",
            should_ask=False,
            question_budget=0,
            primary_move="interpretar",
        )

    return ConversationalFocus()


def _mentions_resistance_after_arriving_home(text: str) -> bool:
    return (
        "cheguei" in text
        and "casa" in text
        and any(word in text for word in ("trabalhar", "estudar", "fazer"))
        and any(phrase in text for phrase in ("nao sei se", "não sei se", "nao tenho muita vontade", "não tenho muita vontade"))
    )


def _mentions_document_restart_resistance(text: str) -> bool:
    return (
        any(word in text for word in ("documento", "relatorio", "relatório", "texto"))
        and any(phrase in text for phrase in ("quase pronto", "praticamente pronto", "quase acabado"))
        and any(phrase in text for phrase in ("nao me apetece", "não me apetece", "preguica", "preguiça", "voltar a pegar"))
    )


def _mentions_cognitive_overload(text: str) -> bool:
    activity_count = sum(1 for word in ("reunioes", "reuniões", "treino", "estudar", "estudo") if word in text)
    return activity_count >= 2 and any(
        phrase in text
        for phrase in (
            "ja nao consigo pensar",
            "já não consigo pensar",
            "nao consigo pensar",
            "não consigo pensar",
            "sem cabeca",
            "sem cabeça",
        )
    )


def _normalize_text(text: str) -> str:
    return text.lower()
