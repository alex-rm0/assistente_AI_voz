from __future__ import annotations

from dataclasses import dataclass, field

from assistant.cognition.context_builder import ContextBundle
from assistant.cognition.intent_engine import IntentResult
from assistant.cognition.reflection_engine import ConversationalFocus, ReflectionResult


@dataclass(frozen=True)
class ReasoningResult:
    intent: IntentResult
    plan: list[str] = field(default_factory=list)
    conclusions: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    needs_more_context: bool = False
    should_delegate: bool = False
    delegation_target: str = ""
    conversational_focus: ConversationalFocus = field(default_factory=ConversationalFocus)

    def facts_for_composer(self) -> list[str]:
        facts: list[str] = []
        if self.conversational_focus.has_focus:
            facts.extend(_focus_facts(self.conversational_focus))
        facts.extend(_specific_conclusions(self.conclusions))
        if self.questions:
            facts.append("Perguntas necessárias: " + " ".join(self.questions))
        return facts


class ReasoningEngine:
    """Builds a non-user-facing plan from intent, context and reflection."""

    def reason(self, context: ContextBundle, reflection: ReflectionResult) -> ReasoningResult:
        intent_name = context.intent.intent
        plan = [
            "usar a intenção como guia",
            "usar apenas contexto relevante",
            "evitar inventar quando falta informação",
        ]
        conclusions = list(reflection.insights)
        actions: list[str] = []
        should_delegate = False
        delegation_target = ""

        focus = reflection.conversational_focus
        if focus.has_focus:
            plan.append("responder ao centro da mensagem, não a cada frase")
            if focus.primary_move:
                plan.append(f"usar uma única ação conversacional: {focus.primary_move}")

        if reflection.needs_more_context:
            plan.append("fazer uma pergunta curta antes de avançar")
            conclusions.append("Ainda não há informação suficiente para uma resposta útil.")
            return ReasoningResult(
                intent=context.intent,
                plan=plan,
                conclusions=conclusions,
                questions=reflection.questions[: max(1, focus.question_budget or 1)],
                needs_more_context=True,
                conversational_focus=focus,
            )

        if intent_name == "retomar_contexto":
            plan.append("reconstruir a continuidade com base na última sessão")
            conclusions.append("A resposta deve destacar continuidade, decisões e próximo passo, sem logs.")

        if intent_name == "explorar_identidade":
            plan.append("interpretar padrões do Personal Model")
            conclusions.append("A resposta deve falar sobre significado pessoal, não sobre registos.")

        if intent_name == "resolver_problema_tecnico":
            plan.append("propor análise técnica incremental")
            should_delegate = True
            delegation_target = "Codex, se for necessário editar código"

        if intent_name == "gerir_tarefas":
            plan.append("consultar ou alterar tarefas apenas pela fonte de verdade")
            actions.append("usar ferramentas de tarefas quando houver alteração real")

        return ReasoningResult(
            intent=context.intent,
            plan=plan,
            conclusions=conclusions,
            recommended_actions=actions,
            questions=[],
            needs_more_context=False,
            should_delegate=should_delegate,
            delegation_target=delegation_target,
            conversational_focus=focus,
        )


def _focus_facts(focus: ConversationalFocus) -> list[str]:
    facts: list[str] = []
    if focus.conversational_focus:
        facts.append(f"Foco conversacional: {focus.conversational_focus}")
    if focus.supporting_context:
        facts.append(f"Contexto de apoio, não assunto principal: {focus.supporting_context}")
    if focus.implied_tension:
        facts.append(f"Tensão implícita: {focus.implied_tension}")
    if focus.response_goal:
        facts.append(f"Objetivo da resposta: {focus.response_goal}")
    if focus.primary_move:
        facts.append(f"Ação conversacional principal: {focus.primary_move}")
    if focus.question_budget:
        facts.append(f"Orçamento de perguntas: {focus.question_budget}")
    return facts


def _specific_conclusions(conclusions: list[str]) -> list[str]:
    generic_markers = (
        "objetivo provavel",
        "objetivo provável",
        "usar a intencao",
        "usar a intenção",
        "usar apenas contexto",
        "evitar inventar",
        "estrategia escolhida",
        "estratégia escolhida",
        "ainda nao ha informacao suficiente",
        "ainda não há informação suficiente",
        "compreender o pedido",
    )
    useful: list[str] = []
    for item in conclusions:
        normalized = item.lower()
        if any(marker in normalized for marker in generic_markers):
            continue
        useful.append(item)
    return useful
