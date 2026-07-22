from assistant.cognition.context_builder import ContextBundle
from assistant.cognition.intent_engine import IntentResult
from assistant.cognition.reasoning_engine import ReasoningEngine
from assistant.cognition.reflection_engine import ReflectionResult


def test_reasoning_stops_to_ask_when_reflection_needs_context() -> None:
    intent = IntentResult("planear_estudo", 0.88, "preparar plano", "pedido amplo")
    context = ContextBundle("Ajuda-me a estudar.", intent)
    reflection = ReflectionResult(
        insights=["O pedido é demasiado amplo."],
        missing_context=["disciplina"],
        questions=["É para que disciplina ou tema?"],
        confidence=0.5,
    )

    reasoning = ReasoningEngine().reason(context, reflection)

    assert reasoning.needs_more_context
    assert reasoning.questions == ["É para que disciplina ou tema?"]
    assert "pergunta curta" in " ".join(reasoning.plan)


def test_reasoning_for_identity_keeps_response_for_composer() -> None:
    intent = IntentResult("explorar_identidade", 0.94, "interpretar identidade", "pedido sobre o utilizador")
    context = ContextBundle("O que sabes sobre mim?", intent, personal_facts=["Usa o Codex."])
    reflection = ReflectionResult(insights=["Há informação do Personal Model que pode ajudar."], confidence=0.8)

    reasoning = ReasoningEngine().reason(context, reflection)

    assert not reasoning.needs_more_context
    assert any("Personal Model" in item for item in reasoning.conclusions)
    assert reasoning.facts_for_composer()
