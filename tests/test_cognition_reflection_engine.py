from assistant.cognition.context_builder import ContextBundle
from assistant.cognition.intent_engine import IntentResult
from assistant.cognition.reflection_engine import ReflectionEngine


def test_reflection_asks_for_context_when_study_request_is_vague() -> None:
    context = ContextBundle(
        user_message="Ajuda-me a estudar.",
        intent=IntentResult("planear_estudo", 0.88, "preparar plano de estudo", "pedido amplo"),
    )

    reflection = ReflectionEngine().reflect(context)

    assert reflection.needs_more_context
    assert any("disciplina" in question.lower() for question in reflection.questions)
    assert any("tempo" in item.lower() for item in reflection.missing_context)


def test_reflection_uses_session_context_without_listing_logs() -> None:
    context = ContextBundle(
        user_message="Onde ficámos?",
        intent=IntentResult("retomar_contexto", 0.92, "reconstruir continuidade", "pedido de continuidade"),
        session_facts=["Da última vez trabalhámos no Personal Model."],
    )

    reflection = ReflectionEngine().reflect(context)

    assert not reflection.needs_more_context
    assert any("continuidade" in insight.lower() for insight in reflection.insights)
