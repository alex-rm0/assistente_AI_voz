from assistant.cognition.executive_function import ConversationCategory, ExecutiveFunction
from assistant.cognition.intent_engine import IntentEngine


def choose(message: str):
    intent = IntentEngine().analyse(message)
    return ExecutiveFunction().choose(message, intent)


def test_social_conversation_uses_system_1_without_memory() -> None:
    strategy = choose("Olá, como estás?")

    assert strategy.category == ConversationCategory.SOCIAL_CONVERSATION
    assert strategy.is_system_1
    assert not strategy.use_personal_model
    assert not strategy.use_session
    assert not strategy.use_tasks
    assert not strategy.use_reflection


def test_personal_model_uses_only_personal_context() -> None:
    strategy = choose("O que sabes sobre mim?")

    assert strategy.category == ConversationCategory.PERSONAL_MODEL
    assert not strategy.is_system_1
    assert strategy.use_personal_model
    assert not strategy.use_session
    assert not strategy.use_tasks


def test_session_continuity_uses_session_not_personal_model_by_default() -> None:
    strategy = choose("Onde ficámos?")

    assert strategy.category == ConversationCategory.SESSION_CONTINUITY
    assert strategy.use_session
    assert not strategy.use_personal_model
    assert strategy.use_reflection


def test_planning_questions_are_limited_to_current_goal() -> None:
    strategy = choose("Preciso de ajuda a planear umas férias.")

    assert strategy.category == ConversationCategory.PLANNING
    assert strategy.allow_clarifying_questions
    assert "orçamento" in strategy.allowed_question_topics
    assert "hobbies genéricos" in strategy.blocked_question_topics
    assert not strategy.use_personal_model


def test_problem_solving_can_use_project_context_without_personal_model_by_default() -> None:
    strategy = choose("Ajuda-me a programar este erro em Python.")

    assert strategy.category == ConversationCategory.PROBLEM_SOLVING
    assert strategy.use_observed_context
    assert strategy.use_session
    assert not strategy.use_personal_model


def test_simple_general_information_uses_light_reflection_without_memory() -> None:
    strategy = choose("Explica-me o que é uma API.")

    assert strategy.category == ConversationCategory.GENERAL_INFORMATION
    assert not strategy.is_system_1
    assert strategy.use_context_builder
    assert strategy.use_reflection
    assert strategy.use_reasoning
    assert not strategy.use_personal_model
    assert not strategy.use_long_term_memory
    assert not strategy.use_observed_context
