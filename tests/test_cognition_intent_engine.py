from assistant.cognition.intent_engine import IntentEngine


def test_intent_engine_detects_session_continuity() -> None:
    result = IntentEngine().analyse("Onde ficámos?")

    assert result.intent == "retomar_contexto"
    assert result.confidence > 0.8
    assert "continuidade" in result.goal


def test_intent_engine_detects_identity_exploration() -> None:
    result = IntentEngine().analyse("O que sabes sobre mim?")

    assert result.intent == "explorar_identidade"
    assert result.confidence > 0.8


def test_intent_engine_detects_study_planning() -> None:
    result = IntentEngine().analyse("Ajuda-me a estudar.")

    assert result.intent == "planear_estudo"
    assert "estudo" in result.reason
