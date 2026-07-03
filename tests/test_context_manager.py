from __future__ import annotations

from assistant.context_manager import ContextManager, ContextType


def test_travel_request_activates_travel_and_personal_contexts() -> None:
    manager = ContextManager()

    contexts = manager.identify("Ajuda-me a planear ferias para a Australia.")
    names = {context.name for context in contexts}

    assert ContextType.TRAVEL_CONTEXT in names
    assert ContextType.PERSONAL_CONTEXT in names


def test_python_error_activates_tech_and_work_contexts() -> None:
    manager = ContextManager()

    contexts = manager.identify("Tenho um erro neste projeto Python.")
    names = {context.name for context in contexts}

    assert ContextType.TECH_CONTEXT in names
    assert ContextType.WORK_CONTEXT in names


def test_context_debug_includes_reason_and_weight() -> None:
    manager = ContextManager()

    debug = manager.debug_summary(manager.identify("Tenho tarefas para organizar hoje."))

    assert "PRODUCTIVITY_CONTEXT" in debug
    assert "peso" in debug
    assert "razao" in debug
