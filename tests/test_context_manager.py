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


# --- Part 2: "erro" must never match as a substring inside "ferro" ---------


def test_program_error_activates_tech_context() -> None:
    manager = ContextManager()

    names = {c.name for c in manager.identify("Tive um erro no programa.")}

    assert ContextType.TECH_CONTEXT in names


def test_arm_wrestling_does_not_activate_tech_context() -> None:
    manager = ContextManager()

    names = {c.name for c in manager.identify("Não vou fazer braço de ferro.")}

    assert ContextType.TECH_CONTEXT not in names


def test_iron_structure_does_not_activate_tech_context() -> None:
    manager = ContextManager()

    names = {c.name for c in manager.identify("A estrutura é em ferro.")}

    assert ContextType.TECH_CONTEXT not in names


def test_bare_erro_word_activates_tech_context() -> None:
    manager = ContextManager()

    names = {c.name for c in manager.identify("Ocorreu um erro.")}

    assert ContextType.TECH_CONTEXT in names
