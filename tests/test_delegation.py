from __future__ import annotations

from assistant.delegation import DelegationManager, DelegationTarget


def test_delegates_codebase_work_to_codex() -> None:
    manager = DelegationManager()

    decision = manager.decide("Implementa testes para este projeto.", "Programacao")

    assert decision.target == DelegationTarget.CODEX
    assert "Codex" in decision.prepared_prompt
    assert "Implementa testes" in decision.prepared_prompt


def test_delegates_broad_reasoning_to_chatgpt() -> None:
    manager = DelegationManager()

    decision = manager.decide("Faz uma estrategia para estudar RVCC.", "RVCC")

    assert decision.target == DelegationTarget.CHATGPT
    assert "ChatGPT" in decision.prepared_prompt


def test_delegates_external_tool_request() -> None:
    manager = DelegationManager()

    decision = manager.decide("Abre o Word para escrever isto.", "Documentos")

    assert decision.target == DelegationTarget.EXTERNAL_TOOL


def test_local_simple_request_stays_local() -> None:
    manager = DelegationManager()

    decision = manager.decide("Ola, como estas?", "Geral")

    assert decision.target == DelegationTarget.LOCAL
