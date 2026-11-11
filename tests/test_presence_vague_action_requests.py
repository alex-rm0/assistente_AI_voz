"""Presence: an action request whose only object is a vague reference
("Marca isso", "Send it") with nothing it can refer to fell through to the
default (NO_USEFUL_FOLLOWUP, question_needed=False). With the question
discipline anchored next to the user message, Echo then stopped asking what
to act on and sometimes claimed it had done it ("Vou marcar a mensagem...").
Presence only knows an antecedent through an active task (it gets no chat
history), so "no antecedent" means no active task.
"""
from __future__ import annotations

import pytest

from assistant.presence_decision import REASON_VAGUE_ACTION_REFERENCE, compute_presence_decision

VAGUE_REQUESTS = ("Marca isso", "Envia isso", "Guarda isto", "Abre aquilo", "Faz isso", "Mark that", "Send it", "Marca isso.")
COMMENTS = ("Isso é interessante", "Faz sentido isso", "Gosto disso", "That makes sense")


def _decide(message: str, *, active_task: bool = False, unresolved: bool = False):
    return compute_presence_decision(
        user_message=message,
        response_language="pt-PT",
        active_task_present=active_task,
        active_task_unresolved=unresolved,
        active_task_dense=False,
        is_restart=False,
    )


@pytest.mark.parametrize("message", VAGUE_REQUESTS)
def test_vague_action_request_without_antecedent_needs_a_question(message: str) -> None:
    decision = _decide(message)

    assert decision.question_needed is True
    assert decision.initiative == "ask"
    assert REASON_VAGUE_ACTION_REFERENCE in decision.reason_codes


@pytest.mark.parametrize("message", COMMENTS)
def test_comments_with_vague_references_are_unchanged(message: str) -> None:
    decision = _decide(message)

    assert decision.question_needed is False


def test_vague_request_with_an_active_task_keeps_current_behaviour() -> None:
    # Obvious antecedent: the previous turn produced the task's draft, so
    # "Envia isso" refers to it. Documents today's decision (unchanged).
    decision = _decide("Envia isso", active_task=True, unresolved=False)

    assert decision.question_needed is False
    assert decision.initiative == "answer_only"
