from __future__ import annotations

import pytest

from tests.test_automatic_routing_regressions import make_engine


def english_engine(tmp_path, replies=None):
    engine, _, local, _ = make_engine(tmp_path, mode="local", ollama_replies=replies)
    engine._set_response_language("en-GB")
    return engine, local


@pytest.mark.parametrize("message,reply", [
    ("Agora preciso que guardes isso num fecheiro, num workspace, por favor.", "I've saved the note in a workspace."),
    ("Qual é o nome do fecheiro?", 'It\'s saved in a workspace named "Reunião da Semana".'),
    ("Porquê que aparece ainda a task no ecrã?", "The task was saved, but it's still visible because it's been added to the current day's schedule."),
    ("Então acrescenta mais um e-mail para ver como fica no ecrã.", "I've added the new email to the workspace."),
])
def test_unexecuted_write_claim_is_not_shown(tmp_path, message, reply):
    engine, local = english_engine(tmp_path)
    response = engine._complete_turn(message, reply, "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")
    assert response != reply
    assert engine._last_response_blocked is True
    assert local.calls == []


@pytest.mark.parametrize("reply", [
    "I can help you save the email.",
    "You said you saved the file yesterday.",
    'The example says: "I have saved the note in a workspace."',
])
def test_nonexecuting_or_attributed_text_is_not_a_false_action(tmp_path, reply):
    engine, _ = english_engine(tmp_path)
    assert engine._complete_turn("Explica isso.", reply, "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION") == reply


def test_email_body_is_not_mistaken_for_assistant_execution(tmp_path):
    engine, _ = english_engine(tmp_path)
    email = "Subject: Update\n\nHi Ana,\n\nI've saved the note in a workspace.\n\nAlexandre"
    assert engine._complete_turn("Escreve o email.", email, "WRITING_WORKFLOW", technical=True, selected_path="WRITING_TASK") == email
