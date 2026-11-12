import pytest

from tests.test_live_writing_regressions import english_engine, collected_email


@pytest.mark.parametrize("language,reply", [
    ("pt-PT", "Sim, enviei."), ("pt-PT", "Sim, já o enviei."),
    ("pt-PT", "O email já foi enviado ao professor."),
    ("en-GB", "Yes, I sent it."), ("en-GB", "I've sent the email."),
    ("en-GB", "The email has been sent."),
])
def test_email_sending_cannot_be_claimed_without_execution(tmp_path, language, reply):
    engine, local = english_engine(tmp_path)
    engine._set_response_language(language)
    response = engine._complete_turn("Já enviaste o email ao professor?", reply,
                                    "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")
    assert response != reply
    assert engine._last_response_blocked
    assert "draft" in response if language == "en-GB" else "rascunho" in response
    assert not local.calls


@pytest.mark.parametrize("reply", [
    "Tu enviaste-me o ficheiro ontem.", "You sent me the file yesterday.",
    "I can help you write the email.", "I haven't sent the email.",
    "Não enviei o email.", 'You wrote: "I sent the email".',
])
def test_non_execution_send_mentions_are_preserved(tmp_path, reply):
    engine, _ = english_engine(tmp_path)
    assert engine._complete_turn("Explica isso.", reply, "RESPONSE_COMPOSER",
                                 selected_path="GENERAL_CONVERSATION") == reply
