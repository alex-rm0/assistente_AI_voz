from __future__ import annotations

import pytest

from tests.test_system_health import _engine, _health, ALL_OK
from tests.test_live_writing_regressions import english_engine, collected_email


@pytest.mark.parametrize("language,question", [("en-GB", "And you"), ("pt-PT", "E tu")])
def test_wellbeing_invites_reciprocity_without_a_service_question(tmp_path, language, question):
    engine, llm = _engine(tmp_path, language)
    engine.system_health = _health(ALL_OK)
    reply = engine.respond("Bom dia, tudo bem?")
    assert question in reply
    assert reply.count("?") == 1
    assert "help" not in reply.lower() and "ajudar" not in reply.lower()
    assert llm.calls == 0


def test_repeated_wellbeing_does_not_repeat_a_question_or_greeting(tmp_path):
    engine, _ = _engine(tmp_path, "en-GB")
    engine.respond("Bom dia, tudo bem?")
    second = engine.respond("Bom dia, tudo bem?")
    assert "?" not in second
    assert "Good morning" not in second


def test_wellbeing_can_refer_to_real_email_without_filling_it(tmp_path):
    engine, local = english_engine(tmp_path)
    artifact = collected_email(engine)
    before = engine._pending_writing_task.recipient
    # Close the previous question, so a social return is not another interview.
    for _ in range(4):
        engine.memory.append_pair("Obrigado.", "You're welcome.")
    reply = engine.respond("Tudo bem?")
    assert "email" in reply.lower() and "?" in reply
    assert "Francisca" not in reply  # no need to recite stored personal details
    assert engine._pending_writing_task.active_artifact_id == artifact
    assert engine._pending_writing_task.recipient == before
    assert local.calls == []


def test_social_reply_never_creates_a_task_or_invents_an_agenda(tmp_path):
    engine, llm = _engine(tmp_path, "en-GB")
    reply = engine.respond("Como estás?")
    assert not any(word in reply.lower() for word in ("meeting", "free day", "scheduled", "project"))
    assert engine._work_session.active_task is None
    assert engine._pending_writing_task is None
    assert llm.calls == 0


def test_social_return_uses_the_real_generic_comparison_kind(tmp_path):
    engine, local = english_engine(tmp_path)
    engine.respond("Compara Samsung com Apple")
    task = engine._work_session.active_task
    for _ in range(4):
        engine.memory.append_pair("Obrigado.", "You're welcome.")
    reply = engine.respond("Como estás?")
    assert "comparison" in reply
    assert engine._work_session.active_task is task
    assert local.calls == []
