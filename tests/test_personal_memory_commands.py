from __future__ import annotations

import pytest

from assistant.personal_model import PersonalModel
from tests.test_automatic_routing_regressions import make_engine


@pytest.fixture
def memory_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHO_CORE_PROFILE_PATH", str(tmp_path / "no-profile.md"))
    def build(language="en-GB", replies=None):
        path = tmp_path / str(len(list(tmp_path.glob("engine*"))))
        path = tmp_path / ("engine" + path.name)
        engine, _, local, _ = make_engine(path, mode="local", ollama_replies=replies)
        engine.personal_model = PersonalModel(path / "data")
        engine.response_composer.personal_model = engine.personal_model
        engine._set_response_language(language)
        return engine, local
    return build


@pytest.mark.parametrize("language", ["pt-PT", "en-GB"])
@pytest.mark.parametrize("message,fact", [
    ("O meu treinador chama-se Miguel Alfayate. Podes-te lembrar disso.", "O meu treinador chama-se Miguel Alfayate"),
    ("lembra-te que o meu treinador é Miguel Alfayate", "o meu treinador é Miguel Alfayate"),
    ("Remember that my coach is Miguel Alfayate", "my coach is Miguel Alfayate"),
    ("My coach is Miguel Alfayate. Can you remember that?", "My coach is Miguel Alfayate"),
])
def test_explicit_memory_writes_real_fact_before_reply(memory_engine, language, message, fact):
    engine, local = memory_engine(language)
    response = engine.respond(message)
    entries = engine.personal_model.entries_about()
    assert len(entries) == 1
    assert entries[0].description.strip(".") == fact
    assert entries[0].source and entries[0].created_at
    assert fact in response
    assert '"' in response
    assert local.calls == []


@pytest.mark.parametrize("command", ["guarda isto", "podes-te lembrar disso", "remember that", "save this"])
def test_anaphoric_save_uses_previous_user_not_assistant(memory_engine, command):
    engine, local = memory_engine()
    engine.memory.append_pair("O meu treinador é Miguel Alfayate.", "Your coach is Someone Invented.")
    response = engine.respond(command)
    assert "Miguel Alfayate" in response
    assert "Someone Invented" not in engine.personal_model.entries_about()[0].description
    assert local.calls == []


def test_unresolved_reference_never_saves_the_assistant_text(memory_engine):
    engine, local = memory_engine()
    engine.memory.append_pair("Quem é o meu treinador?", "Your coach is Someone Invented.")
    response = engine.respond("remember that")
    assert engine.personal_model.entries_about() == []
    assert "?" in response
    assert local.calls == []


def test_failed_write_has_no_success_confirmation(memory_engine, monkeypatch):
    engine, local = memory_engine()
    def fail(**kwargs):
        raise OSError("database unavailable")
    monkeypatch.setattr(engine.personal_model, "add_or_update_entry", fail)
    response = engine.respond("Remember that my coach is Miguel Alfayate")
    assert "couldn't save" in response
    assert engine.personal_model.entries_about() == []
    assert local.calls == []


@pytest.mark.parametrize("language,fact,yes,no", [
    ("en-GB", "My coach is Miguel Alfayate", "yes", "no"),
    ("pt-PT", "O meu treinador é Miguel Alfayate", "sim", "não"),
    ("pt-PT", "Tenho regata no sábado", "sim", "não"),
])
def test_personal_statement_requires_consent(memory_engine, language, fact, yes, no):
    engine, local = memory_engine(language)
    proposal = engine.respond(fact)
    assert "?" in proposal
    assert engine.personal_model.entries_about() == []
    receipt = engine.respond(yes)
    assert fact in receipt
    assert len(engine.personal_model.entries_about()) == 1
    assert local.calls == []


def test_declined_or_stale_proposal_is_not_saved(memory_engine):
    engine, _ = memory_engine()
    engine.respond("My coach is Miguel Alfayate")
    engine.respond("no")
    assert engine.personal_model.entries_about() == []
    assert "save" not in engine.respond("My coach is Miguel Alfayate").lower()
    engine.respond("My sister is Maria")
    engine.respond("Hello")
    engine.respond("yes")
    assert engine.personal_model.entries_about() == []


@pytest.mark.parametrize("message", ["Estou cansado", "It's raining", "My coach is Miguel?", "I need help with an email"])
def test_ordinary_messages_do_not_propose_memory(memory_engine, message):
    engine, _ = memory_engine()
    response = engine.respond(message)
    assert "shall I save" not in response.lower()
    assert "queres que guarde" not in response.lower()
    assert engine.personal_model.entries_about() == []


def test_saved_coach_reaches_composer_across_languages(memory_engine):
    from assistant.response_composer import ComposerRequest
    engine, local = memory_engine(replies=["Your coach is Miguel Alfayate."])
    engine.respond("O meu treinador chama-se Miguel Alfayate. Podes-te lembrar disso.")
    reply = engine.response_composer.compose(ComposerRequest(intent="conversation", user_message="Who is my coach?", response_language="en-GB"))
    assert "Miguel Alfayate" in reply
    assert len(local.calls) == 1
    assert "Miguel Alfayate" in str(local.calls)
    before = len(local.calls)
    engine.response_composer.compose(ComposerRequest(intent="conversation", user_message="What is the capital of France?", response_language="en-GB"))
    assert "Miguel Alfayate" not in str(local.calls[before:])


@pytest.mark.parametrize("language,inventory,forget", [
    ("en-GB", "What do you know about me?", "Forget my coach"),
    ("pt-PT", "O que sabes sobre mim?", "Esquece o meu treinador"),
])
def test_inventory_and_forgetting_are_real_and_deterministic(memory_engine, language, inventory, forget, monkeypatch, tmp_path):
    from assistant.core_profile import load_core_profile
    profile = tmp_path / "profile.md"
    profile.write_text("Alexandre studies software engineering.", encoding="utf-8")
    monkeypatch.setenv("ECHO_CORE_PROFILE_PATH", str(profile))
    engine, local = memory_engine(language)
    monkeypatch.setenv("ECHO_CORE_PROFILE_PATH", str(profile))
    engine.respond("Remember that my coach is Miguel Alfayate")
    response = engine.respond(inventory)
    assert "Miguel Alfayate" in response
    assert "software engineering" in response
    assert '> Alexandre studies software engineering.' in response
    assert local.calls == []
    receipt = engine.respond(forget)
    assert "Miguel Alfayate" in receipt
    assert engine.personal_model.entries_about() == []
    assert "Miguel Alfayate" not in engine.respond(inventory)
    assert profile.read_text(encoding="utf-8") == "Alexandre studies software engineering."
    assert local.calls == []


def test_forget_that_targets_saved_entry_not_assistant(memory_engine):
    engine, local = memory_engine()
    engine.respond("Remember that my coach is Miguel Alfayate")
    engine.respond("Forget that")
    assert engine.personal_model.entries_about() == []
    assert local.calls == []


def test_deleted_personal_fact_is_not_resent_as_history(memory_engine):
    from assistant.response_composer import ComposerRequest
    engine, local = memory_engine()
    engine.respond("Remember that my coach is Miguel Alfayate")
    engine.respond("Forget my coach")
    reply = engine.response_composer.compose(ComposerRequest(
        intent="conversation", user_message="Who is my coach?", response_language="en-GB",
        history=engine.memory.load(),
    ))
    assert "don't know" in reply
    assert local.calls == []


@pytest.mark.parametrize("reply", ["Noted.", "I'll remember that.", "I've saved that in my memory.", "Guardei essa informação.", "Vou lembrar-me disso."])
def test_unsaved_memory_claim_is_blocked_in_all_llm_paths(memory_engine, reply):
    engine, _ = memory_engine()
    engine._begin_turn_trace("My coach is Miguel")
    response = engine._finalize_response("My coach is Miguel", reply, "AGENT_DIRECT")
    assert response != reply
    assert "haven't saved" in response
    assert engine.personal_model.entries_about() == []


def test_composer_cannot_claim_a_write_or_block_conditional_offer(memory_engine):
    from assistant.response_composer import ComposerRequest
    engine, _ = memory_engine(replies=["Noted.", "Tell me and I'll remember it, if you like."])
    request = ComposerRequest(intent="conversation", user_message="We had a good day.", response_language="en-GB")
    assert "haven't saved" in engine.response_composer.compose(request)
    assert "if you like" in engine.response_composer.compose(request)


def test_readonly_tool_is_not_a_memory_write_receipt(memory_engine):
    engine, _ = memory_engine()
    engine._begin_turn_trace("What window is open?")
    reply = engine._finalize_response("What window is open?", "I'll remember that.", "AGENT_DIRECT", tools_used=("get_active_window",))
    assert "haven't saved" in reply


@pytest.mark.parametrize("previous", ["O que sabes sobre mim", "When is my exam", "yes", "sim"])
def test_reference_does_not_turn_question_or_confirmation_into_fact(memory_engine, previous):
    engine, local = memory_engine()
    engine.memory.append_pair(previous, "My coach is Invented.")
    assert "?" in engine.respond("remember that")
    assert engine.personal_model.list_entries() == []
    assert local.calls == []


def test_clear_and_inventory_expire_unanswered_proposals(memory_engine):
    engine, _ = memory_engine()
    engine.respond("My coach is Miguel")
    engine.clear_conversation()
    engine.respond("yes")
    assert engine.personal_model.list_entries() == []
    engine.respond("My sister is Maria")
    engine.respond("What do you know about me?")
    engine.respond("yes")
    assert engine.personal_model.list_entries() == []


def test_forgetting_unsaved_topic_expires_consent(memory_engine):
    engine, _ = memory_engine()
    engine.respond("My coach is Miguel")
    engine.respond("Forget my coach")
    engine.respond("yes")
    assert engine.personal_model.list_entries() == []


@pytest.mark.parametrize("message", ["I have a regatta on Saturday", "O meu treinador é Miguel. Ajuda-me a escrever um email."])
def test_event_and_compound_request_have_appropriate_consent(memory_engine, message):
    from assistant.personal_memory_commands import durable_statement
    assert durable_statement(message) == message.startswith("I have")


def test_noted_prefix_is_a_claim_but_quoted_example_is_not(memory_engine):
    from assistant.personal_memory_commands import guard_memory_claim
    assert "haven't saved" in guard_memory_claim("Noted. Your coach is Miguel.", "en-GB")
    quoted = 'You could write: "I will remember that."'
    assert guard_memory_claim(quoted, "en-GB") == quoted


def test_ambiguous_forgetting_does_not_guess(memory_engine):
    engine, local = memory_engine()
    engine.respond("Remember that my coach is Miguel")
    engine.respond("Remember that my swimming coach is Maria")
    assert "Which fact" in engine.respond("Forget my coach")
    assert len(engine.personal_model.list_entries()) == 2
    assert local.calls == []


def test_memory_survives_reopening_and_source_is_explicit(memory_engine):
    engine, _ = memory_engine()
    engine.respond("My coach is Miguel")
    engine.respond("yes")
    reopened = PersonalModel(engine.personal_model.data_path)
    entry, = reopened.list_entries()
    assert entry.source == "user_confirmed"
    assert entry.created_at and "Consent: yes" in entry.evidence


def test_engine_recalls_saved_fact_after_conversation_is_cleared(memory_engine):
    engine, local = memory_engine(replies=["Your coach is Miguel Alfayate."])
    engine.respond("Remember that my coach is Miguel Alfayate")
    engine.clear_conversation()
    response = engine.respond("Who is my coach?")
    assert "Miguel Alfayate" in response
    assert "Miguel Alfayate" in str(local.calls)


def test_active_workflow_is_not_interrupted_by_personal_proposal(memory_engine):
    engine, _ = memory_engine()
    engine.respond("vamos escrever um mail")
    assert engine._work_session.active_task is not None
    reply = engine.respond("My coach is Miguel")
    assert "Shall I save" not in reply
    assert engine.personal_model.list_entries() == []


def test_private_mode_does_not_retrieve_saved_personal_facts(memory_engine):
    from assistant.presence_manager import PresenceState
    engine, local = memory_engine()
    engine.respond("Remember that my coach is Miguel Alfayate")
    engine.clear_conversation()
    engine.presence.set_state(PresenceState.PRIVATE_MODE)
    response = engine.respond("Who is my coach?")
    assert "Miguel Alfayate" not in response
    assert "Miguel Alfayate" not in str(local.calls)


def test_offline_does_not_bypass_presence_with_inventory(memory_engine):
    from assistant.presence_manager import PresenceState
    engine, _ = memory_engine()
    engine.respond("Remember that my coach is Miguel Alfayate")
    engine.presence.set_state(PresenceState.OFFLINE)
    assert "Miguel Alfayate" not in engine.respond("What do you know about me?")
