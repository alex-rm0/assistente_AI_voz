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
