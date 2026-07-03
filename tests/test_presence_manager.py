from __future__ import annotations

from assistant.presence_manager import PresenceManager, PresenceState


def test_presence_manager_defaults_to_active_conversation() -> None:
    presence = PresenceManager()

    assert presence.state == PresenceState.ACTIVE_CONVERSATION
    assert presence.can_respond()
    assert presence.can_use_tools()
    assert presence.can_store_memory()


def test_private_mode_responds_without_memory_or_observation() -> None:
    presence = PresenceManager(PresenceState.PRIVATE_MODE)

    assert presence.can_respond()
    assert not presence.can_use_tools()
    assert not presence.can_store_memory()
    assert not presence.can_observe_activity()


def test_passive_monitoring_does_not_respond_but_can_observe() -> None:
    presence = PresenceManager(PresenceState.PASSIVE_MONITORING)

    assert not presence.can_respond()
    assert not presence.can_use_tools()
    assert presence.can_store_memory()
    assert presence.can_observe_activity()


def test_offline_disables_everything() -> None:
    presence = PresenceManager(PresenceState.OFFLINE)

    assert not presence.can_respond()
    assert not presence.can_use_tools()
    assert not presence.can_store_memory()
    assert not presence.can_observe_activity()


def test_invalid_state_falls_back_to_active_conversation() -> None:
    presence = PresenceManager("INVALID")

    assert presence.state == PresenceState.ACTIVE_CONVERSATION


def test_presence_manager_detects_mode_requests_from_messages() -> None:
    presence = PresenceManager()

    assert presence.requested_state_from_message("vou trabalhar calado") == PresenceState.PASSIVE_MONITORING
    assert presence.requested_state_from_message("não me interrompas") == PresenceState.FOCUS_MODE
    assert presence.requested_state_from_message("isto é privado") == PresenceState.PRIVATE_MODE
    assert presence.requested_state_from_message("modo conversa") == PresenceState.ACTIVE_CONVERSATION
    assert presence.requested_state_from_message("fica offline") == PresenceState.OFFLINE
    assert presence.requested_state_from_message("olá") is None


def test_presence_manager_confirms_mode_changes() -> None:
    presence = PresenceManager()

    assert presence.confirmation_for(PresenceState.PASSIVE_MONITORING) == "Entendido, vou ficar em modo observador."
    assert presence.confirmation_for(PresenceState.ACTIVE_CONVERSATION) == "Entendido, volto ao modo conversa."


def test_presence_manager_wake_commands() -> None:
    presence = PresenceManager(PresenceState.OFFLINE)

    for message in (
        "volta ao modo conversa",
        "modo conversa",
        "podes voltar a falar comigo",
        "volta",
        "acorda",
        "reativa-te",
        "ativa-te",
    ):
        assert presence.can_wake_from_state(message)
        assert presence.requested_state_from_message(message) == PresenceState.ACTIVE_CONVERSATION


def test_presence_manager_state_report_uses_real_state() -> None:
    presence = PresenceManager(PresenceState.FOCUS_MODE)

    assert presence.state_report().startswith("Estou em FOCUS_MODE.")
