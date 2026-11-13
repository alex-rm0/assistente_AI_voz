"""F0 draft: Core contracts (docs/F0_CONTRACTS_DRAFT.md). Not wired to the runtime."""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from assistant.core.contracts import (
    ActionOutcome,
    ActionPolicy,
    ActionRecord,
    ActionRequest,
    ActionResult,
    ActionStatus,
    AttachmentRef,
    AttentionDecision,
    AttentionOutcome,
    AutonomyLevel,
    ComponentHealth,
    ConfirmationResult,
    ContractError,
    EchoEvent,
    EventStatus,
    HealthState,
    Importance,
    IncomingMessage,
    NEVER_AUTONOMOUS_ACTION_TYPES,
    NotificationPriority,
    OutgoingNotification,
    PrivacyClass,
    RiskClass,
    derived_privacy_class,
)

FIXTURES = Path(__file__).parent / "fixtures" / "core_contracts" / "v1"
T0 = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)


def event(**overrides):
    values = dict(id="evt-1", type="site.down", source="watcher.sites", occurred_at=T0,
                  observed_at=T0 + timedelta(seconds=30), subject="https://example.org",
                  importance=Importance.HIGH, confidence=0.9, dedupe_key="site:example.org:down",
                  status=EventStatus.ACTIVE, privacy_class=PrivacyClass.PUBLIC,
                  payload={"http_status": 503, "checks": [{"ok": False}]})
    values.update(overrides)
    return EchoEvent(**values)


def decision(**overrides):
    values = dict(id="dec-1", event_id="evt-1", decision=AttentionOutcome.INTERRUPT,
                  decided_at=T0, reason="site down for a watched project")
    values.update(overrides)
    return AttentionDecision(**values)


def message(**overrides):
    values = dict(id="msg-1", source_channel="echo_os", sender_id="alexandre", text="Olá",
                  received_at=T0, privacy_class=PrivacyClass.PERSONAL)
    values.update(overrides)
    return IncomingMessage(**values)


def notification(**overrides):
    values = dict(id="ntf-1", target_channel="telegram", recipient="alexandre", created_at=T0,
                  priority=NotificationPriority.HIGH, privacy_class=PrivacyClass.PUBLIC,
                  text="example.org is down", attention_decision_id="dec-1")
    values.update(overrides)
    return OutgoingNotification(**values)


def policy(**overrides):
    values = dict(action_type="email.archive_newsletter", autonomy_level=AutonomyLevel.SUGGEST,
                  max_autonomy_level=AutonomyLevel.AUTHORIZED, reversible=True,
                  external_effect=True, risk_class=RiskClass.LOW)
    values.update(overrides)
    return ActionPolicy(**values)


def authorized_policy(**overrides):
    values = dict(autonomy_level=AutonomyLevel.AUTHORIZED, authorized_by="alexandre", authorized_at=T0)
    values.update(overrides)
    return policy(**values)


def request(**overrides):
    values = dict(id="req-1", action_type="email.archive_newsletter", requested_by="echo.digest",
                  source_channel="echo_os", requested_at=T0, privacy_class=PrivacyClass.RESTRICTED,
                  policy=policy(), parameters={"message_id": "m-42"})
    values.update(overrides)
    return ActionRequest(**values)


def result(**overrides):
    values = dict(request_id="req-1", outcome=ActionOutcome.SUCCEEDED, completed_at=T0,
                  undo_reference="undo:m-42")
    values.update(overrides)
    return ActionResult(**values)


def record(**overrides):
    values = dict(id="rec-1", request_id="req-1", requested_by="echo.digest", source_channel="echo_os",
                  action_type="email.archive_newsletter", autonomy_level=AutonomyLevel.SUGGEST,
                  confirmation_required=True, confirmation_result=ConfirmationResult.CONFIRMED,
                  confirmation_channel="echo_os", reversible=True, status=ActionStatus.SUCCEEDED, recorded_at=T0,
                  parameters={"message_id": "m-42"}, executed_at=T0, result=result(),
                  undo_reference="undo:m-42")
    values.update(overrides)
    return ActionRecord(**values)


def health(**overrides):
    values = dict(component="provider.ollama", state=HealthState.HEALTHY, checked_at=T0)
    values.update(overrides)
    return ComponentHealth(**values)


BUILDERS = {
    "EchoEvent": event, "AttentionDecision": decision, "IncomingMessage": message,
    "OutgoingNotification": notification, "ActionPolicy": policy, "ActionRequest": request,
    "ActionResult": result, "ActionRecord": record, "ComponentHealth": health,
}


# --- shared machinery: every contract -------------------------------------------------------

@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_round_trips_through_json_and_carries_name_and_version(name):
    original = BUILDERS[name]()
    data = json.loads(original.to_json())
    assert data["contract"] == name
    assert data["contract_version"] == type(original).VERSION == original.contract_version
    assert type(original).from_json(original.to_json()) == original


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_is_immutable_including_nested_mappings(name):
    item = BUILDERS[name]()
    first = dataclasses.fields(item)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(item, first, "changed")


def test_nested_payload_is_frozen_and_hashing_ignores_mappings():
    item = event()
    with pytest.raises(TypeError):
        item.payload["http_status"] = 200
    with pytest.raises(TypeError):
        item.payload["checks"][0]["ok"] = True
    assert isinstance(item.payload["checks"], tuple)
    hash(item)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_missing_required_field_is_rejected(name):
    cls = type(BUILDERS[name]())
    data = BUILDERS[name]().to_dict()
    required = [f.name for f in dataclasses.fields(cls)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    del data[required[0]]
    with pytest.raises(ContractError, match=required[0]):
        cls.from_dict(data)


def test_wrong_contract_name_and_unsupported_major_version_are_rejected():
    data = event().to_dict()
    with pytest.raises(ContractError, match="contract"):
        EchoEvent.from_dict({**data, "contract": "AttentionDecision"})
    with pytest.raises(ContractError, match="contract_version"):
        EchoEvent.from_dict({**data, "contract_version": "2.0"})
    with pytest.raises(ContractError, match="contract_version"):
        EchoEvent.from_dict({k: v for k, v in data.items() if k != "contract_version"})
    with pytest.raises(ContractError, match="contract_version"):
        dataclasses.replace(event(), contract_version="1.7")


def test_unknown_fields_are_rejected_in_same_minor_but_ignored_from_newer_minor():
    data = event().to_dict()
    with pytest.raises(ContractError, match="unknown"):
        EchoEvent.from_dict({**data, "colour": "red"})
    newer = EchoEvent.from_dict({**data, "contract_version": "1.4", "colour": "red"})
    assert newer == event() and newer.contract_version == EchoEvent.VERSION


def test_invalid_enum_type_and_naive_timestamps_are_rejected():
    with pytest.raises(ContractError, match="privacy_class"):
        event(privacy_class="SECRET")
    with pytest.raises(ContractError, match="confidence"):
        event(confidence="high")
    with pytest.raises(ContractError, match="occurred_at"):
        event(occurred_at=datetime(2026, 11, 12, 9, 0))
    with pytest.raises(ContractError, match="subject"):
        event(subject="  ")
    with pytest.raises(ContractError, match="payload"):
        event(payload={"when": T0})
    with pytest.raises(ContractError, match="payload"):
        event(payload={"x": float("nan")})
    with pytest.raises(ContractError):
        EchoEvent.from_json("{not json")


def test_strings_from_json_are_coerced_to_enums_and_datetimes():
    item = EchoEvent.from_dict(event().to_dict())
    assert item.importance is Importance.HIGH and item.observed_at == T0 + timedelta(seconds=30)


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_reads_stored_json_of_current_version(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cls = type(BUILDERS[data["contract"]]())
    item = cls.from_dict(data)
    assert item.contract_version == "1.0"
    assert json.loads(item.to_json()) == data


def test_every_contract_has_a_stored_v1_fixture():
    assert {p.stem for p in FIXTURES.glob("*.json")} == set(BUILDERS)


# --- privacy --------------------------------------------------------------------------------

def test_derived_content_inherits_most_restrictive_class():
    assert derived_privacy_class(PrivacyClass.PUBLIC, PrivacyClass.RESTRICTED,
                                 PrivacyClass.PERSONAL) is PrivacyClass.RESTRICTED
    assert derived_privacy_class("PUBLIC", "PRIVATE") is PrivacyClass.PRIVATE
    with pytest.raises(ContractError):
        derived_privacy_class()


def test_event_message_and_request_must_carry_privacy_class():
    for builder in (event, message, request, notification):
        with pytest.raises(ContractError, match="privacy_class"):
            builder(privacy_class=None)


# --- EchoEvent / AttentionDecision ----------------------------------------------------------

def test_event_does_not_contain_the_attention_decision():
    names = {f.name for f in dataclasses.fields(EchoEvent)}
    assert not names & {"decision", "attention", "attention_decision", "deliver_after"}
    data = event().to_dict()
    for forbidden in ("decision", "attention", "attention_decision"):
        with pytest.raises(ContractError, match="attention"):
            EchoEvent.from_dict({**data, forbidden: "INTERRUPT"})
        with pytest.raises(ContractError, match="attention"):
            EchoEvent.from_dict({**data, "contract_version": "1.9", forbidden: "INTERRUPT"})


def test_event_type_and_confidence_are_validated():
    with pytest.raises(ContractError, match="type"):
        event(type="SiteDown")
    with pytest.raises(ContractError, match="confidence"):
        event(confidence=1.5)


def test_attention_decision_has_four_outcomes_and_references_event():
    assert [o.value for o in AttentionOutcome] == ["INTERRUPT", "DIGEST", "RECORD_ONLY", "DROP"]
    with pytest.raises(ContractError, match="event_id"):
        decision(event_id="")
    with pytest.raises(ContractError, match="reason"):
        decision(reason="")


def test_quiet_hours_defer_interrupt_without_changing_the_decision():
    deferred = decision(deliver_after=T0 + timedelta(hours=8))
    assert deferred.decision is AttentionOutcome.INTERRUPT
    for outcome in (AttentionOutcome.DIGEST, AttentionOutcome.RECORD_ONLY, AttentionOutcome.DROP):
        with pytest.raises(ContractError, match="deliver_after"):
            decision(decision=outcome, deliver_after=T0 + timedelta(hours=8))
    with pytest.raises(ContractError, match="deliver_after"):
        decision(deliver_after=T0 - timedelta(minutes=1))


# --- channels -------------------------------------------------------------------------------

def test_incoming_message_needs_text_or_attachment_and_valid_channel():
    with pytest.raises(ContractError, match="text"):
        message(text="  ")
    attached = message(text="", attachments=[{"contract": "AttachmentRef", "contract_version": "1.0",
                                              "id": "a1", "media_type": "image/png",
                                              "reference": "blob:a1"}])
    assert isinstance(attached.attachments[0], AttachmentRef)
    with pytest.raises(ContractError, match="source_channel"):
        message(source_channel="Telegram Bot")


def test_incoming_message_metadata_holds_capture_evidence():
    item = message(metadata={"modality": "voice", "capture_id": "c1", "avg_logprob": -0.4})
    assert IncomingMessage.from_json(item.to_json()).metadata["avg_logprob"] == -0.4


def test_outgoing_notification_needs_content():
    with pytest.raises(ContractError, match="content"):
        notification(text=None, content_ref=None)
    assert notification(text=None, content_ref="digest:2026-11-12").content_ref


# --- autonomy / actions ---------------------------------------------------------------------

def test_autonomy_levels_are_ordered_zero_to_three():
    assert [int(level) for level in AutonomyLevel] == [0, 1, 2, 3]


def test_autonomy_level_cannot_exceed_max():
    with pytest.raises(ContractError, match="max_autonomy_level"):
        policy(autonomy_level=AutonomyLevel.AUTHORIZED, max_autonomy_level=AutonomyLevel.SUGGEST,
               authorized_by="alexandre", authorized_at=T0)


@pytest.mark.parametrize("action_type", sorted(NEVER_AUTONOMOUS_ACTION_TYPES))
def test_never_autonomous_actions_cannot_have_max_level_three(action_type):
    with pytest.raises(ContractError, match="max_autonomy_level"):
        policy(action_type=action_type)
    assert policy(action_type=action_type, max_autonomy_level=AutonomyLevel.SUGGEST)


def test_level_three_requires_reversible_low_risk_and_prior_authorization():
    with pytest.raises(ContractError, match="reversible"):
        policy(reversible=False)
    with pytest.raises(ContractError, match="risk_class"):
        policy(risk_class=RiskClass.MEDIUM)
    with pytest.raises(ContractError, match="authoriz"):
        policy(autonomy_level=AutonomyLevel.AUTHORIZED)
    with pytest.raises(ContractError, match="authoriz"):
        policy(authorized_by="alexandre", authorized_at=T0)
    assert authorized_policy().confirmation_required is False
    assert policy().confirmation_required is True


def test_request_is_executable_only_from_level_two_and_matches_policy():
    for level in (AutonomyLevel.OFF, AutonomyLevel.NOTIFY):
        with pytest.raises(ContractError, match="autonomy_level"):
            request(policy=policy(autonomy_level=level))
    with pytest.raises(ContractError, match="action_type"):
        request(action_type="email.send")
    assert request().confirmation_required is True
    assert request(policy=authorized_policy()).confirmation_required is False


def test_action_result_failure_needs_error_and_success_has_none():
    with pytest.raises(ContractError, match="error"):
        result(outcome=ActionOutcome.FAILED, undo_reference=None)
    with pytest.raises(ContractError, match="error"):
        result(error="boom")
    assert result(outcome=ActionOutcome.FAILED, error="timeout", undo_reference=None)


def test_record_never_executes_without_confirmation_when_required():
    for answer in (ConfirmationResult.PENDING, ConfirmationResult.DECLINED,
                   ConfirmationResult.EXPIRED, ConfirmationResult.NOT_REQUIRED):
        channel = "echo_os" if answer is ConfirmationResult.DECLINED else None
        with pytest.raises(ContractError, match="confirmation_result"):
            record(confirmation_result=answer, confirmation_channel=channel)


def test_record_confirmation_matches_level():
    with pytest.raises(ContractError, match="confirmation_required"):
        record(confirmation_required=False)
    auto = record(autonomy_level=AutonomyLevel.AUTHORIZED, confirmation_required=False,
                  confirmation_result=ConfirmationResult.NOT_REQUIRED, confirmation_channel=None)
    assert auto.status is ActionStatus.SUCCEEDED
    with pytest.raises(ContractError, match="reversible"):
        record(autonomy_level=AutonomyLevel.AUTHORIZED, confirmation_required=False,
               confirmation_result=ConfirmationResult.NOT_REQUIRED, confirmation_channel=None, reversible=False,
               undo_reference=None, result=result(undo_reference=None))
    with pytest.raises(ContractError, match="autonomy_level"):
        record(autonomy_level=AutonomyLevel.NOTIFY)


def test_record_states_are_consistent():
    awaiting = record(status=ActionStatus.AWAITING_CONFIRMATION,
                      confirmation_result=ConfirmationResult.PENDING, confirmation_channel=None,
                      executed_at=None, result=None, undo_reference=None)
    assert awaiting.result is None
    with pytest.raises(ContractError, match="executed"):
        record(status=ActionStatus.AWAITING_CONFIRMATION, confirmation_result=ConfirmationResult.PENDING,
               confirmation_channel=None)
    declined = record(status=ActionStatus.NOT_EXECUTED, confirmation_result=ConfirmationResult.DECLINED,
                      executed_at=None, result=None, undo_reference=None)
    assert declined.executed_at is None
    with pytest.raises(ContractError, match="result"):
        record(result=None, undo_reference=None)
    with pytest.raises(ContractError, match="request_id"):
        record(result=result(request_id="other"))
    with pytest.raises(ContractError, match="outcome"):
        record(status=ActionStatus.FAILED)
    with pytest.raises(ContractError, match="undo_reference"):
        record(undo_reference="undo:other")
    with pytest.raises(ContractError, match="undo_reference"):
        record(reversible=False)


# --- health ---------------------------------------------------------------------------------

def test_health_states_and_reason_for_non_healthy():
    assert [s.value for s in HealthState] == ["healthy", "degraded", "unavailable"]
    with pytest.raises(ContractError, match="reason"):
        health(state=HealthState.DEGRADED)
    assert health(state=HealthState.UNAVAILABLE, reason="connection refused",
                  since=T0 - timedelta(minutes=5)).reason
    with pytest.raises(ContractError, match="since"):
        health(state=HealthState.DEGRADED, reason="slow", since=T0 + timedelta(minutes=1))


# --- Alexandre's answers of 12/11 (docs/F0_CONTRACTS_DRAFT.md) ------------------------------

def test_echo_os_messages_are_personal_by_default():
    from assistant.core.contracts.privacy import ECHO_OS_DEFAULT_PRIVACY_CLASS

    assert ECHO_OS_DEFAULT_PRIVACY_CLASS is PrivacyClass.PERSONAL


@pytest.mark.parametrize("privacy_class", [PrivacyClass.PRIVATE, PrivacyClass.RESTRICTED])
def test_private_content_never_goes_to_telegram_only_content_free_notices(privacy_class):
    with pytest.raises(ContractError, match="content_free"):
        notification(privacy_class=privacy_class, text="Assunto: proposta de 40 000 EUR ...")
    with pytest.raises(ContractError, match="content_ref"):
        notification(privacy_class=privacy_class, content_free=True, text=None, content_ref="email:m-42")
    notice = notification(privacy_class=privacy_class, content_free=True,
                          text="Chegou um email importante de Francisca")
    assert OutgoingNotification.from_json(notice.to_json()).content_free is True
    # The local window may show the content itself.
    assert notification(target_channel="echo_os", privacy_class=privacy_class, text="Assunto: ...")


def test_confirmation_from_another_channel_only_if_allowlisted():
    from assistant.core.contracts.channels import CONFIRMATION_CHANNELS

    assert {"echo_os", "telegram"} <= CONFIRMATION_CHANNELS
    assert record(source_channel="echo_os", confirmation_channel="telegram").confirmation_channel == "telegram"
    with pytest.raises(ContractError, match="confirmation_channel"):
        record(confirmation_channel="whatsapp")  # unknown channel: never
    with pytest.raises(ContractError, match="confirmation_channel"):
        record(confirmation_channel=None)  # an answer always says where it came from
    with pytest.raises(ContractError, match="confirmation_channel"):
        record(status=ActionStatus.AWAITING_CONFIRMATION, confirmation_result=ConfirmationResult.PENDING,
               executed_at=None, result=None, undo_reference=None)  # no answer yet, no channel


@pytest.mark.parametrize("action_type", ["data.share", "data.publish", "permissions.change",
                                         "credentials.change", "calendar.accept_invitation",
                                         "commitment.accept"])
def test_sharing_permissions_and_commitments_are_never_autonomous(action_type):
    assert action_type in NEVER_AUTONOMOUS_ACTION_TYPES
