"""Autonomy model and action contracts (North Star §8).

Level 3 is prior authorization, not a confirmation bypass: it needs an explicit
decision recorded on the policy, a max level that allows it, and an action
that is reversible and low risk. Some action types can never reach level 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Mapping

from assistant.core.contracts._base import IDENTIFIER_RE, Contract, require
from assistant.core.contracts.channels import CONFIRMATION_CHANNELS
from assistant.core.contracts.privacy import PrivacyClass


class AutonomyLevel(IntEnum):
    OFF = 0          # does not do or suggest
    NOTIFY = 1       # only informs
    SUGGEST = 2      # prepares the action and asks yes/no
    AUTHORIZED = 3   # executes under prior authorization and reports; reversible only


class RiskClass(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# North Star §8.1, "máximo nível 2". Destructive or externally binding actions
# are also capped by the reversible/risk rules; this list catches the named
# categories even if someone marks them reversible or low risk.
NEVER_AUTONOMOUS_ACTION_TYPES = frozenset({
    "email.send",
    "message.send",
    "file.delete",
    "data.delete",
    "calendar.cancel_meeting",
    "payment.make",
    "purchase.make",
    # Alexandre, 12/11:
    "data.share",
    "data.publish",
    "permissions.change",
    "credentials.change",
    "calendar.accept_invitation",
    "commitment.accept",
})


@dataclass(frozen=True)
class ActionPolicy(Contract):
    CONTRACT = "ActionPolicy"

    action_type: str
    autonomy_level: AutonomyLevel
    max_autonomy_level: AutonomyLevel
    reversible: bool
    external_effect: bool
    risk_class: RiskClass
    authorized_by: str | None = None
    authorized_at: datetime | None = None
    contract_version: str | None = None

    def _check(self):
        require(IDENTIFIER_RE.match(self.action_type) and "." in self.action_type,
                f"action_type: expected a dotted lowercase name, got {self.action_type!r}")
        require(self.autonomy_level <= self.max_autonomy_level,
                "max_autonomy_level: autonomy_level cannot exceed it")
        if self.max_autonomy_level is AutonomyLevel.AUTHORIZED:
            require(self.action_type not in NEVER_AUTONOMOUS_ACTION_TYPES,
                    f"max_autonomy_level: {self.action_type} is capped at level 2")
            require(self.reversible, "reversible: level 3 is only allowed for reversible actions")
            require(self.risk_class is RiskClass.LOW, "risk_class: level 3 is only allowed for low-risk actions")
        authorized = self.authorized_by is not None or self.authorized_at is not None
        if self.autonomy_level is AutonomyLevel.AUTHORIZED:
            require(self.authorized_by is not None and self.authorized_at is not None,
                    "authorized_by/authorized_at: level 3 needs an explicit prior authorization")
        else:
            require(not authorized, "authorized_by/authorized_at: only a level 3 policy carries an authorization")

    @property
    def confirmation_required(self):
        return self.autonomy_level is not AutonomyLevel.AUTHORIZED


@dataclass(frozen=True)
class ActionRequest(Contract):
    CONTRACT = "ActionRequest"

    id: str
    action_type: str
    requested_by: str
    source_channel: str
    requested_at: datetime
    privacy_class: PrivacyClass
    policy: ActionPolicy
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)
    contract_version: str | None = None

    def _check(self):
        require(self.policy.action_type == self.action_type, "action_type: must match the policy")
        require(self.policy.autonomy_level >= AutonomyLevel.SUGGEST,
                "autonomy_level: levels 0 and 1 never produce an executable request")

    @property
    def confirmation_required(self):
        return self.policy.confirmation_required


class ActionOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ActionResult(Contract):
    CONTRACT = "ActionResult"

    request_id: str
    outcome: ActionOutcome
    completed_at: datetime
    output: Mapping[str, Any] = field(default_factory=dict, hash=False)
    error: str | None = None
    undo_reference: str | None = None
    contract_version: str | None = None

    def _check(self):
        if self.outcome is ActionOutcome.FAILED:
            require(self.error is not None, "error: a failed action must say why")
            require(self.undo_reference is None, "undo_reference: nothing to undo after a failure")
        else:
            require(self.error is None, "error: a succeeded action carries no error")


class ConfirmationResult(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DECLINED = "declined"
    EXPIRED = "expired"
    NOT_REQUIRED = "not_required"


class ActionStatus(str, Enum):
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    NOT_EXECUTED = "not_executed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_EXECUTED = {ActionStatus.SUCCEEDED: ActionOutcome.SUCCEEDED, ActionStatus.FAILED: ActionOutcome.FAILED}


@dataclass(frozen=True)
class ActionRecord(Contract):
    """Action Ledger entry (§8.2). "What did you do today?" is answered from these."""
    CONTRACT = "ActionRecord"

    id: str
    request_id: str
    requested_by: str
    source_channel: str
    action_type: str
    autonomy_level: AutonomyLevel
    confirmation_required: bool
    confirmation_result: ConfirmationResult
    reversible: bool
    status: ActionStatus
    recorded_at: datetime
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)
    executed_at: datetime | None = None
    result: ActionResult | None = None
    undo_reference: str | None = None
    # Where the confirmation answer came from: may differ from source_channel,
    # but only an allowlisted channel answers (Alexandre, 12/11).
    confirmation_channel: str | None = None
    contract_version: str | None = None

    def _check(self):
        if self.confirmation_result in (ConfirmationResult.CONFIRMED, ConfirmationResult.DECLINED):
            require(self.confirmation_channel in CONFIRMATION_CHANNELS,
                    f"confirmation_channel: an answer must come from one of {sorted(CONFIRMATION_CHANNELS)}, "
                    f"got {self.confirmation_channel!r}")
        else:
            require(self.confirmation_channel is None, "confirmation_channel: only an answered confirmation has one")
        require(self.autonomy_level >= AutonomyLevel.SUGGEST,
                "autonomy_level: only level 2 and 3 actions reach the ledger")
        require(self.confirmation_required == (self.autonomy_level is AutonomyLevel.SUGGEST),
                "confirmation_required: true exactly for level 2")
        if self.confirmation_required:
            require(self.confirmation_result is not ConfirmationResult.NOT_REQUIRED,
                    "confirmation_result: a level 2 action needs a confirmation answer")
        else:
            require(self.confirmation_result is ConfirmationResult.NOT_REQUIRED,
                    "confirmation_result: must be not_required when no confirmation is required")
            require(self.reversible, "reversible: level 3 actions must be reversible")
        executed = self.status in _EXECUTED
        if executed:
            require(not self.confirmation_required or self.confirmation_result is ConfirmationResult.CONFIRMED,
                    "confirmation_result: never executed without confirmation")
            require(self.executed_at is not None and self.result is not None,
                    "result: an executed action needs executed_at and result")
            require(self.result.request_id == self.request_id, "request_id: result belongs to another request")
            require(self.result.outcome is _EXECUTED[self.status], "outcome: status and result outcome disagree")
        else:
            require(self.executed_at is None and self.result is None,
                    "executed_at/result: only executed actions have them")
            pending = self.status is ActionStatus.AWAITING_CONFIRMATION
            require((self.confirmation_result is ConfirmationResult.PENDING) == pending,
                    "confirmation_result: pending exactly while awaiting confirmation")
        if self.undo_reference is not None:
            require(self.reversible and self.status is ActionStatus.SUCCEEDED,
                    "undo_reference: only a succeeded reversible action can be undone")
        if self.result is not None:
            require(self.result.undo_reference == self.undo_reference,
                    "undo_reference: must match the result")
