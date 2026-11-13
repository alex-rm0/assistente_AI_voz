"""Core contracts, F0 draft (docs/F0_CONTRACTS_DRAFT.md).

Pure data: no imports from the runtime and no runtime imports this package yet.
"""
from assistant.core.contracts._base import Contract, ContractError
from assistant.core.contracts.actions import (
    NEVER_AUTONOMOUS_ACTION_TYPES,
    ActionOutcome,
    ActionPolicy,
    ActionRecord,
    ActionRequest,
    ActionResult,
    ActionStatus,
    AutonomyLevel,
    ConfirmationResult,
    RiskClass,
)
from assistant.core.contracts.channels import (
    AttachmentRef,
    IncomingMessage,
    NotificationPriority,
    OutgoingNotification,
)
from assistant.core.contracts.events import (
    AttentionDecision,
    AttentionOutcome,
    EchoEvent,
    EventStatus,
    Importance,
)
from assistant.core.contracts.health import ComponentHealth, HealthState
from assistant.core.contracts.privacy import PrivacyClass, derived_privacy_class

__all__ = [
    "NEVER_AUTONOMOUS_ACTION_TYPES", "ActionOutcome", "ActionPolicy", "ActionRecord", "ActionRequest",
    "ActionResult", "ActionStatus", "AttachmentRef", "AttentionDecision", "AttentionOutcome",
    "AutonomyLevel", "ComponentHealth", "ConfirmationResult", "Contract", "ContractError", "EchoEvent",
    "EventStatus", "HealthState", "Importance", "IncomingMessage", "NotificationPriority",
    "OutgoingNotification", "PrivacyClass", "RiskClass", "derived_privacy_class",
]
