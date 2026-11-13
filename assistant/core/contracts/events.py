"""EchoEvent (the observed fact) and AttentionDecision (what Echo decided about it).

North Star §2.1.4 and §3.1: the event never carries the decision; the decision
references the event by id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from assistant.core.contracts._base import IDENTIFIER_RE, Contract, require
from assistant.core.contracts.privacy import PrivacyClass


class Importance(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(str, Enum):
    ACTIVE = "active"            # the observed condition still holds
    RESOLVED = "resolved"        # a later transition closed it (site.down -> site.recovered)
    SUPERSEDED = "superseded"    # replaced by a newer event with the same dedupe_key


@dataclass(frozen=True)
class EchoEvent(Contract):
    CONTRACT = "EchoEvent"
    _FORBIDDEN_FIELDS = frozenset({"decision", "attention", "attention_decision"})
    _FORBIDDEN_REASON = "the attention decision is a separate AttentionDecision, never part of the event"

    id: str
    type: str
    source: str
    occurred_at: datetime
    observed_at: datetime
    subject: str
    importance: Importance
    confidence: float
    dedupe_key: str
    status: EventStatus
    privacy_class: PrivacyClass
    payload: Mapping[str, Any] = field(default_factory=dict, hash=False)
    related_task_id: str | None = None
    related_project_id: str | None = None
    contract_version: str | None = None

    def _check(self):
        require(IDENTIFIER_RE.match(self.type) and "." in self.type,
                f"type: expected a dotted lowercase name like 'site.down', got {self.type!r}")
        require(IDENTIFIER_RE.match(self.source), f"source: expected a lowercase identifier, got {self.source!r}")
        require(0.0 <= self.confidence <= 1.0, "confidence: must be between 0 and 1")


class AttentionOutcome(str, Enum):
    INTERRUPT = "INTERRUPT"
    DIGEST = "DIGEST"
    RECORD_ONLY = "RECORD_ONLY"
    DROP = "DROP"


@dataclass(frozen=True)
class AttentionDecision(Contract):
    CONTRACT = "AttentionDecision"

    id: str
    event_id: str
    decision: AttentionOutcome
    decided_at: datetime
    reason: str
    deliver_after: datetime | None = None
    contract_version: str | None = None

    def _check(self):
        # Quiet hours move the delivery moment, never the decision (§4).
        if self.deliver_after is not None:
            require(self.decision is AttentionOutcome.INTERRUPT,
                    "deliver_after: only an INTERRUPT can be deferred; other outcomes have no immediate delivery")
            require(self.deliver_after >= self.decided_at, "deliver_after: must not be before decided_at")
