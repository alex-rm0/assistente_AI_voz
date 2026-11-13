"""Channel boundary (North Star §3.3): what every ChannelAdapter translates to and from."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from assistant.core.contracts._base import IDENTIFIER_RE, Contract, require
from assistant.core.contracts.privacy import PrivacyClass


def _check_channel(name, value):
    require(IDENTIFIER_RE.match(value), f"{name}: expected a lowercase identifier like 'telegram', got {value!r}")


@dataclass(frozen=True)
class AttachmentRef(Contract):
    CONTRACT = "AttachmentRef"

    id: str
    media_type: str
    reference: str
    name: str | None = None
    contract_version: str | None = None


@dataclass(frozen=True)
class IncomingMessage(Contract):
    CONTRACT = "IncomingMessage"
    _ALLOW_EMPTY = frozenset({"text"})

    id: str
    source_channel: str
    sender_id: str
    text: str
    received_at: datetime
    privacy_class: PrivacyClass
    attachments: tuple[AttachmentRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)
    contract_version: str | None = None

    def _check(self):
        _check_channel("source_channel", self.source_channel)
        require(self.text.strip() or self.attachments, "text: empty message without attachments")


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass(frozen=True)
class OutgoingNotification(Contract):
    CONTRACT = "OutgoingNotification"

    id: str
    target_channel: str
    recipient: str
    created_at: datetime
    priority: NotificationPriority
    privacy_class: PrivacyClass
    text: str | None = None
    content_ref: str | None = None
    attention_decision_id: str | None = None
    in_reply_to_message_id: str | None = None
    contract_version: str | None = None

    def _check(self):
        _check_channel("target_channel", self.target_channel)
        require(self.text is not None or self.content_ref is not None, "content: needs text or content_ref")
