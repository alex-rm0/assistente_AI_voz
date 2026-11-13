"""Channel boundary (North Star §3.3): what every ChannelAdapter translates to and from."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from assistant.core.contracts._base import IDENTIFIER_RE, Contract, require
from assistant.core.contracts.privacy import PrivacyClass


# Channels that run on Alexandre's own machine; every other channel (today
# Telegram) is an external service.
LOCAL_CHANNELS = frozenset({"echo_os"})
# Channels allowed to answer a confirmation, wherever the action was requested
# (Alexandre, 12/11). An unknown channel never confirms.
CONFIRMATION_CHANNELS = frozenset({"echo_os", "telegram"})
_CONTENT_FREE_ONLY = frozenset({PrivacyClass.PRIVATE, PrivacyClass.RESTRICTED})


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
    # A notice that something arrived ("Chegou um email importante de X"),
    # without the content itself.
    content_free: bool = False
    attention_decision_id: str | None = None
    in_reply_to_message_id: str | None = None
    contract_version: str | None = None

    def _check(self):
        _check_channel("target_channel", self.target_channel)
        require(self.text is not None or self.content_ref is not None, "content: needs text or content_ref")
        if self.target_channel not in LOCAL_CHANNELS and self.privacy_class in _CONTENT_FREE_ONLY:
            # Alexandre, 12/11: PRIVATE/RESTRICTED content never leaves for an
            # external channel; only a content-free notice does.
            require(self.content_free, f"content_free: {self.privacy_class.value} content never goes to "
                                       f"{self.target_channel}; only a content-free notice")
            require(self.content_ref is None, "content_ref: a content-free notice carries no content reference")
