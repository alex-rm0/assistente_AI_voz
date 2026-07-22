from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class UIEvent:
    """Semantic event emitted by the backend for adaptive UI surfaces."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class UIEventAdapter:
    """Serializes internal semantic events into QWebChannel-safe JSON."""

    @staticmethod
    def serialize(event_type: str, payload: dict[str, Any] | None = None) -> str:
        clean_type = str(event_type or "").strip()
        if not clean_type:
            raise ValueError("UI event type is required.")
        event = UIEvent(type=clean_type, payload=payload or {})
        document = {"type": event.type, "timestamp": event.timestamp, **event.payload}
        return json.dumps(document, ensure_ascii=False)
