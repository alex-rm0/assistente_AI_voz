"""Health model (North Star §2.1.10): each main component reports its own state."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from assistant.core.contracts._base import IDENTIFIER_RE, Contract, require


class HealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ComponentHealth(Contract):
    CONTRACT = "ComponentHealth"

    component: str
    state: HealthState
    checked_at: datetime
    reason: str | None = None
    since: datetime | None = None
    contract_version: str | None = None

    def _check(self):
        require(IDENTIFIER_RE.match(self.component),
                f"component: expected a lowercase identifier like 'provider.ollama', got {self.component!r}")
        if self.state is not HealthState.HEALTHY:
            require(self.reason is not None, "reason: a degraded or unavailable component must say why")
        if self.since is not None:
            require(self.since <= self.checked_at, "since: must not be after checked_at")
