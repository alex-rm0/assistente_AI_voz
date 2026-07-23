from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SystemCommand:
    """Common command envelope for UI now and voice/keyboard/API later."""

    intent: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str = "ui"


SYSTEM_COMMAND_INTENTS = {
    "set_model_mode",
    "open_panel_section",
    "recenter_echo",
    "reset_workspace",
    "lock_workspace_item",
    "get_model_status",
    "get_budget_status",
    "set_automatic_claude_enabled",
    "set_model_budget",
    "save_anthropic_key",
    "remove_anthropic_key",
    "test_anthropic_connection",
}


def parse_system_command(payload: dict[str, Any] | str, *, source: str = "ui") -> SystemCommand:
    if isinstance(payload, str):
        return SystemCommand(intent=payload.strip(), source=source)
    if not isinstance(payload, dict):
        return SystemCommand(intent="", source=source)
    intent = str(payload.get("intent") or "").strip()
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    command_source = str(payload.get("source") or source or "ui").strip()
    return SystemCommand(intent=intent, parameters=dict(parameters), source=command_source)
