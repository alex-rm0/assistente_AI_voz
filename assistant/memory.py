from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data"
DEFAULT_HISTORY_FILE = "history.json"


def load_history(data_path: Path | None = None, history_file: str = DEFAULT_HISTORY_FILE) -> list[dict[str, str]]:
    """Load saved conversation messages from the local data folder."""

    history_path = _history_path(data_path, history_file)
    if not history_path.exists():
        return []

    try:
        data: Any = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    # Keep only valid chat messages. This avoids loading malformed JSON content.
    messages: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    return messages


def save_history(
    messages: list[dict[str, str]],
    data_path: Path | None = None,
    history_file: str = DEFAULT_HISTORY_FILE,
) -> None:
    """Save conversation messages to the local data folder."""

    history_path = _history_path(data_path, history_file)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(messages, ensure_ascii=True, indent=2), encoding="utf-8")


def clear_history(data_path: Path | None = None, history_file: str = DEFAULT_HISTORY_FILE) -> None:
    """Clear the saved conversation history."""

    save_history([], data_path, history_file)


class ConversationMemory:
    """Small wrapper around the local JSON history file."""

    def __init__(self, data_path: Path, history_file: str, max_messages: int = 20) -> None:
        self.data_path = data_path.resolve()
        self.history_file = history_file
        self.max_messages = max(1, max_messages)
        self.data_path.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, str]]:
        return load_history(self.data_path, self.history_file)[-self.max_messages :]

    def append_pair(self, user_message: str, assistant_message: str) -> None:
        messages = self.load()
        messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        self.save(messages[-self.max_messages :])

    def save(self, messages: list[dict[str, str]]) -> None:
        save_history(messages[-self.max_messages :], self.data_path, self.history_file)

    def clear(self) -> None:
        clear_history(self.data_path, self.history_file)


def _history_path(data_path: Path | None, history_file: str) -> Path:
    data_root = (data_path or DEFAULT_DATA_PATH).resolve()
    target = (data_root / history_file).resolve()

    if target != data_root and data_root not in target.parents:
        raise ValueError("History path must stay inside the data folder.")

    return target
