from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.profiles import get_profile, profile_names
from assistant.tool_registry import tool_registry

# Importing assistant.tools registers all built-in tools automatically.
import assistant.tools  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "config" / "settings.json"


def load_settings() -> dict[str, Any]:
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from assistant.llm import OllamaClient, OllamaSettings
    from ui.main_window import MainWindow

    settings = load_settings()

    app_name = settings.get("app_name", "AssistenteIA")
    workspace_config = settings.get("workspace", {})
    workspace_path = (BASE_DIR / workspace_config.get("path", "workspace")).resolve()

    memory_config = settings.get("memory", {})
    data_path = (BASE_DIR / memory_config.get("data_path", "data")).resolve()
    memory = ConversationMemory(
        data_path=data_path,
        history_file=memory_config.get("history_file", "history.json"),
        max_messages=int(memory_config.get("max_messages", 20)),
    )

    ollama_config = settings.get("ollama", {})
    ollama_settings = OllamaSettings(
        base_url=ollama_config.get("base_url", "http://127.0.0.1:11434"),
        model=ollama_config.get("model", "llama3.2"),
        timeout_seconds=int(ollama_config.get("timeout_seconds", 120)),
    )
    default_profile = get_profile("Geral")
    llm = OllamaClient(settings=ollama_settings, system_prompt=default_profile.system_prompt)
    long_term_memory = LongTermMemory(
        data_path=data_path,
        db_file=memory_config.get("long_term_db", "long_term_memory.sqlite"),
        embedder=llm,
    )
    engine = AssistantEngine(
        llm=llm,
        memory=memory,
        long_term_memory=long_term_memory,
        tools=tool_registry,
        workspace_path=workspace_path,
        base_system_prompt=default_profile.system_prompt,
    )

    def change_profile(name: str) -> None:
        engine.set_profile(get_profile(name).system_prompt)

    qt_app = QApplication(sys.argv)
    window = MainWindow(
        app_name=app_name,
        model_name=ollama_settings.model,
        responder=engine.respond,
        clear_history=engine.clear_history,
        change_profile=change_profile,
        profile_names=profile_names(),
        initial_messages=engine.history(),
    )
    window.show()
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
