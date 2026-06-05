from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.tool_registry import tool_registry

import assistant.tools  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "config" / "settings.json"

app = Flask(__name__)


def load_settings() -> dict[str, Any]:
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_engine() -> tuple[AssistantEngine, str, str]:
    from assistant.llm import OllamaClient, OllamaSettings

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
    model_name = ollama_settings.model
    assistant_config = settings.get("assistant", {})
    system_prompt = assistant_config.get("system_prompt", "")
    llm = OllamaClient(settings=ollama_settings, system_prompt=system_prompt)
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
        base_system_prompt=system_prompt,
    )
    return engine, app_name, model_name


engine, APP_NAME, MODEL_NAME = create_engine()


@app.route("/")
def index():
    history = engine.history()
    return render_template("index.html", app_name=APP_NAME, model_name=MODEL_NAME, history=history)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Mensagem vazia."}), 400

    try:
        response = engine.respond(user_message)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify({"response": response})


@app.route("/clear", methods=["POST"])
def clear():
    engine.clear_history()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
