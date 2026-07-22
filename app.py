from __future__ import annotations

import json
import os
import sys
import argparse
from datetime import date
from pathlib import Path
from typing import Any

from assistant.conversation import AssistantEngine
from assistant.context_observer import ContextObserver
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.personal_model import PersonalModel
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.session_manager import SessionManager
from assistant.tool_registry import tool_registry
from assistant.voice_input import MicrophoneCheckError, check_microphone, check_voice_runtime

# Importing assistant.tools registers all built-in tools automatically.
import assistant.tools  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "config" / "settings.json"


def load_settings() -> dict[str, Any]:
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from assistant.llm import OLLAMA_MODEL, OllamaClient, OllamaSettings
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Echo desktop app")
    parser.add_argument("--ui", choices=("classic", "echo-os"), default="classic")
    args, qt_args = parser.parse_known_args()

    settings = load_settings()

    app_name = settings.get("app_name", "AssistenteIA")
    debug = bool(settings.get("debug", False))
    debug_agent = bool(settings.get("DEBUG_AGENT", False))
    debug_performance = bool(settings.get("DEBUG_PERFORMANCE", False))
    debug_ollama_payload = bool(settings.get("DEBUG_OLLAMA_PAYLOAD", False))
    voice_config = settings.get("voice", {})
    voice_requested = bool(voice_config.get("enabled", False))
    voice_missing = check_voice_runtime() if voice_requested else []
    voice_microphone_ok = False
    voice_microphone_message = ""
    voice_sample_rate = int(voice_config.get("sample_rate", 44100))
    voice_input_device = voice_config.get("input_device", "default")
    voice_auto_select_input = bool(voice_config.get("auto_select_input", True))
    voice_silent_rms_threshold = float(voice_config.get("silent_rms_threshold", 0.001))
    voice_channels = int(voice_config.get("channels", 1))
    voice_probe_duration = float(voice_config.get("probe_duration_seconds", 0.5))
    voice_min_record_seconds = float(voice_config.get("min_record_seconds", 2.0))
    voice_preroll_ms = int(voice_config.get("preroll_ms", 500))
    voice_ready_delay_ms = int(voice_config.get("ready_delay_ms", 200))
    if voice_requested and not voice_missing:
        try:
            voice_microphone_message = check_microphone(
                sample_rate=voice_sample_rate,
                input_device=voice_input_device,
                auto_select=voice_auto_select_input,
                silent_rms_threshold=voice_silent_rms_threshold,
                channels=voice_channels,
                probe_duration=voice_probe_duration,
            )
            voice_microphone_ok = True
        except MicrophoneCheckError as exc:
            voice_microphone_message = str(exc)
    voice_runtime_enabled = voice_requested and not voice_missing
    voice_status = (
        "Voz pronta"
        if voice_runtime_enabled and voice_microphone_ok
        else "Microfone por testar: " + voice_microphone_message
        if voice_runtime_enabled and voice_microphone_message
        else "Microfone por testar"
        if voice_runtime_enabled
        else "Voz desligada"
        if not voice_requested
        else "Microfone indisponivel: " + voice_microphone_message
        if not voice_missing and voice_microphone_message
        else "Voz indisponivel: falta " + ", ".join(voice_missing)
    )
    presence = PresenceManager(settings.get("default_presence", "ACTIVE_CONVERSATION"))
    workspace_config = settings.get("workspace", {})
    workspace_path = (BASE_DIR / workspace_config.get("path", "workspace")).resolve()
    desktop_config = settings.get("desktop_actions", {})
    known_projects = desktop_config.get("known_projects", {"AssistenteIA": "."})

    memory_config = settings.get("memory", {})
    data_path = (BASE_DIR / memory_config.get("data_path", "data")).resolve()
    memory = ConversationMemory(
        data_path=data_path,
        history_file=memory_config.get("history_file", "history.json"),
        max_messages=int(memory_config.get("max_messages", 20)),
    )
    observer_config = settings.get("context_observer", {})
    context_observer = ContextObserver(
        data_path=data_path,
        project_root=BASE_DIR,
        db_file=observer_config.get("db_file", "context_observer.sqlite"),
        recent_files_limit=int(observer_config.get("recent_files_limit", 10)),
        summary_min_seconds=float(observer_config.get("summary_min_seconds", 60)),
        debug_context=bool(settings.get("DEBUG_CONTEXT", False)),
    )
    observer_enabled = bool(observer_config.get("enabled", True))
    observer_interval_ms = int(observer_config.get("interval_seconds", 15)) * 1000

    ollama_config = settings.get("ollama", {})
    model = os.environ.get("OLLAMA_MODEL")
    if not model:
        legacy_composer = os.environ.get("OLLAMA_COMPOSER_MODEL")
        legacy_critic = os.environ.get("OLLAMA_VOICE_CRITIC_MODEL")
        if legacy_composer or legacy_critic:
            print(
                "[OLLAMA CONFIG] aviso: OLLAMA_COMPOSER_MODEL/OLLAMA_VOICE_CRITIC_MODEL estao "
                "obsoletas e o Echo volta a usar um unico modelo; define OLLAMA_MODEL em vez disso."
            )
        model = legacy_composer or OLLAMA_MODEL
    print("[OLLAMA CONFIG]")
    print(f"model={model}")

    ollama_settings = OllamaSettings(
        base_url=ollama_config.get("base_url", "http://127.0.0.1:11434"),
        model=model,
        timeout_seconds=int(ollama_config.get("timeout_seconds", 120)),
        debug_performance=debug_performance,
        debug_ollama_payload=debug_ollama_payload,
    )
    default_system_prompt = get_base_system_prompt()
    llm = OllamaClient(settings=ollama_settings, system_prompt=default_system_prompt)
    long_term_memory = LongTermMemory(
        data_path=data_path,
        db_file=memory_config.get("long_term_db", "long_term_memory.sqlite"),
        embedder=llm,
    )
    personal_model = PersonalModel(
        data_path=data_path,
        db_file=memory_config.get("personal_model_db", "personal_model.sqlite"),
    )
    session_manager = SessionManager(
        data_path=data_path,
        db_file=memory_config.get("session_db", "session_manager.sqlite"),
    )
    context_observer.set_summary_callback(
        lambda summary: long_term_memory.remember_context_summary(
            summary.summary,
            event_date=date.fromtimestamp(summary.end_at),
            project=summary.project,
        )
    )
    engine = AssistantEngine(
        llm=llm,
        memory=memory,
        long_term_memory=long_term_memory,
        personal_model=personal_model,
        tools=tool_registry,
        workspace_path=workspace_path,
        base_system_prompt=default_system_prompt,
        debug=debug,
        debug_agent=debug_agent,
        debug_performance=debug_performance,
        debug_ollama_payload=debug_ollama_payload,
        presence_manager=presence,
        context_observer=context_observer,
        session_manager=session_manager,
        known_projects=known_projects,
        desktop_config=desktop_config,
        voice_enabled=voice_requested,
        voice_missing_dependencies=voice_missing,
        voice_microphone_ok=voice_microphone_ok,
        voice_microphone_message=voice_microphone_message,
        voice_sample_rate=voice_sample_rate,
        voice_input_device=voice_input_device,
        voice_auto_select_input=voice_auto_select_input,
        voice_silent_rms_threshold=voice_silent_rms_threshold,
        voice_channels=voice_channels,
        voice_probe_duration=voice_probe_duration,
        voice_min_record_seconds=voice_min_record_seconds,
        voice_model=voice_config.get("model", "base"),
        voice_language=voice_config.get("language", "pt"),
    )

    def change_presence(name: str) -> None:
        engine.set_presence_state(name)

    def observe_context() -> None:
        if observer_enabled and presence.can_observe_activity():
            context_observer.observe_once()

    qt_app = QApplication([sys.argv[0], *qt_args])
    if args.ui == "echo-os":
        from prototype_web_ui.window import EchoOSWindow

        window = EchoOSWindow(
            responder=engine.respond,
            title=app_name,
            clear_conversation=engine.clear_conversation,
            on_close=engine.close_session,
        )
    else:
        from ui.main_window import MainWindow

        window = MainWindow(
            app_name=app_name,
            model_name=ollama_settings.model,
            responder=engine.respond,
            clear_history=engine.clear_history,
            change_presence=change_presence,
            get_presence_state=engine.presence_state,
            get_pending_tasks=engine.pending_tasks_summary,
            get_pending_task_count=engine.pending_task_count,
            get_tasks_panel_expanded=engine.tasks_panel_expanded,
            set_tasks_panel_expanded=engine.set_tasks_panel_expanded,
            on_close=engine.close_session,
            presence_names=PresenceManager.names(),
            active_presence=engine.presence_state(),
            debug_contexts=debug_agent,
            get_context_debug=engine.context_debug,
            voice_available=voice_runtime_enabled,
            voice_status=voice_status,
            voice_model=voice_config.get("model", "base"),
            voice_language=voice_config.get("language", "pt"),
            voice_sample_rate=voice_sample_rate,
            voice_input_device=voice_input_device,
            voice_auto_select_input=voice_auto_select_input,
            voice_silent_rms_threshold=voice_silent_rms_threshold,
            voice_channels=voice_channels,
            voice_probe_duration=voice_probe_duration,
            voice_min_record_seconds=voice_min_record_seconds,
            voice_preroll_ms=voice_preroll_ms,
            voice_ready_delay_ms=voice_ready_delay_ms,
            initial_messages=[
                *engine.history(),
                {"role": "assistant", "content": engine.startup_greeting()},
            ],
        )
    observer_timer = QTimer(window)
    observer_timer.setInterval(max(1000, observer_interval_ms))
    observer_timer.timeout.connect(observe_context)
    observer_timer.start()
    window.context_observer = context_observer
    window.context_observer_timer = observer_timer
    window.show()
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
