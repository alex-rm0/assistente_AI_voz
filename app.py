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
from assistant.model_provider import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_MODEL,
    OllamaProvider,
    ProviderBackedLLM,
    resolve_model_provider,
)
from assistant.model_router import AutomaticRoutingConfig, ModelRouter, ModelRoutingConfig, ModelUsageBudget, RoutedLLM, resolve_model_routing_config
from assistant.model_runtime import ModelRuntimeBridge, UserSettingsStore
from assistant.personal_model import PersonalModel
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.session_manager import SessionManager
from assistant.secret_storage import WindowsCredentialManagerSecretStorage, resolve_anthropic_api_key
from assistant.tool_registry import tool_registry
from assistant.voice_input import MicrophoneCheckError, check_microphone, check_voice_runtime

# Importing assistant.tools registers all built-in tools automatically.
import assistant.tools  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "config" / "settings.json"


def load_settings() -> dict[str, Any]:
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_ollama_model(
    *,
    cli_model: str | None = None,
    env: dict[str, str] | None = None,
    settings: dict[str, Any] | None = None,
    default_model: str = DEFAULT_OLLAMA_MODEL,
) -> tuple[str, str]:
    """Compatibility wrapper around the provider resolver for old tests/imports."""
    resolved = resolve_model_provider(
        cli_provider="ollama",
        cli_model=cli_model,
        env=env,
        settings=settings if settings is not None else {},
    )
    source = "settings.json" if resolved.model_source == "settings.json:ollama" else resolved.model_source
    if source == "default" and default_model != DEFAULT_OLLAMA_MODEL:
        return default_model, "default"
    return resolved.model or default_model, source


def main() -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from assistant.anthropic_provider import AnthropicProvider
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Echo desktop app")
    parser.add_argument("--ui", choices=("classic", "echo-os"), default="classic")
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="", help="Model provider for this run")
    parser.add_argument("--model", default="", help="Model name for this run")
    parser.add_argument("--model-mode", choices=("local", "claude", "automatic"), default="", help="Model routing mode")
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
    user_settings_store = UserSettingsStore(data_path / "user_settings.json")
    secret_storage = WindowsCredentialManagerSecretStorage()
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

    resolved_model = resolve_model_provider(
        cli_provider=args.provider,
        cli_model=args.model,
        settings=settings,
    )
    routing_config = resolve_model_routing_config(cli_mode=args.model_mode, settings=settings)
    cli_locked_model_routing = bool(args.provider or args.model_mode)
    automatic_claude_enabled_source = _settings_automatic_source(settings)
    if not cli_locked_model_routing and not os.environ.get("ECHO_MODEL_MODE", "").strip():
        preferences = user_settings_store.preferences()
        automatic = routing_config.automatic
        routing_config = ModelRoutingConfig(
            mode=preferences.model_routing_mode or routing_config.mode,
            mode_source=preferences.model_routing_mode_source or routing_config.mode_source,
            automatic=AutomaticRoutingConfig(
                claude_enabled=(
                    automatic.claude_enabled
                    if preferences.automatic_claude_enabled is None
                    else preferences.automatic_claude_enabled
                ),
                daily_budget_usd=preferences.daily_budget_usd if preferences.daily_budget_usd is not None else automatic.daily_budget_usd,
                max_single_call_estimated_usd=(
                    preferences.max_single_call_estimated_usd
                    if preferences.max_single_call_estimated_usd is not None
                    else automatic.max_single_call_estimated_usd
                ),
            ),
        )
        if preferences.automatic_claude_enabled_source:
            automatic_claude_enabled_source = preferences.automatic_claude_enabled_source
    print("[MODEL CONFIG]")
    print(f"provider={resolved_model.provider}")
    print(f"provider_source={resolved_model.provider_source}")
    print(f"model={resolved_model.model}")
    print(f"model_source={resolved_model.model_source}")
    print(f"model_routing_mode={routing_config.mode}")
    print(f"model_routing_mode_source={routing_config.mode_source}")

    default_system_prompt = get_base_system_prompt()
    ollama_config = settings.get("ollama", {}) if isinstance(settings.get("ollama", {}), dict) else {}
    anthropic_config = settings.get("anthropic", {}) if isinstance(settings.get("anthropic", {}), dict) else {}
    model_settings = settings.get("model", {}) if isinstance(settings.get("model", {}), dict) else {}
    cli_model_for_ollama = bool(args.model and (args.provider == "ollama" or (not args.provider and routing_config.mode == "local")))
    cli_model_for_anthropic = bool(args.model and (args.provider == "anthropic" or (not args.provider and routing_config.mode in {"claude", "automatic"})))
    ollama_model_name = (
        args.model
        if cli_model_for_ollama
        else resolved_model.model
        if resolved_model.provider == "ollama"
        else str(model_settings.get("name") or DEFAULT_OLLAMA_MODEL)
    )
    anthropic_model_name = (
        args.model
        if cli_model_for_anthropic
        else resolved_model.model
        if resolved_model.provider == "anthropic"
        else str(anthropic_config.get("model") or DEFAULT_ANTHROPIC_MODEL)
    )
    ollama_provider = OllamaProvider(
        model=ollama_model_name,
        base_url=str(ollama_config.get("base_url") or "http://127.0.0.1:11434"),
        timeout_seconds=int(ollama_config.get("timeout_seconds", 120)),
    )
    anthropic_provider = AnthropicProvider(
        model=anthropic_model_name,
        api_key_getter=lambda: resolve_anthropic_api_key(os.environ, secret_storage)[0],
        base_url=str(anthropic_config.get("base_url") or "https://api.anthropic.com"),
        timeout_seconds=int(anthropic_config.get("timeout_seconds", 120)),
        max_tokens=int(anthropic_config.get("max_tokens", 1024)),
    )
    if args.provider:
        # Explicit provider remains a manual override. No fallback is allowed:
        # provider configuration errors are surfaced by AssistantEngine.
        provider = anthropic_provider if resolved_model.provider == "anthropic" else ollama_provider
        llm = ProviderBackedLLM(
            provider=provider,
            system_prompt=default_system_prompt,
            model_source=resolved_model.model_source,
        )
        budget = None
        router = None
    else:
        budget = ModelUsageBudget(data_path / "model_routing_usage.json")
        router = ModelRouter(
            routing_config,
            ollama_model=ollama_provider.model,
            anthropic_model=anthropic_provider.model,
            budget=budget,
            anthropic_key_available=lambda: bool(resolve_anthropic_api_key(os.environ, secret_storage)[0]),
        )
        llm = RoutedLLM(
            providers={"ollama": ollama_provider, "anthropic": anthropic_provider},
            router=router,
            system_prompt=default_system_prompt,
            model_source=resolved_model.model_source,
        )
    model_runtime = ModelRuntimeBridge(
        router=router,
        budget=budget,
        store=user_settings_store,
        cli_locked=cli_locked_model_routing,
        ollama_model=ollama_provider.model,
        anthropic_model=anthropic_provider.model,
        secret_storage=secret_storage,
        settings_source="settings.json",
        model_routing_mode_source=routing_config.mode_source,
        automatic_claude_enabled_source=automatic_claude_enabled_source,
        context_observer_state="enabled" if observer_enabled else "disabled",
    )
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
        model_runtime=model_runtime,
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
            get_telemetry=engine.get_last_turn_telemetry,
            model_runtime=model_runtime,
        )
    else:
        from ui.main_window import MainWindow

        window = MainWindow(
            app_name=app_name,
            model_name=resolved_model.model,
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

def _settings_automatic_source(settings: dict[str, Any]) -> str:
    routing = settings.get("model_routing", {}) if isinstance(settings.get("model_routing", {}), dict) else {}
    automatic = routing.get("automatic", {}) if isinstance(routing.get("automatic", {}), dict) else {}
    return "settings.json" if "claude_enabled" in automatic else "default"


if __name__ == "__main__":
    raise SystemExit(main())
