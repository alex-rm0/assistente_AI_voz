from __future__ import annotations

import json
from pathlib import Path

from assistant.anthropic_provider import PAID_CALL_CONFIRMATION_ENV
from assistant.model_provider import ModelResponse
from assistant.model_router import AutomaticRoutingConfig, ModelRouter, ModelRoutingConfig, ModelUsageBudget
from assistant.model_runtime import ModelRuntimeBridge, UserSettingsStore, human_reason_label
from assistant.secret_storage import ANTHROPIC_SECRET_NAME, TAVILY_SECRET_NAME, InMemorySecretStorage


def make_bridge(
    tmp_path: Path,
    *,
    mode: str = "local",
    env: dict[str, str] | None = None,
    cli_locked: bool = False,
    secret_storage: InMemorySecretStorage | None = None,
    automatic_claude_enabled: bool = False,
):
    storage = secret_storage or InMemorySecretStorage()
    environment = env or {}
    budget = ModelUsageBudget(tmp_path / "usage.json")
    router = ModelRouter(
        ModelRoutingConfig(
            mode=mode,
            mode_source="test",
            automatic=AutomaticRoutingConfig(
                claude_enabled=automatic_claude_enabled,
                daily_budget_usd=0.25,
                max_single_call_estimated_usd=0.05,
            ),
        ),
        ollama_model="llama3.1:8b",
        anthropic_model="claude-haiku-4-5-20251001",
        budget=budget,
        env=environment,
        anthropic_key_available=lambda: bool(environment.get(ANTHROPIC_SECRET_NAME) or storage.has_secret(ANTHROPIC_SECRET_NAME)),
    )
    store = UserSettingsStore(tmp_path / "user_settings.json")
    return (
        ModelRuntimeBridge(
            router=router,
            budget=budget,
            store=store,
            env=environment,
            cli_locked=cli_locked,
            secret_storage=storage,
            model_routing_mode_source="test",
            automatic_claude_enabled_source="test",
        ),
        router,
        store,
        budget,
    )


def test_user_settings_store_persists_active_voice_id_without_secrets(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path / "user_settings.json")

    store.update(active_voice_id="portuguese_ptpt_primary", api_key="secret")

    data = store.load()
    assert data["active_voice_id"] == "portuguese_ptpt_primary"
    assert "api_key" not in data


def test_user_settings_store_persists_response_language_without_secrets(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path / "user_settings.json")

    store.update(response_language="en-GB", api_key="secret")

    data = store.load()
    assert data["response_language"] == "en-GB"
    assert "api_key" not in data


def test_ui_reads_real_mode_and_local_changes_backend(tmp_path: Path) -> None:
    bridge, router, store, _budget = make_bridge(tmp_path, mode="automatic")

    payload = bridge.set_mode("local")

    assert router.config.mode == "local"
    assert payload["mode"] == "local"
    assert payload["configured_model_mode"] == "local"
    assert payload["execution_path"] == "idle"
    assert payload["execution_provider"] == "ollama"
    assert payload["execution_model"] == "NONE"
    assert payload["provider"] == "ollama"
    assert payload["model"] == "NONE"
    assert json.loads(store.path.read_text(encoding="utf-8"))["model_routing_mode"] == "local"


def test_claude_mode_requires_key_and_paid_authorization(tmp_path: Path) -> None:
    bridge, router, _store, _budget = make_bridge(tmp_path, env={})

    missing_key = bridge.set_mode("claude")

    assert router.config.mode == "local"
    assert missing_key["state"] == "error"
    assert missing_key["provider_error_type"] == "missing_api_key"
    assert "API KEY NOT CONFIGURED" in missing_key["note"]

    bridge, router, _store, _budget = make_bridge(tmp_path, env={"ANTHROPIC_API_KEY": "sk-ant-test-value"})
    paid_disabled = bridge.set_mode("claude")

    assert router.config.mode == "local"
    assert paid_disabled["state"] == "error"
    assert paid_disabled["provider_error_type"] == "paid_calls_not_confirmed"
    assert "sk-ant-test-value" not in json.dumps(paid_disabled)


def test_claude_mode_changes_backend_when_allowed(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "sk-ant-test-value", PAID_CALL_CONFIRMATION_ENV: "true"}
    bridge, router, _store, _budget = make_bridge(tmp_path, env=env)

    payload = bridge.set_mode("claude")

    assert router.config.mode == "claude"
    assert payload["mode"] == "claude"
    assert payload["configured_model_mode"] == "claude"
    assert payload["execution_path"] == "idle"
    assert payload["execution_provider"] == "anthropic"
    assert payload["execution_model"] == "NONE"
    assert payload["provider"] == "anthropic"
    assert payload["model"] == "NONE"
    assert payload["paid_call"] is True
    assert "sk-ant-test-value" not in json.dumps(payload)


def test_automatic_mode_and_auto_claude_preference_persist(tmp_path: Path) -> None:
    env = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}
    bridge, router, store, _budget = make_bridge(tmp_path, env=env)

    bridge.set_mode("automatic")
    payload = bridge.set_automatic_claude_enabled(True)

    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert router.config.mode == "automatic"
    assert router.config.automatic.claude_enabled is True
    assert payload["automatic_claude_enabled"] is True
    assert data["model_routing_mode"] == "automatic"
    assert data["automatic_claude_enabled"] is True


def test_automatic_mode_with_claude_disabled_reports_cloud_routing_disabled(tmp_path: Path) -> None:
    bridge, router, _store, _budget = make_bridge(tmp_path)

    payload = bridge.set_mode("automatic")

    assert router.config.mode == "automatic"
    assert payload["reason_label"] == "Cloud routing disabled"
    assert payload["automatic_claude_enabled"] is False


def test_cli_override_blocks_ui_mode_change(tmp_path: Path) -> None:
    bridge, router, _store, _budget = make_bridge(tmp_path, mode="local", cli_locked=True)

    payload = bridge.set_mode("automatic")

    assert router.config.mode == "local"
    assert payload["state"] == "error"
    assert payload["mode_locked"] is True
    assert payload["provider_error_type"] == "cli_locked"


def test_telemetry_payload_uses_real_tokens_cost_and_budget(tmp_path: Path) -> None:
    bridge, _router, _store, budget = make_bridge(tmp_path)
    budget.register(
        ModelResponse(
            text="ok",
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            input_tokens=1000,
            output_tokens=100,
            latency_ms=1200,
            estimated_cost_usd=0.0015,
        )
    )

    payload = bridge.telemetry_payload(
        {
            "model_routing_mode": "automatic",
            "model_routing_provider": "anthropic",
            "model_routing_model": "claude-haiku-4-5-20251001",
            "model_routing_reason_code": "structured_summary",
            "model_routing_paid_call": True,
            "llm_calls": 1,
            "input_tokens": 1000,
            "output_tokens": 100,
            "estimated_cost_usd": 0.0015,
            "llm_call_tokens": [{"latency_ms": 1200, "provider": "anthropic", "model": "claude-haiku-4-5-20251001"}],
            "llm_call_details": [{"component": "RESPONSE_COMPOSER"}],
        }
    )

    assert payload["provider"] == "anthropic"
    assert payload["configured_model_mode"] == "automatic"
    assert payload["execution_path"] == "llm"
    assert payload["execution_provider"] == "anthropic"
    assert payload["execution_model"] == "claude-haiku-4-5-20251001"
    assert payload["reason_label"] == "Structured summary"
    assert payload["latency_ms"] == 1200
    assert payload["input_tokens"] == 1000
    assert payload["output_tokens"] == 100
    assert payload["estimated_cost_usd"] == 0.0015
    assert payload["daily_used_usd"] == 0.0015


def test_fast_path_and_memory_show_zero_tokens_and_cost(tmp_path: Path) -> None:
    bridge, _router, _store, _budget = make_bridge(tmp_path)

    payload = bridge.telemetry_payload({"selected_path": "MEMORY_RECALL", "llm_calls": 0})

    assert payload["provider"] == "memory"
    assert payload["configured_model_mode"] == "local"
    assert payload["execution_path"] == "memory_recall"
    assert payload["execution_provider"] == "memory"
    assert payload["execution_model"] == "NONE"
    assert payload["model"] == "NONE"
    assert payload["input_tokens"] is None
    assert payload["output_tokens"] is None
    assert payload["estimated_cost_usd"] == 0.0
    assert payload["reason_label"] == "Memory recall"


def test_telemetry_payload_preserves_semantic_extraction_fields(tmp_path: Path) -> None:
    bridge, _router, _store, _budget = make_bridge(tmp_path)

    payload = bridge.telemetry_payload(
        {
            "selected_path": "RESEARCH_REQUEST",
            "llm_calls": 0,
            "semantic_extraction_calls": 1,
            "semantic_extraction_status": "all_rejected",
            "semantic_candidates_before_validation": 3,
            "semantic_candidates_after_validation": 0,
            "semantic_extraction_failure_reason": "invalid_destination_candidate",
            "semantic_extraction_batches": 1,
            "semantic_extraction_latency_ms": 1234,
            "semantic_extraction_input_chars": 2048,
            "semantic_extraction_sources_used": 4,
        }
    )

    assert payload["semantic_extraction_calls"] == 1
    assert payload["semantic_extraction_status"] == "all_rejected"
    assert payload["semantic_candidates_before_validation"] == 3
    assert payload["semantic_candidates_after_validation"] == 0
    assert payload["semantic_extraction_failure_reason"] == "invalid_destination_candidate"
    assert payload["semantic_extraction_batches"] == 1
    assert payload["semantic_extraction_latency_ms"] == 1234
    assert payload["semantic_extraction_input_chars"] == 2048
    assert payload["semantic_extraction_sources_used"] == 4


def test_social_fast_path_payload_does_not_inherit_previous_paid_call(tmp_path: Path) -> None:
    bridge, _router, _store, budget = make_bridge(tmp_path)
    budget.register(
        ModelResponse(
            text="ok",
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            input_tokens=1000,
            output_tokens=100,
            latency_ms=1200,
            estimated_cost_usd=0.0015,
        )
    )

    payload = bridge.telemetry_payload(
        {
            "selected_path": "SOCIAL_PATH",
            "response_source": "SOCIAL_FAST_PATH",
            "model_routing_mode": "automatic",
            "model_routing_provider": "anthropic",
            "model_routing_model": "claude-haiku-4-5-20251001",
            "model_routing_reason_code": "document_synthesis",
            "model_routing_paid_call": True,
            "llm_calls": 0,
            "estimated_cost_usd": 0.0,
        }
    )

    assert payload["provider"] == "local"
    assert payload["configured_model_mode"] == "automatic"
    assert payload["execution_path"] == "social_fast_path"
    assert payload["execution_provider"] == "local"
    assert payload["execution_model"] == "NONE"
    assert payload["model"] == "NONE"
    assert payload["reason_code"] == "social_fast_path"
    assert payload["paid_call"] is False
    assert payload["estimated_cost_usd"] == 0.0
    assert payload["daily_used_usd"] == 0.0015


def test_reason_codes_are_humanized() -> None:
    assert human_reason_label("paid_calls_not_confirmed") == "Cloud disabled"
    assert human_reason_label("daily_budget_exceeded") == "Daily budget reached"
    assert human_reason_label("unknown_reason") == "Unknown Reason"


def test_system_command_and_direct_button_path_change_mode_the_same_way(tmp_path: Path) -> None:
    bridge, router, _store, _budget = make_bridge(tmp_path, mode="local")

    direct = bridge.set_mode("automatic")
    bridge.set_mode("local")
    command = bridge.execute_command({"intent": "set_model_mode", "parameters": {"mode": "automatic"}, "source": "ui"})

    assert direct["mode"] == command["mode"] == "automatic"
    assert router.config.mode == "automatic"


def test_secret_key_is_never_returned_in_payload_or_user_settings(tmp_path: Path) -> None:
    secret_storage = InMemorySecretStorage()
    bridge, _router, store, _budget = make_bridge(tmp_path, secret_storage=secret_storage)

    payload = bridge.execute_command({"intent": "save_anthropic_key", "parameters": {"api_key": "secret-value"}, "source": "ui"})

    serialized = json.dumps(payload)
    assert payload["api_key_configured"] is True
    assert payload["api_key_source"] == "secure_storage"
    assert "secret-value" not in serialized
    assert "secret-value" not in store.path.read_text(encoding="utf-8") if store.path.exists() else True


def test_claude_failure_keeps_active_runtime_state_local(tmp_path: Path) -> None:
    bridge, router, _store, _budget = make_bridge(tmp_path, mode="local", env={})

    before = bridge.current_payload()
    error = bridge.execute_command({"intent": "set_model_mode", "parameters": {"mode": "claude"}, "source": "ui"})
    after = bridge.current_payload()

    assert router.config.mode == "local"
    assert before["mode"] == after["mode"] == "local"
    assert error["state"] == "error"
    assert error["provider_error_type"] == "missing_api_key"


def test_secret_storage_unavailable_blocks_save_without_file_fallback(tmp_path: Path) -> None:
    storage = InMemorySecretStorage(available=False)
    bridge, _router, store, _budget = make_bridge(tmp_path, secret_storage=storage)

    payload = bridge.save_anthropic_key("secret-value")

    assert payload["state"] == "error"
    assert payload["provider_error_type"] == "secret_storage_unavailable"
    assert payload["secret_storage_available"] is False
    assert not store.path.exists()
    assert "secret-value" not in json.dumps(payload)


def test_environment_key_wins_over_secure_storage_and_cannot_be_removed(tmp_path: Path) -> None:
    storage = InMemorySecretStorage()
    storage.set_secret(ANTHROPIC_SECRET_NAME, "stored-secret")
    bridge, _router, _store, _budget = make_bridge(tmp_path, env={ANTHROPIC_SECRET_NAME: "env-secret"}, secret_storage=storage)

    payload = bridge.current_payload()
    removed = bridge.remove_anthropic_key()

    assert payload["api_key_configured"] is True
    assert payload["api_key_source"] == "environment"
    assert bridge.get_anthropic_api_key() == "env-secret"
    assert removed["state"] == "error"
    assert removed["provider_error_type"] == "environment_key"
    assert "env-secret" not in json.dumps(payload)
    assert "stored-secret" not in json.dumps(payload)


def test_automatic_claude_toggle_does_not_change_current_mode(tmp_path: Path) -> None:
    env = {ANTHROPIC_SECRET_NAME: "sk-ant-test-value", PAID_CALL_CONFIRMATION_ENV: "true"}
    bridge, router, store, _budget = make_bridge(tmp_path, mode="local", env=env)

    payload = bridge.set_automatic_claude_enabled(True)

    assert router.config.mode == "local"
    assert payload["mode"] == "local"
    assert payload["automatic_claude_enabled"] is True
    assert payload["automatic_claude_enabled_source"] == "ui"
    assert json.loads(store.path.read_text(encoding="utf-8"))["automatic_claude_enabled"] is True


def test_automatic_claude_toggle_can_be_enabled_without_key_but_reports_unavailable(tmp_path: Path) -> None:
    bridge, router, _store, _budget = make_bridge(tmp_path, mode="automatic", env={})

    payload = bridge.set_automatic_claude_enabled(True)

    assert router.config.mode == "automatic"
    assert payload["automatic_claude_enabled"] is True
    assert payload["api_key_configured"] is False
    assert payload["api_key_source"] == "none"


def test_test_connection_is_configuration_only_and_does_not_expose_key(tmp_path: Path) -> None:
    env = {ANTHROPIC_SECRET_NAME: "sk-ant-test-value", PAID_CALL_CONFIRMATION_ENV: "true"}
    bridge, _router, _store, _budget = make_bridge(tmp_path, env=env)

    payload = bridge.test_anthropic_connection()

    assert payload["state"] == "configuration_status"
    assert payload["api_key_source"] == "environment"
    assert payload["paid_calls_enabled"] is True
    assert "sk-ant-test-value" not in json.dumps(payload)


def test_tavily_secret_key_is_saved_securely_and_never_returned(tmp_path: Path) -> None:
    storage = InMemorySecretStorage()
    bridge, _router, store, _budget = make_bridge(tmp_path, secret_storage=storage)

    payload = bridge.execute_command({"intent": "save_tavily_key", "parameters": {"api_key": "tvly-test-secret"}, "source": "ui"})

    serialized = json.dumps(payload)
    assert storage.get_secret(TAVILY_SECRET_NAME) == "tvly-test-secret"
    assert payload["tavily_key_configured"] is True
    assert payload["tavily_key_source"] == "secure_storage"
    assert "tvly-test-secret" not in serialized
    assert "tvly-test-secret" not in store.path.read_text(encoding="utf-8") if store.path.exists() else True


def test_tavily_status_reports_only_configured_without_secret(tmp_path: Path) -> None:
    storage = InMemorySecretStorage()
    storage.set_secret(TAVILY_SECRET_NAME, "tvly-stored-secret")
    bridge, _router, _store, _budget = make_bridge(tmp_path, secret_storage=storage)

    payload = bridge.current_payload(state="system_status")

    assert payload["tavily_key_configured"] is True
    assert payload["tavily_key_source"] == "secure_storage"
    assert "tvly-stored-secret" not in json.dumps(payload)


def test_tavily_remove_restores_not_configured_state(tmp_path: Path) -> None:
    storage = InMemorySecretStorage()
    storage.set_secret(TAVILY_SECRET_NAME, "tvly-stored-secret")
    bridge, _router, _store, _budget = make_bridge(tmp_path, secret_storage=storage)

    payload = bridge.remove_tavily_key()

    assert storage.has_secret(TAVILY_SECRET_NAME) is False
    assert payload["tavily_key_configured"] is False
    assert payload["tavily_key_source"] == "none"
    assert "tvly-stored-secret" not in json.dumps(payload)


def test_tavily_environment_key_wins_and_cannot_be_removed(tmp_path: Path) -> None:
    storage = InMemorySecretStorage()
    storage.set_secret(TAVILY_SECRET_NAME, "tvly-stored-secret")
    bridge, _router, _store, _budget = make_bridge(tmp_path, env={TAVILY_SECRET_NAME: "tvly-env-secret"}, secret_storage=storage)

    payload = bridge.current_payload()
    removed = bridge.remove_tavily_key()

    assert payload["tavily_key_configured"] is True
    assert payload["tavily_key_source"] == "environment"
    assert bridge.get_tavily_api_key() == "tvly-env-secret"
    assert removed["state"] == "error"
    assert removed["provider_error_type"] == "environment_key"
    assert "tvly-env-secret" not in json.dumps(payload)
    assert "tvly-stored-secret" not in json.dumps(payload)


def test_tavily_test_connection_without_key_is_clear_and_does_not_call_network(tmp_path: Path, monkeypatch) -> None:
    class FailingProvider:
        def __init__(self, **_kwargs):
            raise AssertionError("network provider should not be built")

    monkeypatch.setattr("assistant.model_runtime.TavilyResearchProvider", FailingProvider)
    bridge, _router, _store, _budget = make_bridge(tmp_path)

    payload = bridge.test_tavily_connection()

    assert payload["state"] == "error"
    assert payload["provider_error_type"] == "missing_tavily_api_key"
    assert payload["tavily_key_configured"] is False


def test_tavily_test_connection_uses_bounded_provider_without_exposing_secret(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    class FakeTavilyProvider:
        def __init__(self, *, api_key_getter, config):
            seen["key_getter"] = api_key_getter
            seen["max_results"] = config.max_results
            seen["timeout_seconds"] = config.timeout_seconds

        def search_with_status(self, query: str):
            seen["query"] = query
            assert seen["key_getter"]() == ("tvly-test-secret", "secure_storage")
            return [{"title": "OpenAI", "url": "https://openai.com", "snippet": "Official site"}], "completed", ""

    monkeypatch.setattr("assistant.model_runtime.TavilyResearchProvider", FakeTavilyProvider)
    storage = InMemorySecretStorage()
    storage.set_secret(TAVILY_SECRET_NAME, "tvly-test-secret")
    bridge, _router, _store, _budget = make_bridge(tmp_path, secret_storage=storage)

    payload = bridge.test_tavily_connection()

    assert payload["state"] == "configuration_status"
    assert payload["tavily_test_status"] == "success"
    assert payload["tavily_key_configured"] is True
    assert seen == {
        "key_getter": seen["key_getter"],
        "max_results": 1,
        "timeout_seconds": 8.0,
        "query": "OpenAI official website",
    }
    assert "tvly-test-secret" not in json.dumps(payload)


def test_tavily_test_connection_sanitizes_provider_failure(tmp_path: Path, monkeypatch) -> None:
    class FakeTavilyProvider:
        def __init__(self, **_kwargs):
            pass

        def search_with_status(self, query: str):
            return [], "failed", "tvly-test-secret timeout"

    monkeypatch.setattr("assistant.model_runtime.TavilyResearchProvider", FakeTavilyProvider)
    storage = InMemorySecretStorage()
    storage.set_secret(TAVILY_SECRET_NAME, "tvly-test-secret")
    bridge, _router, _store, _budget = make_bridge(tmp_path, secret_storage=storage)

    payload = bridge.test_tavily_connection()

    assert payload["state"] == "error"
    assert payload["provider_error_type"] == "tavily_connection_failed"
    assert payload["tavily_test_status"] == "failed"
    assert "tvly-test-secret" not in json.dumps(payload)
