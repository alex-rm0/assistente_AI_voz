"""Paid model calls are switched on by a setting, not by an environment variable.

`allow_paid_calls` lives in data/user_settings.json and is off unless Alexandre
turns it on. ECHO_ALLOW_PAID_MODEL_CALLS only overrides it: "true" forces paid
calls on, "false" forces them off. The daily and per-call budget still apply.
"""
from __future__ import annotations

import json

import pytest

from assistant.anthropic_provider import PAID_CALL_CONFIRMATION_ENV, AnthropicProvider
from assistant.model_provider import ProviderConfigurationError
from assistant.model_router import (
    AutomaticRoutingConfig,
    ModelRouter,
    ModelRoutingConfig,
    ModelRoutingInput,
    paid_calls_allowed,
)
from assistant.model_runtime import ModelRuntimeBridge, UserSettingsStore, paid_calls_getter


@pytest.mark.parametrize("env,setting,expected", [
    ({}, None, False),                                   # nothing set: off
    ({}, False, False),
    ({}, True, True),                                    # the setting alone switches them on
    ({}, "true", False),                                 # only a real boolean counts
    ({PAID_CALL_CONFIRMATION_ENV: "true"}, False, True),   # the variable forces on
    ({PAID_CALL_CONFIRMATION_ENV: "TRUE"}, None, True),
    ({PAID_CALL_CONFIRMATION_ENV: "false"}, True, False),  # and forces off
    ({PAID_CALL_CONFIRMATION_ENV: ""}, True, True),        # empty is not an override
    ({PAID_CALL_CONFIRMATION_ENV: "maybe"}, True, True),
])
def test_paid_calls_allowed(env, setting, expected) -> None:
    assert paid_calls_allowed(env, setting) is expected


def test_setting_is_stored_in_user_settings_and_defaults_to_off(tmp_path) -> None:
    store = UserSettingsStore(tmp_path / "user_settings.json")
    allowed = paid_calls_getter(store, env={})

    assert allowed() is False
    store.update(allow_paid_calls=True)
    assert json.loads((tmp_path / "user_settings.json").read_text(encoding="utf-8"))["allow_paid_calls"] is True
    assert allowed() is True  # read again on every call: no restart needed
    store.update(allow_paid_calls=False)
    assert allowed() is False


def _router(allowed) -> ModelRouter:
    return ModelRouter(
        ModelRoutingConfig(mode="automatic", automatic=AutomaticRoutingConfig(claude_enabled=True)),
        env={}, anthropic_key_available=lambda: True, paid_calls_allowed=allowed,
    )


def test_router_follows_the_setting_while_running(tmp_path) -> None:
    store = UserSettingsStore(tmp_path / "user_settings.json")
    router = _router(paid_calls_getter(store, env={}))
    request = ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="Hoje estou cansado.", privacy_class="PERSONAL")

    off = router.decide(request)
    assert (off.provider, off.reason_code, off.fallback_reason) == ("ollama", "paid_calls_not_confirmed", "paid_calls_not_confirmed")

    store.update(allow_paid_calls=True)
    on = router.decide(request)
    assert (on.provider, on.paid_call) == ("anthropic", True)


def test_budget_still_applies_with_the_setting_on(tmp_path) -> None:
    from assistant.model_router import ModelUsageBudget

    store = UserSettingsStore(tmp_path / "user_settings.json")
    store.update(allow_paid_calls=True)
    router = ModelRouter(
        ModelRoutingConfig(mode="automatic", automatic=AutomaticRoutingConfig(claude_enabled=True, daily_budget_usd=0.0001)),
        env={}, anthropic_key_available=lambda: True, paid_calls_allowed=paid_calls_getter(store, env={}),
        budget=ModelUsageBudget(tmp_path / "usage.json"),
    )

    decision = router.decide(ModelRoutingInput(source="RESPONSE_COMPOSER", user_message="Hoje estou cansado.", privacy_class="PERSONAL"))

    assert (decision.provider, decision.reason_code) == ("ollama", "daily_budget_exceeded")


def test_runtime_reports_and_changes_the_setting(tmp_path) -> None:
    store = UserSettingsStore(tmp_path / "user_settings.json")
    runtime = ModelRuntimeBridge(router=_router(paid_calls_getter(store, env={})), budget=None, store=store, env={})

    assert runtime.current_payload()["paid_calls_enabled"] is False
    payload = runtime.execute_command({"intent": "set_allow_paid_calls", "parameters": {"enabled": True}})
    assert payload["paid_calls_enabled"] is True and store.load()["allow_paid_calls"] is True
    assert runtime.execute_command({"intent": "set_allow_paid_calls", "parameters": {"enabled": False}})["paid_calls_enabled"] is False

    forced = ModelRuntimeBridge(router=None, budget=None, store=store, env={PAID_CALL_CONFIRMATION_ENV: "true"})
    assert forced.current_payload()["paid_calls_enabled"] is True


def test_anthropic_provider_asks_the_setting_on_every_call(monkeypatch) -> None:
    from assistant import anthropic_provider

    state = {"allowed": False, "posts": 0}

    class _Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}}

    def post(*args, **kwargs):
        state["posts"] += 1
        return _Response()

    monkeypatch.setattr(anthropic_provider.requests, "post", post)
    monkeypatch.delenv(PAID_CALL_CONFIRMATION_ENV, raising=False)
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="test-key", allow_paid_calls_getter=lambda: state["allowed"])

    with pytest.raises(ProviderConfigurationError) as blocked:
        provider.chat([{"role": "user", "content": "hi"}])
    assert blocked.value.provider_error_type == "paid_calls_not_confirmed" and state["posts"] == 0

    state["allowed"] = True
    assert provider.chat([{"role": "user", "content": "hi"}]).text == "ok"
    assert state["posts"] == 1
