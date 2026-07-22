from __future__ import annotations

import pytest

from assistant.anthropic_provider import AnthropicProvider
from assistant.model_provider import DEFAULT_OLLAMA_MODEL, resolve_model_provider


def test_cli_provider_and_model_win_over_everything() -> None:
    resolved = resolve_model_provider(
        cli_provider="anthropic",
        cli_model="cli-model",
        env={"ECHO_MODEL_PROVIDER": "ollama", "ECHO_MODEL_NAME": "env-model", "OLLAMA_MODEL": "legacy-model"},
        settings={"model": {"provider": "ollama", "name": "settings-model"}},
    )

    assert resolved.provider == "anthropic"
    assert resolved.provider_source == "cli"
    assert resolved.model == "cli-model"
    assert resolved.model_source == "cli"


def test_echo_model_provider_and_name_win_over_settings() -> None:
    resolved = resolve_model_provider(
        env={"ECHO_MODEL_PROVIDER": "anthropic", "ECHO_MODEL_NAME": "env-model"},
        settings={"model": {"provider": "ollama", "name": "settings-model"}},
    )

    assert resolved.provider == "anthropic"
    assert resolved.provider_source == "ECHO_MODEL_PROVIDER"
    assert resolved.model == "env-model"
    assert resolved.model_source == "ECHO_MODEL_NAME"


def test_ollama_model_legacy_env_still_works() -> None:
    resolved = resolve_model_provider(
        env={"OLLAMA_MODEL": "legacy-model"},
        settings={"model": {"provider": "ollama", "name": "settings-model"}},
    )

    assert resolved.provider == "ollama"
    assert resolved.model == "legacy-model"
    assert resolved.model_source == "OLLAMA_MODEL"


def test_new_settings_model_is_used_without_env_or_cli() -> None:
    resolved = resolve_model_provider(
        env={},
        settings={"model": {"provider": "ollama", "name": "settings-model"}},
    )

    assert resolved.provider == "ollama"
    assert resolved.provider_source == "settings.json"
    assert resolved.model == "settings-model"
    assert resolved.model_source == "settings.json"


def test_legacy_ollama_settings_remain_supported() -> None:
    resolved = resolve_model_provider(env={}, settings={"ollama": {"model": "legacy-settings-model"}})

    assert resolved.provider == "ollama"
    assert resolved.model == "legacy-settings-model"
    assert resolved.model_source == "settings.json:ollama"


def test_default_is_ollama_llama_without_configuration() -> None:
    resolved = resolve_model_provider(env={}, settings={})

    assert resolved.provider == "ollama"
    assert resolved.provider_source == "default"
    assert resolved.model == DEFAULT_OLLAMA_MODEL
    assert resolved.model_source == "default"


def test_empty_strings_are_ignored() -> None:
    resolved = resolve_model_provider(
        cli_provider=" ",
        cli_model=" ",
        env={"ECHO_MODEL_PROVIDER": "", "ECHO_MODEL_NAME": "", "OLLAMA_MODEL": "  "},
        settings={"model": {"provider": "ollama", "name": "settings-model"}},
    )

    assert resolved.provider == "ollama"
    assert resolved.model == "settings-model"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Provider de modelo desconhecido"):
        resolve_model_provider(env={"ECHO_MODEL_PROVIDER": "paid-magic"}, settings={})


def test_anthropic_does_not_fallback_to_ollama_when_key_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="", allow_paid_calls=True)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        provider.chat([{"role": "user", "content": "Ola"}])
