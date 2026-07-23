from __future__ import annotations

import types

import pytest

from assistant import secret_storage
from assistant.secret_storage import (
    ANTHROPIC_SECRET_NAME,
    InMemorySecretStorage,
    NullSecretStorage,
    WindowsCredentialManagerSecretStorage,
    resolve_anthropic_api_key,
    sanitize_secret_error,
)


def test_null_secret_storage_is_unavailable_and_has_no_file_fallback() -> None:
    storage = NullSecretStorage()

    assert storage.is_available() is False
    assert storage.has_secret(ANTHROPIC_SECRET_NAME) is False
    assert storage.get_secret(ANTHROPIC_SECRET_NAME) is None
    with pytest.raises(RuntimeError):
        storage.set_secret(ANTHROPIC_SECRET_NAME, "secret")


def test_in_memory_secret_storage_supports_interface_without_exposing_repr() -> None:
    storage = InMemorySecretStorage()

    storage.set_secret(ANTHROPIC_SECRET_NAME, "secret-value")

    assert storage.is_available() is True
    assert storage.has_secret(ANTHROPIC_SECRET_NAME) is True
    assert storage.get_secret(ANTHROPIC_SECRET_NAME) == "secret-value"
    assert "secret-value" not in repr(storage)
    storage.delete_secret(ANTHROPIC_SECRET_NAME)
    assert storage.has_secret(ANTHROPIC_SECRET_NAME) is False


def test_resolve_anthropic_api_key_prefers_environment_over_secure_storage() -> None:
    storage = InMemorySecretStorage()
    storage.set_secret(ANTHROPIC_SECRET_NAME, "stored-secret")

    value, source = resolve_anthropic_api_key({ANTHROPIC_SECRET_NAME: "env-secret"}, storage)

    assert value == "env-secret"
    assert source == "environment"


def test_resolve_anthropic_api_key_uses_secure_storage_when_env_missing() -> None:
    storage = InMemorySecretStorage()
    storage.set_secret(ANTHROPIC_SECRET_NAME, "stored-secret")

    value, source = resolve_anthropic_api_key({}, storage)

    assert value == "stored-secret"
    assert source == "secure_storage"


def test_resolve_anthropic_api_key_returns_none_source_without_secret() -> None:
    value, source = resolve_anthropic_api_key({}, InMemorySecretStorage())

    assert value == ""
    assert source == "none"


def test_sanitize_secret_error_redacts_obvious_secret_markers() -> None:
    assert sanitize_secret_error("failed with sk-ant-secret-value") == "failed with <redacted>"
    assert sanitize_secret_error("ANTHROPIC_API_KEY=secret") == "<redacted>"


def test_windows_storage_rejects_non_windows_backend(monkeypatch) -> None:
    monkeypatch.setattr(secret_storage.os, "name", "posix")

    storage = WindowsCredentialManagerSecretStorage()

    assert storage.is_available() is False
    assert storage.unavailable_reason == "windows_credential_manager_required"


def test_windows_storage_rejects_insecure_keyring_backend(monkeypatch) -> None:
    class PlainBackend:
        pass

    fake_keyring = types.SimpleNamespace(get_keyring=lambda: PlainBackend())
    monkeypatch.setattr(secret_storage.os, "name", "nt")
    monkeypatch.setitem(__import__("sys").modules, "keyring", fake_keyring)

    storage = WindowsCredentialManagerSecretStorage()

    assert storage.is_available() is False
    assert storage.unavailable_reason == "insecure_keyring_backend"
