from __future__ import annotations

import os
from typing import Protocol


ANTHROPIC_SECRET_NAME = "ANTHROPIC_API_KEY"
SECRET_SERVICE_NAME = "EchoOS"


class SecretStorage(Protocol):
    """Small secret storage interface.

    Implementations must never persist secrets to project files or expose
    secret values through repr/logging payloads.
    """

    def is_available(self) -> bool: ...

    def has_secret(self, name: str) -> bool: ...

    def set_secret(self, name: str, value: str) -> None: ...

    def get_secret(self, name: str) -> str | None: ...

    def delete_secret(self, name: str) -> None: ...


class NullSecretStorage:
    """Unavailable storage used as a safe default."""

    unavailable_reason = "secure_storage_unavailable"

    def is_available(self) -> bool:
        return False

    def has_secret(self, name: str) -> bool:
        return False

    def set_secret(self, name: str, value: str) -> None:
        raise RuntimeError("Secure secret storage is not available.")

    def get_secret(self, name: str) -> str | None:
        return None

    def delete_secret(self, name: str) -> None:
        return None


class InMemorySecretStorage:
    """Test-only secret storage. It never writes to disk."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self._values: dict[str, str] = {}

    def is_available(self) -> bool:
        return self.available

    def has_secret(self, name: str) -> bool:
        return self.available and bool(self._values.get(name))

    def set_secret(self, name: str, value: str) -> None:
        if not self.available:
            raise RuntimeError("Secure secret storage is not available.")
        self._values[name] = str(value or "")

    def get_secret(self, name: str) -> str | None:
        if not self.available:
            return None
        return self._values.get(name) or None

    def delete_secret(self, name: str) -> None:
        if self.available:
            self._values.pop(name, None)

    def __repr__(self) -> str:
        return "InMemorySecretStorage(<redacted>)"


class WindowsCredentialManagerSecretStorage:
    """Windows Credential Manager adapter through keyring.

    Availability is deliberately strict. If keyring is not installed, the OS is
    not Windows, or keyring resolves to a non-secure backend, the storage is
    treated as unavailable. There is no plaintext file fallback.
    """

    def __init__(self, *, service_name: str = SECRET_SERVICE_NAME) -> None:
        self.service_name = service_name
        self.unavailable_reason = ""
        self._keyring = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        self._available = self._resolve_keyring() is not None
        return self._available

    def has_secret(self, name: str) -> bool:
        value = self.get_secret(name)
        return bool(value)

    def set_secret(self, name: str, value: str) -> None:
        backend = self._require_keyring()
        backend.set_password(self.service_name, _safe_secret_name(name), str(value or ""))

    def get_secret(self, name: str) -> str | None:
        backend = self._resolve_keyring()
        if backend is None:
            return None
        try:
            return backend.get_password(self.service_name, _safe_secret_name(name)) or None
        except Exception:
            return None

    def delete_secret(self, name: str) -> None:
        backend = self._resolve_keyring()
        if backend is None:
            return
        try:
            backend.delete_password(self.service_name, _safe_secret_name(name))
        except Exception:
            return

    def _require_keyring(self):
        backend = self._resolve_keyring()
        if backend is None:
            raise RuntimeError("Secure secret storage is not available.")
        return backend

    def _resolve_keyring(self):
        if self._keyring is not None:
            return self._keyring
        if os.name != "nt":
            self.unavailable_reason = "windows_credential_manager_required"
            return None
        try:
            import keyring
        except Exception:
            self.unavailable_reason = "keyring_not_installed"
            return None
        try:
            backend = keyring.get_keyring()
        except Exception:
            self.unavailable_reason = "keyring_backend_unavailable"
            return None
        if not _looks_like_secure_windows_backend(backend):
            self.unavailable_reason = "insecure_keyring_backend"
            return None
        self._keyring = keyring
        return self._keyring


def resolve_anthropic_api_key(env: dict[str, str], storage: SecretStorage) -> tuple[str, str]:
    env_value = str(env.get(ANTHROPIC_SECRET_NAME) or "").strip()
    if env_value:
        return env_value, "environment"
    stored = storage.get_secret(ANTHROPIC_SECRET_NAME) if storage.is_available() else None
    if stored:
        return stored, "secure_storage"
    return "", "none"


def sanitize_secret_error(message: object) -> str:
    text = str(message or "")
    if len(text) > 300:
        text = text[:300] + "..."
    for marker in ("sk-ant-", "ANTHROPIC_API_KEY="):
        index = text.find(marker)
        if index >= 0:
            return text[:index] + "<redacted>"
    return text


def _safe_secret_name(name: str) -> str:
    clean = str(name or "").strip()
    return clean or ANTHROPIC_SECRET_NAME


def _looks_like_secure_windows_backend(backend: object) -> bool:
    backend_name = f"{backend.__class__.__module__}.{backend.__class__.__name__}".lower()
    if "windows" not in backend_name:
        return False
    insecure_markers = ("fail", "null", "plain", "file", "keyrings.alt", "chainer")
    return not any(marker in backend_name for marker in insecure_markers)
