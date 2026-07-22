from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


ALLOWED_ACTIONS = (
    "listar ficheiros",
    "ler ficheiro",
    "resumir ficheiro",
    "criar ficheiro",
    "abrir aplicacao",
    "abrir ficheiro",
    "abrir pasta",
    "abrir URL",
    "abrir projeto",
)

BLOCKED_ACTIONS = (
    "apagar ficheiro",
    "mover ficheiro",
    "executar comandos",
    "aceder fora da workspace",
)

# Placeholder for future actions that may ask the user before proceeding.
SENSITIVE_ACTIONS_REQUIRING_CONFIRMATION: tuple[str, ...] = ()


@dataclass(frozen=True)
class SecurityDecision:
    allowed: bool
    action: str | None = None
    message: str | None = None
    requires_confirmation: bool = False


def check_user_request(message: str) -> SecurityDecision:
    """Apply the safety policy before any local file action runs."""

    text = _normalize_text(message)

    blocked_action = _detect_blocked_action(text, message)
    if blocked_action is not None:
        return SecurityDecision(
            allowed=False,
            action=blocked_action,
            message=(
                f"Não posso realizar esta ação: {blocked_action}. "
                "Nesta versão não apago, movo, executo comandos nem acedo fora da workspace."
            ),
        )

    allowed_action = _detect_allowed_action(text)
    requires_confirmation = allowed_action in SENSITIVE_ACTIONS_REQUIRING_CONFIRMATION
    return SecurityDecision(
        allowed=True,
        action=allowed_action,
        requires_confirmation=requires_confirmation,
    )


def _detect_blocked_action(text: str, original_message: str) -> str | None:
    if _mentions_path_outside_workspace(text, original_message):
        return "aceder fora da workspace"

    if any(word in text for word in ("apaga", "apagar", "remove", "remover", "elimina", "eliminar", "delete")):
        return "apagar ficheiro"

    if any(word in text for word in ("move", "mover", "renomeia", "renomear", "copia para", "copiar para")):
        return "mover ficheiro"

    command_words = (
        "executa",
        "executar",
        "corre comando",
        "correr comando",
        "cmd",
        "powershell",
        "terminal",
        "shell",
        "subprocess",
    )
    if any(word in text for word in command_words):
        return "executar comandos"

    return None


def _detect_allowed_action(text: str) -> str | None:
    if any(word in text for word in ("cria ficheiro", "cria um ficheiro", "criar ficheiro", "criar um ficheiro", "novo ficheiro")):
        return "criar ficheiro"

    if any(phrase in text for phrase in ("abre o projeto", "abrir projeto", "abre projeto")):
        return "abrir projeto"

    if any(phrase in text for phrase in ("abre http://", "abre https://", "abrir http://", "abrir https://")):
        return "abrir URL"

    if any(phrase in text for phrase in ("abre a pasta", "abre pasta", "abrir pasta")):
        return "abrir pasta"

    if any(phrase in text for phrase in ("abre o ficheiro", "abre ficheiro", "abrir ficheiro")):
        return "abrir ficheiro"

    if any(word in text for word in ("abre", "abrir")):
        return "abrir aplicacao"

    if any(word in text for word in ("resume", "resumir", "sumariza", "sumarizar", "resumo do", "resumo de")):
        return "resumir ficheiro"

    if any(word in text for word in ("le ", "ler ", "abre ", "abrir ", "conteudo", "conteudo do", "conteudo de")):
        return "ler ficheiro"

    if any(word in text for word in ("lista", "listar", "mostra", "mostrar")) and any(
        word in text for word in ("ficheiro", "ficheiros", "documento", "documentos", "pasta", "workspace")
    ):
        return "listar ficheiros"

    return None


def _mentions_path_outside_workspace(text: str, original_message: str) -> bool:
    if re.search(r"(^|[\\/])\.\.([\\/]|$)", original_message):
        return True

    windows_drive = re.search(r"\b[a-zA-Z]:[\\/]", original_message)
    if windows_drive is not None:
        return True

    outside_words = ("fora da workspace", "fora do workspace", "fora da pasta workspace")
    return any(word in text for word in outside_words)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
