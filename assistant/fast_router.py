from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlparse


URL_ALIASES = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "outlook": "https://outlook.office.com",
}
BLOCKED_SCHEMES = {"file", "javascript", "data"}
BLOCKED_TARGET_WORDS = {"powershell", "cmd", "terminal"}
BLOCKED_EXECUTABLE_SUFFIXES = {".exe", ".bat", ".ps1", ".cmd"}
SAFE_URL_REFUSAL = (
    "Nao posso abrir esse destino por seguranca. "
    "So aceito URLs http/https seguros ou atalhos conhecidos como youtube e google."
)


@dataclass(frozen=True)
class FastRoute:
    kind: str
    response: str | None = None
    tool_name: str | None = None
    arguments: dict[str, str] | None = None
    reason: str = "Comando simples resolvido pelo router rapido."


def route_fast_command(message: str) -> FastRoute | None:
    text = _normalize_text(message).strip(" .,!?:;")

    if text in {"limpar conversa", "limpa conversa", "limpa a conversa", "apaga conversa"}:
        return FastRoute(kind="clear_conversation")

    if text in {"testar microfone", "testa microfone", "testa o microfone", "testar o microfone"}:
        return FastRoute(kind="test_microphone")

    open_target = _extract_open_target(message)
    if open_target is not None and _is_dangerous_open_target(open_target):
        return FastRoute(kind="denied", response=SAFE_URL_REFUSAL)

    url = _url_from_open_target(open_target)
    if url:
        return FastRoute(
            kind="tool",
            tool_name="open_url",
            arguments={"url": url},
            reason="O utilizador pediu para abrir um URL simples.",
        )

    return None


def _extract_open_target(message: str) -> str | None:
    text = _normalize_text(message).strip()
    if not re.match(r"^(abre|abrir|abre-me|abre me)\b", text):
        return None

    raw_target = re.sub(r"^(abre|abrir|abre-me|abre me)\s+", "", message.strip(), flags=re.IGNORECASE)
    raw_target = re.sub(r"^(o|a|os|as)\s+", "", raw_target.strip(), flags=re.IGNORECASE)
    return raw_target.strip().strip("\"'").rstrip(".,;")


def _url_from_open_target(target: str | None) -> str | None:
    if target is None:
        return None
    normalized_target = _normalize_text(target)

    if normalized_target in URL_ALIASES:
        return URL_ALIASES[normalized_target]

    if _is_http_url(target):
        return target.strip()

    if _looks_like_bare_domain(target):
        return _normalize_url(target)

    return None


def _is_dangerous_open_target(target: str) -> bool:
    value = target.strip()
    normalized = _normalize_text(value).strip(" .,!?:;")
    parsed = urlparse(value)

    if parsed.scheme.lower() in BLOCKED_SCHEMES:
        return True

    if normalized in BLOCKED_TARGET_WORDS:
        return True

    if re.match(r"^[a-zA-Z]:[\\/]", value):
        return True

    if value.startswith(("./", ".\\", "../", "..\\")):
        return True

    lower_value = value.lower()
    lower_path = parsed.path.lower() if parsed.scheme else lower_value
    if any(lower_path.endswith(suffix) for suffix in BLOCKED_EXECUTABLE_SUFFIXES):
        return True

    return False


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    return _valid_host(parsed.netloc)


def _looks_like_bare_domain(value: str) -> bool:
    if "://" in value:
        return False
    parsed = urlparse(f"https://{value}")
    return _valid_host(parsed.netloc)


def _valid_host(host: str) -> bool:
    if not host:
        return False
    if any(char.isspace() for char in host):
        return False
    return "." in host


def _normalize_url(value: str) -> str:
    cleaned = value.strip()
    if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        cleaned = f"https://{cleaned}"
    return cleaned


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
