from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus, urlparse


DEFAULT_QUICK_SITES = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "outlook": "https://outlook.office.com",
    "chatgpt": "https://chatgpt.com",
    "github": "https://github.com",
}
BLOCKED_SCHEMES = {"file", "javascript", "data"}
BLOCKED_TARGET_WORDS = {"powershell", "cmd", "terminal"}
BLOCKED_EXECUTABLE_SUFFIXES = {".exe", ".bat", ".ps1", ".cmd"}
QUICK_SITES_PATH = Path(__file__).resolve().parents[1] / "config" / "quick_sites.json"
SEARCH_ENGINES = {
    "google": ("Google", "https://www.google.com/search?q={query}"),
    "youtube": ("YouTube", "https://www.youtube.com/results?search_query={query}"),
}
SAFE_URL_REFUSAL = (
    "Não posso abrir esse destino por segurança. "
    "Só aceito URLs http/https seguros ou atalhos configurados."
)


@dataclass(frozen=True)
class FastRoute:
    kind: str
    response: str | None = None
    tool_name: str | None = None
    arguments: dict[str, str] | None = None
    reason: str = "Comando simples resolvido pelo router rapido."


def route_fast_command(message: str, quick_sites: dict[str, str] | None = None) -> FastRoute | None:
    text = _normalize_text(message).strip(" .,!?:;")

    if text in {"limpar conversa", "limpa conversa", "limpa a conversa", "apaga conversa"}:
        return FastRoute(kind="clear_conversation")

    if text in {"testar microfone", "testa microfone", "testa o microfone", "testar o microfone"}:
        return FastRoute(kind="test_microphone")

    search_route = _search_route(message)
    if search_route is not None:
        return search_route

    open_target = _extract_open_target(message)
    if open_target is not None and _is_dangerous_open_target(open_target):
        return FastRoute(kind="denied", response=SAFE_URL_REFUSAL)

    url = _url_from_open_target(open_target, quick_sites=quick_sites)
    if url == "":
        return FastRoute(kind="denied", response=SAFE_URL_REFUSAL)
    if url:
        arguments = {"url": url}
        display_name = _display_name_for_open_target(open_target, url)
        if display_name:
            arguments["display_name"] = display_name
        return FastRoute(
            kind="tool",
            tool_name="open_url",
            arguments=arguments,
            reason="O utilizador pediu para abrir um URL simples.",
        )

    return None


def _search_route(message: str) -> FastRoute | None:
    match = re.match(
        r"^\s*(?:pesquisa|pesquisar|procura|procurar)\s+no\s+(google|youtube)(?:\s+por)?\s+(.+?)\s*$",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    engine = _normalize_text(match.group(1))
    query = match.group(2).strip()
    if not query:
        return None

    engine_info = SEARCH_ENGINES.get(engine)
    if engine_info is None:
        return None

    engine_label, url_template = engine_info
    encoded_query = quote_plus(query)
    url = url_template.format(query=encoded_query)
    return FastRoute(
        kind="tool",
        tool_name="open_url",
        arguments={
            "url": url,
            "search_engine": engine_label,
            "search_query": query,
        },
        reason=f"O utilizador pediu uma pesquisa rapida no {engine_label}.",
    )


def _extract_open_target(message: str) -> str | None:
    text = _normalize_text(message).strip()
    if not re.match(r"^(abre|abrir|abre-me|abre me)\b", text):
        return None

    raw_target = re.sub(r"^(abre|abrir|abre-me|abre me)\s+", "", message.strip(), flags=re.IGNORECASE)
    raw_target = re.sub(r"^(o|a|os|as)\s+", "", raw_target.strip(), flags=re.IGNORECASE)
    return raw_target.strip().strip("\"'").rstrip(".,;")


def load_quick_sites(config_path: Path | None = None) -> dict[str, str]:
    path = config_path or QUICK_SITES_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_QUICK_SITES)

    if isinstance(raw, dict) and isinstance(raw.get("quick_sites"), dict):
        raw_sites = raw["quick_sites"]
    elif isinstance(raw, dict):
        raw_sites = raw
    else:
        return dict(DEFAULT_QUICK_SITES)

    sites: dict[str, str] = {}
    for name, url in raw_sites.items():
        if isinstance(name, str) and isinstance(url, str):
            normalized_name = _normalize_text(name).strip(" .,!?:;")
            if normalized_name:
                sites[normalized_name] = url.strip()
    return sites or dict(DEFAULT_QUICK_SITES)


def _url_from_open_target(target: str | None, quick_sites: dict[str, str] | None = None) -> str | None:
    if target is None:
        return None
    normalized_target = _normalize_text(target)
    sites = _normalized_quick_sites(quick_sites)

    if normalized_target in sites:
        configured_url = sites[normalized_target]
        if not _is_http_url(configured_url):
            return ""
        return configured_url.strip()

    if _is_http_url(target):
        return target.strip()

    if _looks_like_bare_domain(target):
        return _normalize_url(target)

    return None


def _display_name_for_open_target(target: str | None, url: str) -> str:
    normalized = _normalize_text(target or "").strip(" .,!?:;")
    if normalized in {"gmail", "mail", "email", "correio"} or "mail.google.com" in url:
        return "Gmail"
    return ""


def _normalized_quick_sites(quick_sites: dict[str, str] | None = None) -> dict[str, str]:
    sites = quick_sites if quick_sites is not None else load_quick_sites()
    normalized: dict[str, str] = {}
    for name, url in sites.items():
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        key = _normalize_text(name).strip(" .,!?:;")
        if key:
            normalized[key] = url.strip()
    return normalized


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
