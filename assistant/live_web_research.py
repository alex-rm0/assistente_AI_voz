from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests


_DANGEROUS_TEXT_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "api key",
    "password",
    "execute command",
    "run command",
    "powershell",
    "cmd.exe",
)


@dataclass(frozen=True)
class LiveWebResearchConfig:
    timeout_seconds: float = 8.0
    max_results: int = 5
    max_fetches: int = 0
    user_agent: str = "EchoResearch/0.1"


class LiveWebResearchProvider:
    """
    Minimal live-web adapter for the Task Research layer.

    It uses DuckDuckGo's public Instant Answer endpoint as a no-key search
    source, then returns the same JSON shape expected by assistant.task_research.
    Page fetching is deliberately bounded and treated only as evidence text.
    """

    def __init__(self, session: Any | None = None, config: LiveWebResearchConfig | None = None) -> None:
        self.session = session or requests
        self.config = config or LiveWebResearchConfig()

    def run(self, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        payload = {"results": self.search(query)}
        return json.dumps(payload, ensure_ascii=False)

    def search(self, query: str) -> list[dict[str, Any]]:
        if not query:
            return []

        try:
            items = self._search_bing_rss(query)
        except Exception:
            items = []

        if not items:
            items = self._search_duckduckgo_instant_answer(query)
        if not items:
            return []

        return self._normalize_items(query, items)

    def _search_bing_rss(self, query: str) -> list[dict[str, str]]:
        response = self.session.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
            headers={
                "Accept": "application/rss+xml, application/xml, text/xml",
                "User-Agent": self.config.user_agent,
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return self._items_from_bing_rss(getattr(response, "text", "") or "")

    def _search_duckduckgo_instant_answer(self, query: str) -> list[dict[str, str]]:
        try:
            response = self.session.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_redirect": "1",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
                headers={"User-Agent": self.config.user_agent},
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []
        items = self._items_from_duckduckgo(payload)
        return items

    def _normalize_items(self, query: str, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        fetched = 0
        for item in items:
            if len(results) >= self.config.max_results:
                break
            title = _clean_text(str(item.get("title") or item.get("heading") or "Resultado web"))
            url = str(item.get("url") or "").strip()
            snippet = _clean_text(str(item.get("snippet") or ""))
            if not title or not _is_http_url(url):
                continue
            key = _canonical_key(url, title)
            if key in seen:
                continue
            seen.add(key)

            evidence = snippet
            if fetched < self.config.max_fetches:
                fetched_text = self._fetch_evidence(url)
                if fetched_text:
                    evidence = fetched_text
                    fetched += 1

            evidence = _sanitize_evidence(evidence or snippet)
            result = {
                "title": title,
                "url": url,
                "domain": _domain(url),
                "source_quality": _source_quality(url, query),
                "snippet": evidence,
                "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "claims": [_claim_from_result(query, title, url, evidence)],
            }
            results.append(result)
        return results

    def _fetch_evidence(self, url: str) -> str:
        if not _is_http_url(url):
            return ""
        try:
            response = self.session.get(
                url,
                headers={"User-Agent": self.config.user_agent},
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except Exception:
            return ""

        content_type = str(getattr(response, "headers", {}).get("content-type", "")).lower()
        if content_type and not any(kind in content_type for kind in ("text/html", "text/plain")):
            return ""
        text = getattr(response, "text", "") or ""
        return _html_to_text(text)[:900]

    def _items_from_duckduckgo(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        abstract = _clean_text(str(payload.get("AbstractText") or payload.get("Abstract") or ""))
        abstract_url = str(payload.get("AbstractURL") or "").strip()
        heading = _clean_text(str(payload.get("Heading") or ""))
        if abstract and abstract_url:
            items.append({"title": heading or "Resultado web", "url": abstract_url, "snippet": abstract})

        for raw in payload.get("Results") or []:
            self._append_duckduckgo_item(items, raw)
        for raw in payload.get("RelatedTopics") or []:
            self._append_duckduckgo_item(items, raw)
        return items

    def _append_duckduckgo_item(self, items: list[dict[str, str]], raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        if isinstance(raw.get("Topics"), list):
            for child in raw["Topics"]:
                self._append_duckduckgo_item(items, child)
            return
        text = _clean_text(str(raw.get("Text") or raw.get("Result") or ""))
        url = str(raw.get("FirstURL") or "").strip()
        if text and url:
            title = text.split(" - ", 1)[0].split(".", 1)[0][:120].strip() or "Resultado web"
            items.append({"title": title, "url": url, "snippet": text})

    def _items_from_bing_rss(self, xml_text: str) -> list[dict[str, str]]:
        if not xml_text.strip():
            return []
        root = ET.fromstring(xml_text)
        items: list[dict[str, str]] = []
        for item in root.findall(".//item"):
            title = _clean_text(item.findtext("title") or "")
            url = _clean_text(item.findtext("link") or "")
            snippet = _clean_text(item.findtext("description") or "")
            if title and url:
                items.append({"title": title, "url": url, "snippet": snippet})
        return items


def _claim_from_result(query: str, title: str, url: str, evidence: str) -> dict[str, Any]:
    return {
        "subject": _subject_from_query(query, title),
        "predicate": _predicate_from_query(query),
        "value": _short_value(evidence or title),
        "kind": "claim",
        "confidence": _confidence_for_source(url),
        "epistemic_type": "web_snippet",
        "evidence": evidence or title,
    }


def _subject_from_query(query: str, title: str = "") -> str:
    text = query.strip()
    lowered_query = text.lower()
    if "destinos" in lowered_query or "destination" in lowered_query:
        candidate = re.split(r"\s[-|:]\s|,|\(", title, maxsplit=1)[0].strip()
        if candidate:
            return candidate[:80]
    markers = (
        " camera ",
        " review ",
        " battery ",
        " customer support ",
        " current price",
        " activities",
        " opening_hours",
        " transport",
        " cost",
        " destination",
        " reputation",
    )
    lowered = f" {text.lower()} "
    cut = len(text)
    for marker in markers:
        idx = lowered.find(marker)
        if idx >= 0:
            cut = min(cut, max(0, idx))
    subject = text[:cut].strip(" -,:")
    return subject or text[:80] or "pesquisa"


def _predicate_from_query(query: str) -> str:
    text = query.lower()
    if any(token in text for token in ("camera", "câmara", "camara", "photo", "video")):
        return "camera_quality"
    if any(token in text for token in ("battery", "bateria", "autonomia", "charging")):
        return "battery"
    if any(token in text for token in ("support", "suporte", "reputation", "reviews")):
        return "support_quality"
    if any(token in text for token in ("price", "preço", "preco", "cost", "custo")):
        return "price"
    if any(token in text for token in ("activities", "things", "visitar", "fazer")):
        return "activities"
    if "transport" in text:
        return "transport"
    return "web_evidence"


def _source_quality(url: str, query: str) -> str:
    domain = _domain(url)
    if domain.endswith(".gov") or ".gov." in domain or "visit" in domain or "tourism" in domain:
        return "official"
    if any(site in domain for site in ("dxomark", "rtings", "notebookcheck", "gsmarena", "pcmag", "trustedreviews")):
        return "independent_review"
    if any(site in domain for site in ("reddit", "forum", "quora")):
        return "community"
    if "wikipedia.org" in domain:
        return "reference"
    if _subject_from_query(query).lower().replace(" ", "") in domain.replace("-", ""):
        return "official_or_related"
    return "web"


def _confidence_for_source(url: str) -> float:
    quality = _source_quality(url, "")
    if quality in {"official", "independent_review", "reference"}:
        return 0.72
    if quality == "community":
        return 0.55
    return 0.62


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _canonical_key(url: str, title: str) -> str:
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}".rstrip("/")
    return normalized or title.lower()


def _html_to_text(value: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header).*?</\1>", " ", value)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return _clean_text(html.unescape(text))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _sanitize_evidence(value: str) -> str:
    text = _clean_text(value)
    lowered = text.lower()
    for pattern in _DANGEROUS_TEXT_PATTERNS:
        if pattern in lowered:
            text = re.sub(re.escape(pattern), "[texto removido]", text, flags=re.IGNORECASE)
            lowered = text.lower()
    return text[:900].strip()


def _short_value(value: str) -> str:
    text = _clean_text(value)
    if len(text) <= 180:
        return text
    return text[:177].rsplit(" ", 1)[0].strip() + "..."
