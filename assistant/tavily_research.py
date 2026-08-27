from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import requests

from assistant.live_web_research import (
    LiveWebResearchProvider,
    _claim_from_result,
    _clean_text,
    _canonical_key,
    _domain,
    _is_http_url,
    _sanitize_evidence,
    _source_quality,
)
from assistant.secret_storage import sanitize_secret_error


ApiKeyGetter = Callable[[], tuple[str, str]]


@dataclass(frozen=True)
class TavilyResearchConfig:
    timeout_seconds: float = 8.0
    max_results: int = 5
    search_depth: str = "basic"
    include_answer: bool = False
    include_raw_content: bool = False


class TavilyResearchProvider:
    """Tavily adapter for the existing ResearchProvider contract."""

    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        *,
        api_key_getter: ApiKeyGetter,
        session: Any | None = None,
        config: TavilyResearchConfig | None = None,
    ) -> None:
        self.api_key_getter = api_key_getter
        self.session = session or requests
        self.config = config or TavilyResearchConfig()

    def __repr__(self) -> str:
        return "TavilyResearchProvider(<redacted>)"

    def run(self, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        results, status, reason = self.search_with_status(query)
        return json.dumps(
            {
                "provider": "tavily",
                "provider_status": status,
                "fallback_reason": reason,
                "results": results,
            },
            ensure_ascii=False,
        )

    def search_with_status(self, query: str) -> tuple[list[dict[str, Any]], str, str]:
        if not query:
            return [], "failed", "empty_query"
        api_key, key_source = self.api_key_getter()
        if not api_key:
            return [], "not_configured", "not_configured"

        payload = {
            "query": query,
            "topic": "general",
            "search_depth": self.config.search_depth,
            "max_results": max(0, min(20, self.config.max_results)),
            "include_answer": self.config.include_answer,
            "include_raw_content": self.config.include_raw_content,
            "include_images": False,
        }
        try:
            response = self.session.post(
                self.endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "EchoResearch/0.1",
                },
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout:
            return [], "failed", "timeout"
        except Exception as exc:
            return [], "failed", _safe_failure_reason(exc)

        results = self._normalize_results(query, data, key_source=key_source)
        if not results:
            return [], "no_results", "no_results"
        return results, "completed", ""

    def _normalize_results(self, query: str, payload: dict[str, Any], *, key_source: str) -> list[dict[str, Any]]:
        raw_results = payload.get("results", []) if isinstance(payload, dict) else []
        if not isinstance(raw_results, list):
            return []
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")
        for raw in raw_results:
            if len(normalized) >= self.config.max_results:
                break
            if not isinstance(raw, dict):
                continue
            title = _clean_text(str(raw.get("title") or "Resultado Tavily"))
            url = str(raw.get("url") or "").strip()
            content = _clean_text(str(raw.get("content") or raw.get("raw_content") or ""))
            if not title or not content or not _is_http_url(url):
                continue
            key = _canonical_key(url, title)
            if key in seen:
                continue
            seen.add(key)
            evidence = _sanitize_evidence(content)
            score = _float_or_none(raw.get("score"))
            result = {
                "title": title,
                "url": url,
                "domain": _domain(url),
                "source_quality": _source_quality(url, query),
                "snippet": evidence,
                "retrieved_at": retrieved_at,
                "provider": "tavily",
                "provider_key_source": key_source,
                "score": score,
                "claims": [_claim_from_result(query, title, url, evidence)],
            }
            normalized.append(result)
        return normalized


class LiveResearchProviderChain:
    """Prefer Tavily, then fall back to the existing free provider."""

    def __init__(
        self,
        *,
        tavily_provider: TavilyResearchProvider,
        fallback_provider: LiveWebResearchProvider | None = None,
    ) -> None:
        self.tavily_provider = tavily_provider
        self.fallback_provider = fallback_provider or LiveWebResearchProvider()

    def run(self, arguments: dict[str, Any]) -> str:
        tavily_payload = json.loads(self.tavily_provider.run(arguments))
        if tavily_payload.get("results"):
            return json.dumps(tavily_payload, ensure_ascii=False)

        query = str(arguments.get("query") or "").strip()
        fallback_results = self.fallback_provider.search(query)
        for result in fallback_results:
            result["provider"] = result.get("provider") or "bing_rss"
            result["fallback_from"] = "tavily"
            result["fallback_reason"] = str(tavily_payload.get("fallback_reason") or "no_results")
        return json.dumps(
            {
                "provider": "bing_rss" if fallback_results else "none",
                "provider_status": "completed" if fallback_results else "failed",
                "fallback_from": "tavily",
                "fallback_reason": str(tavily_payload.get("fallback_reason") or "no_results"),
                "results": fallback_results,
            },
            ensure_ascii=False,
        )


def _safe_failure_reason(exc: Exception) -> str:
    text = sanitize_secret_error(exc).lower()
    if "timeout" in text:
        return "timeout"
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return "auth_error"
    if "429" in text or "rate" in text:
        return "rate_limited"
    if "connection" in text or "network" in text or "dns" in text:
        return "network_error"
    return "provider_error"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
