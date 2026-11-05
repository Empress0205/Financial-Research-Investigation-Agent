"""The web_search tool: Tavily first, DuckDuckGo as a fallback.

Every hit is normalised into the same shape (`SearchHit`) whichever engine
produced it, so the agent never needs to know which backend answered.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import requests

from ..config import Settings
from .base import Tool, ToolResult, make_evidence

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
MAX_QUERY_CHARS = 400
MAX_SNIPPET_CHARS = 600


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    published_date: str | None  # None = the source gave none. We never guess a date.
    engine: str


def _normalise(raw_hits: Any, *, engine: str, url_key: str, snippet_key: str,
               date_key: str) -> list[SearchHit]:
    """Turn engine-specific dicts into SearchHits, dropping anything malformed."""
    if not isinstance(raw_hits, list):
        return []
    hits: list[SearchHit] = []
    for raw in raw_hits:
        if not isinstance(raw, dict):
            continue
        url = str(raw.get(url_key) or "").strip()
        if not url.startswith(("http://", "https://")):
            continue  # a result without a real URL can't be cited, so it is useless as evidence
        date = raw.get(date_key)
        hits.append(SearchHit(
            title=str(raw.get("title") or "(untitled)").strip(),
            url=url,
            snippet=str(raw.get(snippet_key) or "").strip()[:MAX_SNIPPET_CHARS],
            published_date=str(date).strip() if date else None,
            engine=engine,
        ))
    return hits


def _tavily_search(query: str, *, api_key: str, topic: str, max_results: int,
                   timeout: float) -> list[SearchHit]:
    response = requests.post(
        TAVILY_URL,
        json={"query": query, "topic": topic, "max_results": max_results, "search_depth": "basic"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("Malformed Tavily response: missing 'results' list")
    return _normalise(results, engine="tavily", url_key="url", snippet_key="content",
                      date_key="published_date")


def _ddg_search(query: str, *, topic: str, max_results: int) -> list[SearchHit]:
    from ddgs import DDGS  # optional dependency; the caller handles ImportError

    ddgs = DDGS()
    if topic == "news":
        raw = ddgs.news(query, max_results=max_results)
        return _normalise(raw, engine="duckduckgo", url_key="url", snippet_key="body", date_key="date")
    raw = ddgs.text(query, max_results=max_results)
    return _normalise(raw, engine="duckduckgo", url_key="href", snippet_key="body", date_key="date")


def _to_evidence(hits: list[SearchHit]) -> list[dict[str, Any]]:
    return [make_evidence(url=h.url, title=h.title, published_date=h.published_date, snippet=h.snippet,
                          source_tool="web_search", date_source="search engine" if h.published_date else None)
            for h in hits]


def make_web_search_tool(settings: Settings) -> Tool:
    """Build the tool. A factory lets the function 'remember' settings via closure."""

    def web_search(query: str, topic: str = "general") -> ToolResult:
        query = (query or "").strip() if isinstance(query, str) else ""
        if not query:
            return ToolResult(ok=False, error="Query must be a non-empty string.")
        query = query[:MAX_QUERY_CHARS]
        if topic not in ("general", "news"):
            topic = "general"

        errors: list[str] = []

        # 1) Primary engine
        if settings.tavily_api_key:
            try:
                hits = _tavily_search(query, api_key=settings.tavily_api_key, topic=topic,
                                      max_results=settings.search_max_results,
                                      timeout=settings.http_timeout_seconds)
                if hits:
                    logger.info("Tavily returned %d result(s)", len(hits))
                    return ToolResult(ok=True, data={"engine": "tavily",
                                                     "results": [asdict(h) for h in hits]},
                                      evidence=_to_evidence(hits), summary=f"{len(hits)} results (tavily)")
                errors.append("Tavily returned no usable results")
            except Exception as exc:
                errors.append(f"Tavily failed: {type(exc).__name__}: {exc}")
        else:
            errors.append("TAVILY_API_KEY not set")

        # 2) Fallback engine
        logger.warning("Falling back to DuckDuckGo (%s)", "; ".join(errors))
        try:
            hits = _ddg_search(query, topic=topic, max_results=settings.search_max_results)
            if hits:
                logger.info("DuckDuckGo returned %d result(s)", len(hits))
                return ToolResult(ok=True, data={"engine": "duckduckgo", "note": "; ".join(errors),
                                                 "results": [asdict(h) for h in hits]},
                                  evidence=_to_evidence(hits), summary=f"{len(hits)} results (duckduckgo)")
            errors.append("DuckDuckGo returned no usable results")
        except ImportError:
            errors.append("DuckDuckGo fallback unavailable (pip install ddgs)")
        except Exception as exc:
            errors.append(f"DuckDuckGo failed: {type(exc).__name__}: {exc}")

        # 3) Both failed: report it as data, don't raise
        return ToolResult(ok=False, error="; ".join(errors))

    return Tool(
        name="web_search",
        description=(
            "Search the web. Use it to DISCOVER sources: returns title, url, a short snippet and "
            "published_date (null if unknown) per result. Snippets are short and can be misleading; "
            "read important sources in full with fetch_page before quoting figures. "
            "Use topic='news' for recent developments, 'general' for background and official pages."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query."},
                "topic": {"type": "string", "enum": ["general", "news"],
                          "description": "'news' for recent events, 'general' otherwise."},
            },
            "required": ["query"],
        },
        func=web_search,
    )
