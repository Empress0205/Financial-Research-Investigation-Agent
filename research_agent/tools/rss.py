"""The read_rss_feed tool: the latest items from a news or press-release feed.

Search engines rank by relevance; RSS feeds list items by date, straight from
the publisher. That makes RSS the right tool for "what has the ECB announced
this week?", and the wrong one for "what is Tanzania's inflation rate?".

The catalogue below is a curated list of feeds that were checked to work on
2026-11-05. Naming them in the tool description is how the model learns what
it can read. The model may also pass any other feed URL.
"""
from __future__ import annotations

import calendar
import time
from datetime import datetime, timezone
from typing import Any

import feedparser
from bs4 import BeautifulSoup

from ..config import Settings
from .base import Tool, ToolResult, make_evidence
from .http import FetchError, safe_get

FEED_CATALOGUE: dict[str, dict[str, str]] = {
    "allafrica_tanzania": {"url": "https://allafrica.com/tools/headlines/rdf/tanzania/headlines.rdf",
                           "about": "Tanzania news headlines (AllAfrica aggregator)"},
    "allafrica_kenya": {"url": "https://allafrica.com/tools/headlines/rdf/kenya/headlines.rdf",
                        "about": "Kenya news headlines (AllAfrica aggregator)"},
    "allafrica_business": {"url": "https://allafrica.com/tools/headlines/rdf/business/headlines.rdf",
                           "about": "African business news (AllAfrica aggregator)"},
    "dailynews_tz": {"url": "https://dailynews.co.tz/feed/",
                     "about": "Daily News, Tanzanian state-owned newspaper (all topics)"},
    "bis_central_bank_speeches": {"url": "https://www.bis.org/doclist/cbspeeches.rss",
                                  "about": "Speeches by central bankers worldwide (Bank for International Settlements)"},
    "ecb_press": {"url": "https://www.ecb.europa.eu/rss/press.html",
                  "about": "European Central Bank press releases and speeches"},
    "fed_press": {"url": "https://www.federalreserve.gov/feeds/press_all.xml",
                  "about": "US Federal Reserve press releases"},
    "bbc_business": {"url": "https://feeds.bbci.co.uk/news/business/rss.xml",
                     "about": "BBC business news (global, UK-leaning)"},
}
MAX_ITEMS_CAP = 20
SUMMARY_CHARS = 300


def _entry_date(entry: Any) -> tuple[str | None, str | None]:
    for key, label in (("published_parsed", "feed published"), ("updated_parsed", "feed updated")):
        parsed: time.struct_time | None = entry.get(key)
        if parsed:
            stamp = datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
            return stamp.isoformat(), label
    return None, None


def _strip_html(text: str) -> str:
    return " ".join(BeautifulSoup(text or "", "html.parser").get_text(" ").split())


def _keyword_list(keywords: Any) -> list[str]:
    if isinstance(keywords, str):
        keywords = keywords.split(",")
    if not isinstance(keywords, list):
        return []
    return [str(k).strip().lower() for k in keywords if str(k).strip()]


def parse_feed(body: bytes, *, feed_name: str, keywords: list[str], max_items: int) -> dict[str, Any]:
    parsed = feedparser.parse(body)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"Not a valid RSS/Atom feed ({type(parsed.bozo_exception).__name__}).")

    items = []
    for entry in parsed.entries:
        link = (entry.get("link") or "").strip()
        if not link.startswith(("http://", "https://")):
            continue
        title = _strip_html(entry.get("title", "")) or "(untitled)"
        summary = _strip_html(entry.get("summary", ""))[:SUMMARY_CHARS]
        if keywords and not any(k in f"{title} {summary}".lower() for k in keywords):
            continue
        published, date_source = _entry_date(entry)
        items.append({"title": title, "url": link, "summary": summary,
                      "published_date": published, "date_source": date_source})

    # Newest first; items without a date go last.
    items.sort(key=lambda i: i["published_date"] or "", reverse=True)
    return {"feed": feed_name, "feed_title": parsed.feed.get("title", ""),
            "total_entries": len(parsed.entries), "matched": len(items), "items": items[:max_items]}


def make_rss_tool(settings: Settings) -> Tool:
    def read_rss_feed(feed: str, keywords: Any = None, max_items: int | str = 10) -> ToolResult:
        if not isinstance(feed, str) or not feed.strip():
            return ToolResult(ok=False, error="feed must be a catalogue name or an http(s) URL.")
        feed = feed.strip()
        if feed in FEED_CATALOGUE:
            name, url = feed, FEED_CATALOGUE[feed]["url"]
        elif feed.startswith(("http://", "https://")):
            name, url = "custom", feed
        else:
            return ToolResult(ok=False, error=f"Unknown feed '{feed}'. Use one of: "
                                               f"{', '.join(FEED_CATALOGUE)}, or a full feed URL.")
        try:
            limit = max(1, min(int(max_items), MAX_ITEMS_CAP))
        except (TypeError, ValueError):
            limit = 10
        words = _keyword_list(keywords)

        try:
            response = safe_get(url, timeout=settings.http_timeout_seconds, max_bytes=settings.fetch_max_bytes)
            data = parse_feed(response.body, feed_name=name, keywords=words, max_items=limit)
        except (FetchError, ValueError) as exc:
            return ToolResult(ok=False, error=str(exc))

        if not data["items"]:
            data["note"] = (f"No items matched keywords {words}." if words and data["total_entries"]
                            else "The feed has no items.")
        evidence = [make_evidence(url=i["url"], title=i["title"], published_date=i["published_date"],
                                  snippet=i["summary"], source_tool="read_rss_feed",
                                  date_source=i["date_source"], feed=name)
                    for i in data["items"]]
        summary = f"{len(data['items'])} of {data['total_entries']} items from {name}"
        if words:
            summary += f" matching {words}"
        return ToolResult(ok=True, data=data, evidence=evidence, summary=summary)

    catalogue_text = "; ".join(f"'{k}' = {v['about']}" for k, v in FEED_CATALOGUE.items())
    return Tool(
        name="read_rss_feed",
        description=(
            "Read the most recent items (newest first) from a news or press-release RSS feed. "
            "Use it for the LATEST announcements and headlines, not for background facts or statistics. "
            "Returns headline, short summary, url and published date per item; use fetch_page to read "
            f"an item in full. Available feeds: {catalogue_text}. Any other RSS/Atom URL also works."
        ),
        parameters={
            "type": "object",
            "properties": {
                "feed": {"type": "string", "description": "A feed name from the catalogue, or a full feed URL."},
                "keywords": {"type": "array", "items": {"type": "string"},
                             "description": "Optional. Keep only items whose title or summary contains "
                                            "any of these words (case-insensitive)."},
                "max_items": {"type": "integer", "description": f"Maximum items to return (1-{MAX_ITEMS_CAP})."},
            },
            "required": ["feed"],
        },
        func=read_rss_feed,
    )
