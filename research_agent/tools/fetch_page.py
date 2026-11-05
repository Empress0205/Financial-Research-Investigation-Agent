"""The fetch_page tool: read the full text of one web page or PDF.

Why this exists: a search snippet is ~300 characters chosen by a search
engine. In Milestone 1 the model misread a headline because a snippet was all
it had. fetch_page lets it read the actual source (a central bank statement,
a statistics release) before stating a figure.

Two things matter beyond "download and strip HTML":
  * Publication date. We look for it in the page's metadata and record *where*
    we found it (date_source), so later milestones can judge how reliable it is.
  * Size. Every character returned goes into the LLM's context and is re-sent
    on every later iteration, so the text is cut to `max_chars`.
"""
from __future__ import annotations

import io
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from ..config import Settings
from .base import Tool, ToolResult, make_evidence
from .http import FetchError, safe_get

logger = logging.getLogger(__name__)

HTML_TYPES = {"text/html", "application/xhtml+xml"}
MAX_PDF_PAGES = 15
NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe"]

# <meta> tags that carry a publication date, most reliable first.
DATE_META = [
    ("property", "article:published_time"),
    ("name", "article:published_time"),
    ("itemprop", "datePublished"),
    ("name", "citation_publication_date"),
    ("name", "dc.date.issued"),
    ("name", "dc.date"),
    ("name", "date"),
    ("name", "pubdate"),
    ("name", "publish-date"),
    ("property", "og:published_time"),
]
JSONLD_DATE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')


def _find_html_date(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    """Return (date, where_it_came_from). Must run before <script> tags are removed."""
    for attr, value in DATE_META:
        tag = soup.find("meta", attrs={attr: re.compile(f"^{re.escape(value)}$", re.I)})
        if tag and tag.get("content"):
            return tag["content"].strip(), f"meta {value}"
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        match = JSONLD_DATE.search(script.string or "")
        if match:
            return match.group(1).strip(), "json-ld datePublished"
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        return time_tag["datetime"].strip(), "<time> element (may not be the publication date)"
    return None, None


def _clean_text(text: str) -> str:
    lines = (re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def extract_html(body: bytes) -> dict[str, Any]:
    soup = BeautifulSoup(body, "html.parser")
    published_date, date_source = _find_html_date(soup)
    title = soup.title.get_text(strip=True) if soup.title else ""

    for tag in soup(NOISE_TAGS):
        tag.decompose()
    # Prefer the main article body if the page marks one; fall back to the whole page.
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = _clean_text(main.get_text("\n"))
    if len(text) < 300 and main is not soup:  # the "article" was a teaser box, not the body
        text = _clean_text(soup.get_text("\n"))
    return {"title": title, "text": text, "published_date": published_date, "date_source": date_source}


def extract_pdf(body: bytes, max_chars: int) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(body))
    parts, length = [], 0
    # Stop reading once we have enough text: big PDFs are slow to parse.
    for page in reader.pages[:MAX_PDF_PAGES]:
        page_text = page.extract_text() or ""
        parts.append(page_text)
        length += len(page_text)
        if length >= max_chars:
            break
    title, published_date, date_source = "", None, None
    try:
        meta = reader.metadata
        if meta:
            title = (meta.title or "").strip()
            if meta.creation_date:
                published_date = meta.creation_date.date().isoformat()
                date_source = "pdf creation date (file date, may differ from publication date)"
    except Exception:  # broken metadata shouldn't lose the text we already have
        logger.debug("Could not read PDF metadata", exc_info=True)
    return {"title": title, "text": _clean_text("\n".join(parts)), "published_date": published_date,
            "date_source": date_source, "pages": len(reader.pages), "pages_read": len(parts)}


def make_fetch_page_tool(settings: Settings) -> Tool:
    def fetch_page(url: str, max_chars: int | str | None = None) -> ToolResult:
        if not isinstance(url, str) or not url.strip():
            return ToolResult(ok=False, error="url must be a non-empty string.")
        try:
            limit = int(max_chars) if max_chars is not None else settings.page_max_chars
        except (TypeError, ValueError):
            limit = settings.page_max_chars
        limit = max(500, min(limit, settings.page_max_chars_cap))

        try:
            response = safe_get(url.strip(), timeout=settings.http_timeout_seconds,
                                max_bytes=settings.fetch_max_bytes)
        except FetchError as exc:
            return ToolResult(ok=False, error=str(exc))

        is_pdf = response.content_type == "application/pdf" or (
            response.content_type in ("", "application/octet-stream") and response.url.lower().endswith(".pdf"))
        try:
            if is_pdf:
                page = extract_pdf(response.body, limit)
            elif response.content_type in HTML_TYPES or response.content_type == "":
                page = extract_html(response.body)
            elif response.content_type.startswith("text/"):
                page = {"title": "", "text": _clean_text(response.body.decode("utf-8", "replace")),
                        "published_date": None, "date_source": None}
            else:
                return ToolResult(ok=False, error=f"Unsupported content type '{response.content_type}'.")
        except Exception as exc:  # corrupt PDF, undecodable HTML, ...
            return ToolResult(ok=False, error=f"Could not extract text: {type(exc).__name__}: {exc}")

        text = page["text"]
        if not text:
            return ToolResult(ok=False, error="The page has no readable text (it may need JavaScript, "
                                               "or be a scanned PDF).")

        total_chars = len(text)
        content = text[:limit]
        title = page["title"] or urlparse(response.url).netloc
        data = {
            "url": response.url,
            "title": title,
            "published_date": page["published_date"],
            "date_source": page["date_source"],
            "content": content,
            "truncated": total_chars > limit,
            "total_chars": total_chars,
        }
        summary = f"{len(content):,} of {total_chars:,} chars from {urlparse(response.url).netloc}"
        if "pages" in page:
            # For PDFs we stop reading early, so total_chars only covers the pages read.
            data["pdf_pages"] = page["pages"]
            data["pdf_pages_read"] = page["pages_read"]
            data["truncated"] = data["truncated"] or page["pages_read"] < page["pages"]
            summary = (f"{len(content):,} chars from PDF pages 1-{page['pages_read']} of "
                       f"{page['pages']}, {urlparse(response.url).netloc}")
        evidence = [make_evidence(url=response.url, title=title, published_date=page["published_date"],
                                  snippet=content[:300], source_tool="fetch_page",
                                  date_source=page["date_source"], requested_url=url.strip())]
        return ToolResult(ok=True, data=data, evidence=evidence, summary=summary)

    return Tool(
        name="fetch_page",
        description=(
            "Download one web page or PDF and return its main text, title and publication date "
            "(with where the date was found). Use it to READ a source in full before quoting a figure "
            "from it, especially official sources such as central banks or statistics offices. "
            "Snippets from web_search are too short to rely on for exact numbers. "
            "Text is truncated to max_chars."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full http(s) URL to read."},
                "max_chars": {"type": "integer",
                              "description": f"Maximum characters of text to return "
                                             f"(default {settings.page_max_chars}, "
                                             f"max {settings.page_max_chars_cap})."},
            },
            "required": ["url"],
        },
        func=fetch_page,
    )
