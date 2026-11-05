"""fetch_page tests. safe_get is replaced, so no network is used."""
import json

from research_agent.config import Settings
from research_agent.tools import fetch_page as fp
from research_agent.tools import make_fetch_page_tool
from research_agent.tools.http import FetchedResponse, FetchError

ARTICLE = """<html><head>
<title>Monetary Policy Statement</title>
<meta property="article:published_time" content="2026-10-02T09:00:00Z">
<script>var tracking = "noise";</script><style>.x{}</style>
</head><body>
<nav>Home | About | Contact</nav>
<article><h1>MPC decision</h1><p>The Central Bank Rate was kept at 5.75 percent.</p>
<p>Headline inflation was 3.2 percent in September 2026.</p>""" + "<p>Detail paragraph.</p>" * 30 + """
</article><footer>Copyright</footer></body></html>"""


def response(body, content_type="text/html", url="https://bank.example/mps"):
    return FetchedResponse(url=url, status=200, content_type=content_type,
                           body=body.encode() if isinstance(body, str) else body, headers={})


def run(monkeypatch, body, content_type="text/html", args=None, url="https://bank.example/mps"):
    monkeypatch.setattr(fp, "safe_get", lambda u, **kw: response(body, content_type, url))
    tool = make_fetch_page_tool(Settings())
    return tool.run(json.dumps(args or {"url": "https://bank.example/mps"}))


def test_extracts_article_text_title_and_date(monkeypatch):
    result = run(monkeypatch, ARTICLE)
    assert result.ok
    data = result.data
    assert data["title"] == "Monetary Policy Statement"
    assert data["published_date"] == "2026-10-02T09:00:00Z"
    assert data["date_source"] == "meta article:published_time"
    assert "5.75 percent" in data["content"]
    for noise in ("tracking", "Home | About", "Copyright"):
        assert noise not in data["content"]


def test_evidence_record(monkeypatch):
    result = run(monkeypatch, ARTICLE)
    [ev] = result.evidence
    assert ev["source_tool"] == "fetch_page"
    assert ev["url"] == "https://bank.example/mps"
    assert ev["published_date"] == "2026-10-02T09:00:00Z"


def test_json_ld_date(monkeypatch):
    html = ('<html><head><script type="application/ld+json">{"datePublished": "2026-09-30"}</script>'
            '</head><body><p>' + "text " * 100 + "</p></body></html>")
    result = run(monkeypatch, html)
    assert result.data["published_date"] == "2026-09-30"
    assert result.data["date_source"] == "json-ld datePublished"


def test_missing_date_stays_none(monkeypatch):
    result = run(monkeypatch, "<html><body><p>" + "words " * 100 + "</p></body></html>")
    assert result.data["published_date"] is None


def test_truncation(monkeypatch):
    result = run(monkeypatch, ARTICLE, args={"url": "https://bank.example/mps", "max_chars": 500})
    assert len(result.data["content"]) == 500
    assert result.data["truncated"] is True
    assert result.data["total_chars"] > 500


def test_max_chars_is_clamped_and_tolerates_strings(monkeypatch):
    result = run(monkeypatch, ARTICLE, args={"url": "https://bank.example/mps", "max_chars": "999999"})
    assert len(result.data["content"]) <= Settings().page_max_chars_cap


def test_redirected_url_is_recorded(monkeypatch):
    result = run(monkeypatch, ARTICLE, url="https://bank.example/final")
    assert result.data["url"] == "https://bank.example/final"
    assert result.evidence[0]["requested_url"] == "https://bank.example/mps"


def test_pdf_is_routed_to_pdf_extractor(monkeypatch):
    monkeypatch.setattr(fp, "extract_pdf", lambda body, limit: {
        "title": "MPS PDF", "text": "Policy rate 5.75%", "published_date": "2026-10-01",
        "date_source": "pdf creation date", "pages": 12, "pages_read": 2})
    result = run(monkeypatch, b"%PDF-1.7 ...", content_type="application/pdf")
    assert result.ok and result.data["pdf_pages"] == 12
    assert "5.75%" in result.data["content"]
    # Only 2 of 12 pages were read, so the text is incomplete even though it is short.
    assert result.data["truncated"] is True
    assert "pages 1-2 of 12" in result.summary


def test_corrupt_pdf_is_an_error_not_a_crash(monkeypatch):
    result = run(monkeypatch, b"not really a pdf", content_type="application/pdf")
    assert not result.ok and "Could not extract text" in result.error


def test_unsupported_content_type(monkeypatch):
    result = run(monkeypatch, b"\x00\x01", content_type="image/png")
    assert not result.ok and "image/png" in result.error


def test_javascript_only_page(monkeypatch):
    result = run(monkeypatch, "<html><body><script>render()</script></body></html>")
    assert not result.ok and "no readable text" in result.error


def test_fetch_error_is_returned(monkeypatch):
    def refuse(url, **kw):
        raise FetchError("Refusing to fetch 'localhost'")
    monkeypatch.setattr(fp, "safe_get", refuse)
    result = make_fetch_page_tool(Settings()).run('{"url": "http://localhost"}')
    assert not result.ok and "localhost" in result.error
