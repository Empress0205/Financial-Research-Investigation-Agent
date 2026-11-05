"""read_rss_feed tests. safe_get is replaced, so no network is used."""
import json

from research_agent.config import Settings
from research_agent.tools import FEED_CATALOGUE, make_rss_tool
from research_agent.tools import rss
from research_agent.tools.http import FetchedResponse, FetchError

FEED = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Central Bank News</title>
<item><title>Older: MPC keeps rate</title><link>https://cb.example/a</link>
  <description>&lt;p&gt;The rate was &lt;b&gt;held&lt;/b&gt;.&lt;/p&gt;</description>
  <pubDate>Mon, 06 Oct 2026 09:00:00 GMT</pubDate></item>
<item><title>Newer: inflation eases</title><link>https://cb.example/b</link>
  <description>Inflation fell to 3.1 percent.</description>
  <pubDate>Tue, 04 Nov 2026 09:00:00 GMT</pubDate></item>
<item><title>No date: tourism</title><link>https://cb.example/c</link></item>
<item><title>Bad link</title><link>javascript:alert(1)</link></item>
</channel></rss>"""


def run(monkeypatch, args, body=FEED):
    seen = {}
    def fake_get(url, **kw):
        seen["url"] = url
        return FetchedResponse(url=url, status=200, content_type="application/rss+xml", body=body, headers={})
    monkeypatch.setattr(rss, "safe_get", fake_get)
    result = make_rss_tool(Settings()).run(json.dumps(args))
    return result, seen


def test_catalogue_name_resolves_to_its_url(monkeypatch):
    _, seen = run(monkeypatch, {"feed": "ecb_press"})
    assert seen["url"] == FEED_CATALOGUE["ecb_press"]["url"]


def test_items_sorted_newest_first_with_dates(monkeypatch):
    result, _ = run(monkeypatch, {"feed": "ecb_press"})
    items = result.data["items"]
    assert [i["url"] for i in items] == ["https://cb.example/b", "https://cb.example/a", "https://cb.example/c"]
    assert items[0]["published_date"].startswith("2026-11-04")
    assert items[2]["published_date"] is None  # never guessed
    assert result.data["total_entries"] == 4  # the javascript: link was dropped from items


def test_html_is_stripped_from_summaries(monkeypatch):
    result, _ = run(monkeypatch, {"feed": "ecb_press"})
    older = next(i for i in result.data["items"] if i["url"].endswith("/a"))
    assert older["summary"] == "The rate was held ."


def test_keyword_filter_accepts_list_or_string(monkeypatch):
    for keywords in (["INFLATION"], "inflation, zzz"):
        result, _ = run(monkeypatch, {"feed": "ecb_press", "keywords": keywords})
        assert [i["url"] for i in result.data["items"]] == ["https://cb.example/b"]


def test_no_keyword_match_gives_a_note(monkeypatch):
    result, _ = run(monkeypatch, {"feed": "ecb_press", "keywords": ["bitcoin"]})
    assert result.ok and result.data["items"] == [] and "No items matched" in result.data["note"]


def test_max_items(monkeypatch):
    result, _ = run(monkeypatch, {"feed": "ecb_press", "max_items": 1})
    assert len(result.data["items"]) == 1


def test_evidence_per_item(monkeypatch):
    result, _ = run(monkeypatch, {"feed": "ecb_press"})
    assert len(result.evidence) == 3
    assert all(e["source_tool"] == "read_rss_feed" for e in result.evidence)


def test_custom_url_allowed(monkeypatch):
    _, seen = run(monkeypatch, {"feed": "https://other.example/feed.xml"})
    assert seen["url"] == "https://other.example/feed.xml"


def test_unknown_feed_name_lists_catalogue(monkeypatch):
    result, _ = run(monkeypatch, {"feed": "imf_news"})
    assert not result.ok and "allafrica_tanzania" in result.error


def test_malformed_feed_is_an_error(monkeypatch):
    result, _ = run(monkeypatch, {"feed": "ecb_press"}, body=b"<html>this is not a feed")
    assert not result.ok and "Not a valid RSS" in result.error


def test_fetch_failure_is_an_error(monkeypatch):
    def refuse(url, **kw):
        raise FetchError("HTTP 404 from x")
    monkeypatch.setattr(rss, "safe_get", refuse)
    result = make_rss_tool(Settings()).run('{"feed": "ecb_press"}')
    assert not result.ok and "404" in result.error
