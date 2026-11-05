"""Tool layer tests. No network: the search backends are replaced with fakes."""
import json

import pytest

from research_agent.config import Settings
from research_agent.tools import Tool, ToolResult, make_web_search_tool
from research_agent.tools import web_search as ws
from research_agent.tools.web_search import SearchHit


# ---- Tool.run: the safety boundary ------------------------------------------

def _echo_tool(func=None):
    return Tool(
        name="echo",
        description="test",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        func=func or (lambda text: ToolResult(ok=True, data=text)),
    )


def test_valid_call_runs():
    assert _echo_tool().run('{"text": "hi"}') == ToolResult(ok=True, data="hi")


def test_malformed_json_is_an_error_not_a_crash():
    result = _echo_tool().run('{"text": "hi"')
    assert not result.ok and "not valid JSON" in result.error


def test_missing_required_argument():
    result = _echo_tool().run("{}")
    assert not result.ok and "text" in result.error


def test_unknown_arguments_are_dropped():
    assert _echo_tool().run('{"text": "hi", "extra": 1}').ok


def test_tool_that_raises_is_contained():
    def boom(text):
        raise RuntimeError("kaboom")
    result = _echo_tool(boom).run('{"text": "x"}')
    assert not result.ok and "kaboom" in result.error


def test_tool_returning_wrong_type_is_flagged():
    result = _echo_tool(lambda text: "a plain string").run('{"text": "x"}')
    assert not result.ok and "malformed" in result.error


def test_schema_shape():
    schema = _echo_tool().schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"


# ---- web_search: primary, fallback, failure ------------------------------------

HIT = SearchHit(title="CPI report", url="https://example.org/cpi", snippet="Inflation 3.1%",
                published_date="2026-10-01", engine="tavily")


@pytest.fixture
def settings():
    return Settings(tavily_api_key="test-key")


def test_tavily_success(monkeypatch, settings):
    monkeypatch.setattr(ws, "_tavily_search", lambda q, **kw: [HIT])
    monkeypatch.setattr(ws, "_ddg_search", lambda q, **kw: pytest.fail("fallback should not run"))
    result = make_web_search_tool(settings).run(json.dumps({"query": "tanzania inflation"}))
    assert result.ok
    assert result.data["engine"] == "tavily"
    assert result.data["results"][0]["url"] == "https://example.org/cpi"


def test_falls_back_to_duckduckgo_when_tavily_fails(monkeypatch, settings):
    def broken(q, **kw):
        raise ConnectionError("tavily down")
    monkeypatch.setattr(ws, "_tavily_search", broken)
    monkeypatch.setattr(ws, "_ddg_search", lambda q, **kw: [HIT])
    result = make_web_search_tool(settings).run('{"query": "x"}')
    assert result.ok
    assert result.data["engine"] == "duckduckgo"
    assert "tavily down" in result.data["note"]


def test_falls_back_when_no_tavily_key(monkeypatch):
    monkeypatch.setattr(ws, "_ddg_search", lambda q, **kw: [HIT])
    result = make_web_search_tool(Settings(tavily_api_key=None)).run('{"query": "x"}')
    assert result.ok and result.data["engine"] == "duckduckgo"


def test_both_engines_fail_returns_error(monkeypatch, settings):
    def broken(q, **kw):
        raise TimeoutError("timed out")
    monkeypatch.setattr(ws, "_tavily_search", broken)
    monkeypatch.setattr(ws, "_ddg_search", broken)
    result = make_web_search_tool(settings).run('{"query": "x"}')
    assert not result.ok
    assert "Tavily failed" in result.error and "DuckDuckGo failed" in result.error


def test_empty_query_rejected(settings):
    result = make_web_search_tool(settings).run('{"query": "   "}')
    assert not result.ok


def test_normalise_drops_malformed_hits():
    raw = [
        {"title": "good", "url": "https://a.com", "content": "c", "published_date": None},
        {"title": "no url", "content": "c"},
        {"title": "bad url", "url": "ftp://x"},
        "not even a dict",
    ]
    hits = ws._normalise(raw, engine="tavily", url_key="url", snippet_key="content", date_key="published_date")
    assert [h.url for h in hits] == ["https://a.com"]
    assert hits[0].published_date is None


def test_malformed_tavily_payload_raises_and_is_caught(monkeypatch, settings):
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"unexpected": "shape"}
    monkeypatch.setattr(ws.requests, "post", lambda *a, **kw: FakeResponse())
    monkeypatch.setattr(ws, "_ddg_search", lambda q, **kw: [])
    result = make_web_search_tool(settings).run('{"query": "x"}')
    assert not result.ok and "Malformed Tavily response" in result.error
