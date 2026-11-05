"""Agent loop tests with a scripted fake LLM and fake tools (no network, no keys)."""
from datetime import date

from research_agent.agent import ResearchAgent, find_unverified_urls, merge_evidence
from research_agent.config import Limits
from research_agent.llm import LLMError, LLMResponse, ToolCall
from research_agent.tools import Tool, ToolRegistry, ToolResult, make_evidence

SOURCE = make_evidence(url="https://www.bot.go.tz/mps-2026", title="BoT Monetary Policy Statement",
                       published_date=None, snippet="Headline inflation was 3.2% in September 2026.",
                       source_tool="web_search")
QUERY_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
URL_SCHEMA = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}


class FakeLLM:
    """Returns pre-written responses in order and records what it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name="web_search", args='{"query": "Tanzania inflation"}', call_id="c1"):
    return ToolCall(call_id, name, args)


def tool_turn(*calls, tokens=100):
    return LLMResponse(content=None, tool_calls=list(calls), total_tokens=tokens)


def final(text):
    return LLMResponse(content=text, total_tokens=200)


def fake_search(results=(SOURCE,)):
    return Tool(name="web_search", description="fake", parameters=QUERY_SCHEMA,
                func=lambda query: ToolResult(ok=True, data={"results": list(results)},
                                              evidence=list(results), summary="fake results"))


def fake_fetch(published_date="2026-10-02"):
    def fetch(url):
        ev = make_evidence(url=url, title="BoT MPS (full)", published_date=published_date,
                           snippet="full text", source_tool="fetch_page", date_source="meta date")
        return ToolResult(ok=True, data={"url": url, "content": "full text"}, evidence=[ev])
    return Tool(name="fetch_page", description="fake", parameters=URL_SCHEMA, func=fetch)


def make_agent(llm, limits=Limits(), tools=None):
    return ResearchAgent(llm=llm, tools=ToolRegistry(tools or [fake_search()]), limits=limits)


# ---- the basic loop -------------------------------------------------------------

def test_search_then_answer():
    answer = f"## Answer\nInflation was 3.2% ({SOURCE['url']}).\n## Sources\n## Research Limitations\nOne source."
    llm = FakeLLM([tool_turn(call()), final(answer)])
    result = make_agent(llm).run("Tanzania inflation?")

    assert result.stopped_reason == "final_answer"
    assert result.iterations == 2 and result.tool_calls == 1
    assert result.total_tokens == 300
    assert [e["url"] for e in result.evidence] == [SOURCE["url"]]
    assert result.unverified_urls == []
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "c1"
    assert "3.2%" in tool_msgs[0]["content"]


def test_all_tool_schemas_are_offered():
    llm = FakeLLM([final("answer")])
    make_agent(llm, tools=[fake_search(), fake_fetch()]).run("q")
    offered = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert offered == {"web_search", "fetch_page"}


def test_system_prompt_contains_todays_date():
    llm = FakeLLM([final("answer")])
    make_agent(llm).run("q")
    assert date.today().isoformat() in llm.calls[0]["messages"][0]["content"]


# ---- multiple tools: selection, usage and evidence ---------------------------------------

def test_search_then_fetch_merges_evidence_and_counts_usage():
    llm = FakeLLM([
        tool_turn(call()),
        tool_turn(call("fetch_page", f'{{"url": "{SOURCE["url"]}"}}', "c2")),
        final(f"Inflation was 3.2% ({SOURCE['url']}, 2026-10-02)."),
    ])
    result = make_agent(llm, tools=[fake_search(), fake_fetch()]).run("q")

    assert result.tool_usage == {"web_search": 1, "fetch_page": 1}
    assert len(result.evidence) == 1  # same URL from two tools = one source
    record = result.evidence[0]
    assert record["seen_by"] == ["web_search", "fetch_page"]
    assert record["published_date"] == "2026-10-02"  # filled in by fetch_page
    assert record["title"] == "BoT MPS (full)"


def test_parallel_tool_calls_in_one_turn():
    llm = FakeLLM([
        tool_turn(call(call_id="a"), call("fetch_page", '{"url": "https://x.org/p"}', "b")),
        final("done"),
    ])
    result = make_agent(llm, tools=[fake_search(), fake_fetch()]).run("q")
    assert result.tool_usage == {"web_search": 1, "fetch_page": 1}
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]


def test_unknown_tool_is_reported_back_with_available_names():
    llm = FakeLLM([tool_turn(call("stock_trader", "{}")), final("ok")])
    result = make_agent(llm, tools=[fake_search(), fake_fetch()]).run("q")
    assert result.stopped_reason == "final_answer"
    assert "Unknown tool" in result.errors[0]
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "web_search" in tool_msg["content"] and "fetch_page" in tool_msg["content"]


def test_fetched_url_after_redirect_counts_as_verified():
    evidence = [{"url": "https://final.org/page", "requested_url": "https://short.ly/abc"}]
    assert find_unverified_urls("see https://short.ly/abc", evidence) == []


# ---- budgets and failures --------------------------------------------------------

def test_invented_url_is_flagged():
    answer = f"See {SOURCE['url']} and https://made-up-source.com/report."
    result = make_agent(FakeLLM([tool_turn(call()), final(answer)])).run("q")
    assert result.unverified_urls == ["https://made-up-source.com/report"]


def test_tool_budget_forces_an_answer():
    llm = FakeLLM([tool_turn(call(call_id="c1")), tool_turn(call(call_id="c2")), final("done")])
    result = make_agent(llm, Limits(max_iterations=5, max_tool_calls=2)).run("q")
    assert result.tool_calls == 2
    assert result.stopped_reason == "final_answer"
    assert llm.calls[2]["tool_choice"] == "none"
    assert llm.calls[2]["tools"] is None


def test_extra_calls_beyond_budget_are_answered_but_not_run():
    llm = FakeLLM([tool_turn(call(call_id="a"), call(call_id="b")), final("done")])
    result = make_agent(llm, Limits(max_iterations=3, max_tool_calls=1)).run("q")
    assert result.tool_calls == 1
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]
    assert "budget exhausted" in tool_msgs[1]["content"]


def test_last_iteration_switches_tools_off():
    llm = FakeLLM([tool_turn(call()), final("answer")])
    make_agent(llm, Limits(max_iterations=2, max_tool_calls=10)).run("q")
    assert llm.calls[0]["tool_choice"] == "auto" and llm.calls[0]["tools"]
    assert llm.calls[1]["tool_choice"] == "none"
    # Regression: gpt-oss ignored tool_choice="none" when schemas were still sent.
    assert llm.calls[1]["tools"] is None


def test_model_ignoring_tool_choice_hits_iteration_limit():
    llm = FakeLLM([tool_turn(call(call_id="c1")), tool_turn(call(call_id="c2"))])
    result = make_agent(llm, Limits(max_iterations=2, max_tool_calls=10)).run("q")
    assert result.stopped_reason == "max_iterations"
    assert result.answer is None
    assert result.evidence


def test_token_budget_stops_the_run():
    result = make_agent(FakeLLM([tool_turn(call(), tokens=5_000)]),
                        Limits(max_iterations=5, max_total_tokens=1_000)).run("q")
    assert result.stopped_reason == "token_budget"


def test_llm_error_does_not_crash():
    result = make_agent(FakeLLM([LLMError("rate limited")])).run("q")
    assert result.stopped_reason == "llm_error"
    assert "rate limited" in result.errors[0]


def test_failed_tool_is_survivable():
    failing = Tool(name="web_search", description="fake", parameters=QUERY_SCHEMA,
                   func=lambda query: ToolResult(ok=False, error="all engines down"))
    llm = FakeLLM([tool_turn(call()), final("I could not retrieve evidence.")])
    result = make_agent(llm, tools=[failing]).run("q")
    assert result.stopped_reason == "final_answer"
    assert result.evidence == []
    assert result.errors == ["web_search: all engines down"]


def test_empty_response():
    result = make_agent(FakeLLM([LLMResponse(content="  ")])).run("q")
    assert result.stopped_reason == "empty_response"


# ---- helpers ---------------------------------------------------------------------

def test_find_unverified_urls_handles_punctuation_and_slashes():
    evidence = [{"url": "https://a.com/x/"}]
    assert find_unverified_urls("Source: https://a.com/x.", evidence) == []
    assert find_unverified_urls("(https://b.com/y)", evidence) == ["https://b.com/y"]


def test_merge_evidence_ignores_malformed_items():
    store = {}
    merge_evidence(store, [{"title": "no url"}, "not a dict", {"url": "https://a.com", "source_tool": "t"}])
    assert list(store) == ["https://a.com"]
