"""Context-management tests: estimating, compacting, and recovering from 'too large'."""
import json

from research_agent.agent import ResearchAgent, compact_context, estimate_tokens
from research_agent.config import Limits
from research_agent.llm import LLMError, LLMResponse, ToolCall
from research_agent.tools import Tool, ToolRegistry, ToolResult, make_evidence

URL_SCHEMA = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}


def big_page_tool(chars=8_000):
    def fetch(url):
        ev = make_evidence(url=url, title=f"Page {url}", published_date="2026-10-01",
                           snippet="Inflation 4.3% in September 2026.", source_tool="fetch_page")
        return ToolResult(ok=True, data={"url": url, "content": "x" * chars}, evidence=[ev])
    return Tool(name="fetch_page", description="fake", parameters=URL_SCHEMA, func=fetch)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sizes = []  # estimated size of each request we received

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.sizes.append(estimate_tokens(messages, tools))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fetch_turn(n):
    return LLMResponse(content=None, total_tokens=10,
                       tool_calls=[ToolCall(f"c{n}", "fetch_page", json.dumps({"url": f"https://s{n}.org"}))])


def agent(llm, context_limit):
    return ResearchAgent(llm=llm, tools=ToolRegistry([big_page_tool()]),
                         limits=Limits(max_iterations=6, max_tool_calls=6, max_context_tokens=context_limit))


def test_estimate_counts_messages_and_tools():
    msgs = [{"role": "user", "content": "x" * 400}]
    assert estimate_tokens(msgs, None) > 100
    assert estimate_tokens(msgs, [{"big": "y" * 4000}]) > estimate_tokens(msgs, None) + 900


def test_requests_stay_under_the_limit():
    llm = FakeLLM([fetch_turn(1), fetch_turn(2), fetch_turn(3), LLMResponse(content="done")])
    result = agent(llm, context_limit=3_000).run("q")
    assert result.stopped_reason == "final_answer"
    # Each page alone is ~2,000 tokens; without compaction request 4 would be ~6,000+.
    assert max(llm.sizes[1:]) <= 3_000
    assert len(result.evidence) == 3  # compaction never loses evidence


def test_oldest_results_are_compacted_first_and_latest_kept():
    llm = FakeLLM([fetch_turn(1), fetch_turn(2), LLMResponse(content="done")])
    ag = agent(llm, context_limit=2_800)
    captured = {}
    original = llm.chat
    def spy(messages, tools=None, tool_choice="auto"):
        captured["messages"] = [dict(m) for m in messages]
        return original(messages, tools, tool_choice)
    llm.chat = spy
    ag.run("q")
    tool_msgs = [m for m in captured["messages"] if m["role"] == "tool"]
    assert '"compacted": true' in tool_msgs[0]["content"]       # c1, old: compacted
    assert '"compacted"' not in tool_msgs[1]["content"]          # c2, latest: intact
    assert "4.3%" in tool_msgs[0]["content"]                     # the excerpt survives


def test_compacted_version_keeps_source_identity():
    result = big_page_tool().func("https://s1.org")
    compact = json.loads(result.to_compact_content())
    assert compact["sources"][0]["url"] == "https://s1.org"
    assert compact["sources"][0]["published_date"] == "2026-10-01"


def test_too_large_error_triggers_compaction_and_one_retry():
    llm = FakeLLM([fetch_turn(1), LLMError("413", kind="too_large"), LLMResponse(content="done")])
    result = agent(llm, context_limit=100_000).run("q")  # limit too high to compact pre-emptively
    assert result.stopped_reason == "final_answer"
    assert llm.sizes[2] < llm.sizes[1]  # the retry was smaller


def test_too_large_twice_stops_cleanly():
    llm = FakeLLM([fetch_turn(1), LLMError("413", kind="too_large"), LLMError("413 again", kind="too_large")])
    result = agent(llm, context_limit=100_000).run("q")
    assert result.stopped_reason == "llm_error"
    assert "413 again" in result.errors[-1]


def test_other_llm_errors_are_not_retried():
    llm = FakeLLM([fetch_turn(1), LLMError("401 bad key", kind="other")])
    result = agent(llm, context_limit=100_000).run("q")
    assert result.stopped_reason == "llm_error" and len(llm.sizes) == 2


def test_compact_context_returns_zero_when_nothing_to_compact():
    msgs = [{"role": "user", "content": "x" * 10_000}]
    assert compact_context(msgs, {}, None, limit=10) == 0
