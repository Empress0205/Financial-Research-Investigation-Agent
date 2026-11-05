"""The research agent (Milestone 2: several tools).

The loop is the same as in Milestone 1:

    1. Send the conversation + tool schemas to the LLM.
    2. If the LLM replies with tool calls, run them and append the results
       to the conversation, then go back to 1.
    3. If the LLM replies with plain text, that's the final answer.

What changed is *what sits around it*:
  * Tools come from a ToolRegistry, so the agent no longer knows or cares
    which tools exist. The LLM chooses between them using their descriptions.
  * Every tool reports its sources as `evidence` in one common shape, so
    citations can be checked whichever tool found the source.
  * We count how often each tool was chosen (`tool_usage`), to make the
    model's tool selection visible.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .config import Limits
from .llm import ChatModel, LLMError, ToolCall
from .tools.base import ToolResult
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial and economic research assistant.

How to use your tools:
- web_search: to DISCOVER sources. Snippets are short and can mislead; never quote an exact
  figure from a snippet alone if you can read the source.
- fetch_page: to READ a source in full. Before stating a key figure (inflation rate, policy rate,
  growth rate), read the page it comes from, preferably an official source (central bank,
  national statistics office, IMF, World Bank) rather than a site that repeats it.
- read_rss_feed: for the LATEST announcements and headlines from a publisher. Not for statistics.
Plan briefly before calling tools, and do not repeat a call that already returned the same thing.
If a tool fails, try another route (another query, another source) or report the gap.
Your working space is small: older tool results may be shortened. So whenever you call a tool,
first write one or two lines noting the exact figures, dates and URLs you have found so far
(your own messages are kept), and keep fetch_page max_chars modest.

Rules you must follow:
- State only facts that appear in tool results you received. Never invent statistics,
  sources, URLs or dates.
- Cite every fact with the exact URL it came from and its published_date. If published_date
  is null, write "date not stated".
- Say which figures you read in the full source and which come only from search snippets.
- If the evidence is thin, old, from a single source, or contradictory, say so plainly.
- You investigate and explain. You never give investment advice or make financial decisions.

Write the final answer in Markdown with exactly these sections:
## Answer
## Sources
(one bullet per source: title - URL - date - how you used it: read in full / snippet only)
## Research Limitations
(what you could not verify, how recent the evidence is, what is missing)
"""

BUDGET_EXHAUSTED_NOTE = (
    "The research budget is used up. Do not request more tools. Write the final answer now, "
    "using only the evidence already retrieved, in the required format."
)

URL_PATTERN = re.compile(r"https?://[^\s<>\"'\])]+")


@dataclass
class AgentRunResult:
    question: str
    answer: str | None
    stopped_reason: str  # final_answer | max_iterations | token_budget | llm_error | empty_response
    iterations: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    tool_usage: dict[str, int] = field(default_factory=dict)  # tool name -> times executed
    evidence: list[dict[str, Any]] = field(default_factory=list)
    unverified_urls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    """Rough token count of a request: about 4 characters per token for English JSON.

    It's an estimate (each model has its own tokenizer), so the limit leaves headroom.
    Tool schemas count too: they're sent with every request.
    """
    chars = len(json.dumps(messages, ensure_ascii=False))
    if tools:
        chars += len(json.dumps(tools, ensure_ascii=False))
    return chars // 4


def compact_context(messages: list[dict[str, Any]], compact_versions: dict[str, str],
                    tools: list[dict[str, Any]] | None, limit: int, *, force_all: bool = False) -> int:
    """Shrink old tool results until the request fits under `limit` tokens.

    Oldest results go first. Results from the latest round (after the last assistant
    message) are kept intact unless nothing else helps, because the model hasn't
    responded to them yet. Returns how many results were compacted.
    """
    last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    candidates = [i for i, m in enumerate(messages)
                  if m["role"] == "tool" and m.get("tool_call_id") in compact_versions
                  and m["content"] != compact_versions[m["tool_call_id"]]]
    older = [i for i in candidates if i < last_assistant]
    latest = [i for i in candidates if i > last_assistant]

    compacted = 0
    for i in older + latest:
        if not force_all and estimate_tokens(messages, tools) <= limit:
            break
        messages[i]["content"] = compact_versions[messages[i]["tool_call_id"]]
        compacted += 1
    return compacted


def normalise_url(url: str) -> str:
    return url.rstrip(".,;:").rstrip("/").lower()


def find_unverified_urls(answer: str, evidence: list[dict[str, Any]]) -> list[str]:
    """URLs that appear in the answer but were never returned by a tool."""
    known = set()
    for e in evidence:
        known.add(normalise_url(e["url"]))
        if e.get("requested_url"):  # fetch_page may have followed a redirect
            known.add(normalise_url(e["requested_url"]))
    unverified: list[str] = []
    for url in URL_PATTERN.findall(answer or ""):
        clean = url.rstrip(".,;:")
        if normalise_url(clean) not in known and clean not in unverified:
            unverified.append(clean)
    return unverified


def merge_evidence(store: dict[str, dict[str, Any]], items: list[dict[str, Any]]) -> None:
    """Add evidence records, keyed by URL.

    The same URL often arrives twice: first as a search snippet, then read in full
    with fetch_page. We keep one record and let later, non-empty fields fill in or
    replace earlier ones, so a date found by fetch_page fills a missing search date.
    `seen_by` records every tool that returned the URL.
    """
    for item in items:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        key = normalise_url(item["url"])
        existing = store.get(key)
        if existing is None:
            store[key] = {**item, "seen_by": [item.get("source_tool")]}
            continue
        for k, v in item.items():
            if v not in (None, "") and k != "source_tool":
                existing[k] = v
        if item.get("source_tool") not in existing["seen_by"]:
            existing["seen_by"].append(item.get("source_tool"))


class ResearchAgent:
    def __init__(self, llm: ChatModel, tools: ToolRegistry, limits: Limits) -> None:
        self.llm = llm
        self.tools = tools
        self.limits = limits

    def _execute(self, call: ToolCall) -> ToolResult:
        logger.info("  -> %s(%s)", call.name, call.arguments)
        result = self.tools.execute(call.name, call.arguments)
        if result.ok:
            logger.info("  <- ok: %s", result.summary or "done")
        else:
            logger.warning("  <- error: %s", result.error)
        return result

    def _call_llm(self, messages, offered, tool_choice, compact_versions) -> tuple[Any, str]:
        """Call the LLM. Returns (response, "") on success or (None, error message).

        Recovery: if the request was too large, compact every tool result and try once more.
        """
        try:
            return self.llm.chat(messages, tools=offered, tool_choice=tool_choice), ""
        except LLMError as exc:
            if exc.kind != "too_large":
                return None, str(exc)
            first_error = str(exc)
        if not compact_context(messages, compact_versions, offered, self.limits.max_context_tokens,
                               force_all=True):
            return None, first_error  # nothing left to shrink
        logger.warning("  request too large; compacted all tool results, retrying once (~%d tokens)",
                       estimate_tokens(messages, offered))
        try:
            return self.llm.chat(messages, tools=offered, tool_choice=tool_choice), ""
        except LLMError as exc:
            return None, str(exc)

    def run(self, question: str) -> AgentRunResult:
        # The model doesn't know today's date; without it, it searches for stale years.
        system = f"Today's date is {date.today().isoformat()}.\n\n{SYSTEM_PROMPT}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        schemas = self.tools.schemas()
        evidence: dict[str, dict[str, Any]] = {}
        usage: Counter[str] = Counter()
        compact_versions: dict[str, str] = {}  # tool_call_id -> short version of its result
        result = AgentRunResult(question=question, answer=None, stopped_reason="max_iterations")
        budget_note_sent = False

        for iteration in range(1, self.limits.max_iterations + 1):
            result.iterations = iteration

            if result.total_tokens >= self.limits.max_total_tokens:
                result.stopped_reason = "token_budget"
                logger.warning("Token budget reached (%d tokens)", result.total_tokens)
                break

            # Switch tools off when the tool budget is spent or this is the last LLM call.
            last_chance = iteration == self.limits.max_iterations
            tools_exhausted = result.tool_calls >= self.limits.max_tool_calls
            tool_choice = "none" if (last_chance or tools_exhausted) else "auto"
            if tool_choice == "none" and not budget_note_sent:
                messages.append({"role": "system", "content": BUDGET_EXHAUSTED_NOTE})
                budget_note_sent = True

            # When forcing an answer we send NO tool schemas. tool_choice="none" alone is a
            # request the model can ignore (gpt-oss does); a tool it can't see, it can't call.
            offered = schemas if tool_choice == "auto" else None

            # Context management: keep each request under the per-request token limit.
            limit = self.limits.max_context_tokens
            before = estimate_tokens(messages, offered)
            if before > limit:
                n = compact_context(messages, compact_versions, offered, limit)
                logger.info("  context ~%d tokens > %d: compacted %d older tool result(s), now ~%d",
                            before, limit, n, estimate_tokens(messages, offered))

            logger.info("Iteration %d: calling LLM (tool_choice=%s, request ~%d tokens, run total=%d)",
                        iteration, tool_choice, estimate_tokens(messages, offered), result.total_tokens)
            response, error = self._call_llm(messages, offered, tool_choice, compact_versions)
            if response is None:
                result.stopped_reason = "llm_error"
                result.errors.append(error)
                logger.error("LLM call failed: %s", error)
                break

            result.total_tokens += response.total_tokens

            # No tool calls means the model has written its answer.
            if not response.tool_calls:
                if response.content and response.content.strip():
                    result.answer = response.content.strip()
                    result.stopped_reason = "final_answer"
                else:
                    result.stopped_reason = "empty_response"
                break

            if len(response.tool_calls) > 1:
                logger.info("  model requested %d tools in one turn", len(response.tool_calls))
            messages.append(response.to_assistant_message())

            # Every tool call MUST get a matching tool message, or the next API call fails.
            for call in response.tool_calls:
                if result.tool_calls >= self.limits.max_tool_calls:
                    tool_result = ToolResult(ok=False, error="Tool-call budget exhausted; call not executed.")
                    logger.warning("  skipped %s: tool budget exhausted", call.name)
                else:
                    result.tool_calls += 1
                    usage[call.name] += 1
                    tool_result = self._execute(call)
                    merge_evidence(evidence, tool_result.evidence)
                compact_versions[call.id] = tool_result.to_compact_content()
                if not tool_result.ok:
                    result.errors.append(f"{call.name}: {tool_result.error}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_result.to_message_content(),
                })

        result.tool_usage = dict(usage)
        result.evidence = list(evidence.values())
        if result.answer:
            result.unverified_urls = find_unverified_urls(result.answer, result.evidence)
            if result.unverified_urls:
                logger.warning("Answer cites %d URL(s) that were never retrieved", len(result.unverified_urls))
        return result
