"""What a tool is, and how to run one safely.

A tool has two halves:
  * a *schema* (name, description, JSON-Schema parameters) that we send to the LLM
    so it knows the tool exists, what it is for, and how to call it;
  * a *Python function* that we run when the LLM asks for it.

The LLM never runs code. It only writes a request ("call web_search with
{"query": ...}"). Our code validates that request, runs the function, and
sends the result back. `Tool.run` is that safety boundary.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    # Sources this call retrieved, in one common shape for every tool:
    #   {"url", "title", "published_date", "snippet", "source_tool"}
    # The agent records these to check citations. They are NOT sent to the LLM
    # separately: the LLM sees `data`.
    evidence: list[dict[str, Any]] = field(default_factory=list)
    # One line for the logs, e.g. "5 results" or "4,000 chars from bot.go.tz".
    summary: str = ""

    def to_message_content(self) -> str:
        """Serialise for the LLM. Errors are sent too, so the model can adapt."""
        if self.ok:
            return json.dumps({"ok": True, "data": self.data}, ensure_ascii=False)
        return json.dumps({"ok": False, "error": self.error}, ensure_ascii=False)

    def to_compact_content(self, excerpt_chars: int = 400) -> str:
        """A short version of this result, used when the context gets too big.

        Keeps what identifies each source (title, url, date) plus a short excerpt,
        and drops the bulk. The full result is still in the agent's evidence store.
        """
        if not self.ok:
            return self.to_message_content()  # errors are already short
        sources = [{"title": e.get("title"), "url": e.get("url"), "published_date": e.get("published_date"),
                    "excerpt": (e.get("snippet") or "")[:excerpt_chars]} for e in self.evidence]
        return json.dumps({"ok": True, "compacted": True,
                           "note": "Older result shortened to save space. Call the tool again if you need full detail.",
                           "sources": sources}, ensure_ascii=False)


def make_evidence(*, url: str, title: str, published_date: str | None, snippet: str,
                  source_tool: str, **extra: Any) -> dict[str, Any]:
    """Build one evidence record. Keeping this in one place keeps the shape consistent."""
    return {"url": url, "title": title, "published_date": published_date,
            "snippet": snippet, "source_tool": source_tool, **extra}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments
    func: Callable[..., ToolResult]

    def schema(self) -> dict[str, Any]:
        """The definition sent to the LLM in the `tools` field."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, raw_arguments: str) -> ToolResult:
        """Parse the model's arguments, call the function, and never raise."""
        try:
            args = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return ToolResult(ok=False, error=f"Arguments were not valid JSON: {exc}")
        if not isinstance(args, dict):
            return ToolResult(ok=False, error="Arguments must be a JSON object.")

        properties = self.parameters.get("properties", {})
        missing = [p for p in self.parameters.get("required", []) if p not in args]
        if missing:
            return ToolResult(ok=False, error=f"Missing required argument(s): {', '.join(missing)}")

        unknown = [k for k in args if k not in properties]
        if unknown:
            logger.warning("Tool %s: ignoring unknown argument(s) %s", self.name, unknown)
            args = {k: v for k, v in args.items() if k in properties}

        try:
            result = self.func(**args)
        except Exception as exc:  # a bug inside a tool must not crash the whole run
            logger.exception("Tool %s raised", self.name)
            return ToolResult(ok=False, error=f"Tool crashed: {type(exc).__name__}: {exc}")

        if not isinstance(result, ToolResult):
            return ToolResult(ok=False, error=f"Tool returned malformed output ({type(result).__name__}).")
        return result
