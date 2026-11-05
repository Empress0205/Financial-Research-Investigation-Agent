"""The tool registry: the agent's catalogue of capabilities.

With one tool, the agent could hard-code it. With several, we need one place
that answers three questions:
  * Which tools exist?           -> names()
  * How do I describe them?      -> schemas()   (sent to the LLM)
  * Run the one the LLM picked.  -> execute()   (by name, safely)

The agent never imports a specific tool. Adding a tool means registering it
here; the agent loop doesn't change.
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import Tool, ToolResult


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"A tool named '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            # Tell the model what *does* exist, so it can correct itself next turn.
            return ToolResult(ok=False, error=f"Unknown tool '{name}'. Available tools: {', '.join(self.names())}")
        return tool.run(raw_arguments)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
