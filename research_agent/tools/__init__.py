from ..config import Settings
from .base import Tool, ToolResult, make_evidence
from .fetch_page import make_fetch_page_tool
from .registry import ToolRegistry
from .rss import FEED_CATALOGUE, make_rss_tool
from .web_search import make_web_search_tool


def build_default_registry(settings: Settings) -> ToolRegistry:
    """Every research tool the agent can use. Add new tools here."""
    return ToolRegistry([
        make_web_search_tool(settings),
        make_fetch_page_tool(settings),
        make_rss_tool(settings),
    ])


__all__ = ["Tool", "ToolResult", "ToolRegistry", "make_evidence", "build_default_registry",
           "make_web_search_tool", "make_fetch_page_tool", "make_rss_tool", "FEED_CATALOGUE"]
