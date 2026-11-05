import pytest

from research_agent.config import Settings
from research_agent.tools import Tool, ToolRegistry, ToolResult, build_default_registry

SCHEMA = {"type": "object", "properties": {}}


def make_tool(name):
    return Tool(name=name, description=f"{name} tool", parameters=SCHEMA,
                func=lambda: ToolResult(ok=True, data=name))


def test_execute_by_name():
    registry = ToolRegistry([make_tool("a"), make_tool("b")])
    assert registry.execute("b", "{}").data == "b"


def test_unknown_tool_lists_available_names():
    result = ToolRegistry([make_tool("a"), make_tool("b")]).execute("c", "{}")
    assert not result.ok and "a, b" in result.error


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        ToolRegistry([make_tool("a"), make_tool("a")])


def test_default_registry_has_the_three_research_tools():
    registry = build_default_registry(Settings())
    assert registry.names() == ["web_search", "fetch_page", "read_rss_feed"]
    for schema in registry.schemas():
        fn = schema["function"]
        assert fn["description"] and fn["parameters"]["type"] == "object"
