"""Settings and hard limits, read from the environment (.env).

Keeping every limit in one place means the agent's budget is visible and
adjustable without touching the agent's logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Groq retires models over time. List the ones your key can use with:
#   python -c "import groq; [print(m.id) for m in groq.Groq().models.list().data]"
DEFAULT_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class Limits:
    """Hard budgets for a single run. The agent stops when any is reached."""

    max_iterations: int = 8         # how many times we call the LLM
    max_tool_calls: int = 8         # how many tools we execute in total
    max_total_tokens: int = 60_000  # prompt + completion tokens across the run
    # Largest request we send in ONE call (estimated). Groq's free tier refuses any single
    # request over 8,000 tokens per minute for gpt-oss-120b, and the reply needs room too.
    max_context_tokens: int = 5_500


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = None
    tavily_api_key: str | None = None
    groq_model: str = DEFAULT_MODEL
    limits: Limits = field(default_factory=Limits)
    search_max_results: int = 5
    http_timeout_seconds: float = 15.0
    page_max_chars: int = 3_000       # default text returned by fetch_page (~750 tokens)
    page_max_chars_cap: int = 6_000   # the most the model may ask for
    fetch_max_bytes: int = 5_000_000  # refuse downloads bigger than ~5 MB


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def load_settings() -> Settings:
    """Load .env (if present) and build a Settings object."""
    load_dotenv()
    defaults = Limits()
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
        limits=Limits(
            max_iterations=_int_env("MAX_ITERATIONS", defaults.max_iterations),
            max_tool_calls=_int_env("MAX_TOOL_CALLS", defaults.max_tool_calls),
            max_total_tokens=_int_env("MAX_TOTAL_TOKENS", defaults.max_total_tokens),
            max_context_tokens=_int_env("MAX_CONTEXT_TOKENS", defaults.max_context_tokens),
        ),
        page_max_chars=_int_env("PAGE_MAX_CHARS", 3_000),
    )
