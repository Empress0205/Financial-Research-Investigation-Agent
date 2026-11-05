"""Command-line entry point:  python -m research_agent "your question"."""
from __future__ import annotations

import argparse
import logging
import sys

from .agent import AgentRunResult, ResearchAgent
from .config import load_settings
from .llm import GroqLLM
from .tools import build_default_registry


def print_report(result: AgentRunResult) -> None:
    print("\n" + "=" * 70)
    if result.answer:
        print(result.answer)
    else:
        print(f"No final answer was produced (stopped: {result.stopped_reason}).")

    if result.unverified_urls:
        print("\n## WARNING: unverified citations")
        print("These URLs appear in the answer but were never returned by a tool:")
        for url in result.unverified_urls:
            print(f"  - {url}")

    print("\n" + "-" * 70)
    print(f"Stopped because : {result.stopped_reason}")
    print(f"LLM iterations  : {result.iterations}")
    print(f"Tool calls      : {result.tool_calls}  "
          f"({', '.join(f'{k} x{v}' for k, v in result.tool_usage.items()) or 'none'})")
    print(f"Tokens used     : {result.total_tokens}")
    print(f"Sources found   : {len(result.evidence)}")
    read_in_full = [e for e in result.evidence if "fetch_page" in e.get("seen_by", [])]
    if read_in_full:
        print("Read in full    :")
        for e in read_in_full:
            print(f"  - {e['url']}  (date: {e.get('published_date') or 'not stated'}"
                  f"{'; from ' + e['date_source'] if e.get('date_source') else ''})")
    if not result.answer and result.evidence:
        print("Evidence retrieved before stopping:")
        for e in result.evidence:
            print(f"  - {e['title']} - {e['url']} - {e.get('published_date') or 'date not stated'}")
    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"  - {err}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Financial research agent (Milestone 2)")
    parser.add_argument("question", help="The research question, in quotes")
    parser.add_argument("-q", "--quiet", action="store_true", help="Hide step-by-step logs")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    # HTTP libraries are noisy at INFO; keep the log focused on the agent.
    for noisy in ("httpx", "httpcore", "urllib3", "primp", "ddgs", "pypdf"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = load_settings()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.", file=sys.stderr)
        return 2

    agent = ResearchAgent(
        llm=GroqLLM(api_key=settings.groq_api_key, model=settings.groq_model),
        tools=build_default_registry(settings),
        limits=settings.limits,
    )
    result = agent.run(args.question)
    print_report(result)
    return 0 if result.stopped_reason == "final_answer" else 1


if __name__ == "__main__":
    sys.exit(main())
