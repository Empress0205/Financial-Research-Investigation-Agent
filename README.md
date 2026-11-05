# Financial Research & Investigation Agent

An autonomous research agent for financial and economic questions. Given a question such as
*"Investigate recent developments in Tanzania's inflation rate"*, it decides what to look up,
gathers evidence with web search, page reading and RSS feeds, and writes a cited answer that
states its own limitations.

The project is built step by step as a learning exercise in agentic AI. Everything up to
Milestone 7 is plain Python with no agent framework, so each mechanism (tool calling,
budgets, context management, evidence tracking) is visible in the code.

## Status

| # | Milestone | Status |
|---|---|---|
| 1 | Single agent with one tool (web search): LLM integration, tool calling | ✅ Done |
| 2 | Several research tools (search, fetch page, RSS): tool selection and execution | ✅ Done |
| 3 | Iterative research loop: agent loops, budgets, stopping conditions | ⏳ Next |
| 4 | Research planner: task decomposition | |
| 5 | Evidence evaluator: source credibility, dates, cross-source comparison | |
| 6 | Agent memory: working state vs persistent storage | |
| 7 | Multi-agent workflow (LangGraph): delegation and coordination | |
| 8 | FastAPI research API | |
| 9 | Production version: tests, logging, rate limits, Docker | |

## Design rules

- **No invented sources or statistics.** Every claim must cite a retrieved source (URL + date).
  After each run the agent flags any cited URL that no tool actually returned.
- **Agreement isn't proof.** Sources are judged on independence, date and methodology (Milestone 5).
- **Hard limits** on LLM iterations, tool calls, total tokens and per-request context size. When
  a limit is hit, the agent stops and reports what it found.
- **Tool failures never crash a run.** Errors and malformed output go back to the model as data.
- **Every answer has a Research Limitations section.**
- **The agent investigates and explains. It never makes financial decisions.**

## How it works

```
question ─► LLM ──tool call──► ToolRegistry ─► web_search / fetch_page / read_rss_feed
             ▲                       │
             └──── tool result ◄─────┘   (repeat until the model answers or a budget runs out)
                                     │
                     evidence store ─┴─► citation check ─► report
```

1. The question and the tool schemas are sent to the LLM (Groq, native tool calling).
2. The model chooses a tool and its arguments. The agent validates and runs the call, and
   appends the result to the conversation.
3. Each tool also reports its sources in one common format. The agent keeps these in an
   evidence store that is separate from the conversation.
4. If a request grows too large, the oldest tool results are shortened (the evidence store is
   kept intact).
5. When the model answers in plain text, the agent checks every cited URL against the evidence store.

## Tools

| Tool | Purpose |
|---|---|
| `web_search` | Find sources. Tavily, with DuckDuckGo as a fallback. |
| `fetch_page` | Read one HTML page or PDF in full. Returns the text, the publication date and where that date was found. |
| `read_rss_feed` | Latest items from a curated list of news and central-bank feeds, or any feed URL. |

`fetch_page` and `read_rss_feed` refuse non-public addresses (for example `localhost` or private
IPs), check every redirect, and cap download size and time.

## Setup

Requires Python 3.11+. These commands are for Windows PowerShell.

```powershell
git clone https://github.com/Empress0205/Financial-Research-Investigation-Agent.git
cd Financial-Research-Investigation-Agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then add your keys to `.env`. Never commit this file.

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com → API Keys (free) |
| `TAVILY_API_KEY` | https://app.tavily.com (free tier; optional, since DuckDuckGo is used without it) |

You can also set `GROQ_MODEL`, `MAX_ITERATIONS`, `MAX_TOOL_CALLS`, `MAX_TOTAL_TOKENS`,
`MAX_CONTEXT_TOKENS` and `PAGE_MAX_CHARS`. See `.env.example`.

## Usage

```powershell
python -m research_agent "Investigate recent developments in Tanzania's inflation rate"
python -m research_agent "What have central bankers said in speeches this week about inflation?" --quiet
```

Logs show each step: which tool the model chose, what came back, and when the context was
shortened. A summary follows the answer: why the run stopped, iterations, tool usage, tokens,
and the sources that were read in full.

On Groq's free tier a run can take a few minutes, because the SDK waits out the per-minute token limit.

## Tests

```powershell
python -m pytest -v
```

The tests use a scripted fake LLM and fake HTTP responses, so they need no API keys or network.

## Project layout

```
research_agent/
  agent.py            the tool-calling loop, budgets, evidence store, context management
  llm.py              Groq wrapper (the only file that talks to Groq)
  config.py           settings and hard limits from .env
  __main__.py         command-line entry point
  tools/
    base.py           Tool / ToolResult and the safe execution boundary
    registry.py       the catalogue of available tools
    http.py           safe HTTP GET (public addresses only, size and time limits)
    web_search.py     Tavily + DuckDuckGo
    fetch_page.py     HTML/PDF text, title and date extraction
    rss.py            RSS/Atom feeds
tests/                unit tests for every module
```

## Disclaimer

This is a research and learning tool. Its output is not financial advice. Check important
figures against the primary source.
