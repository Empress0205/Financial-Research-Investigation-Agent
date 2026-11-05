# Financial Research & Investigation Agent

## What this project is
An autonomous research agent that takes a financial or economic research question
(for example "Investigate recent developments in Tanzania's inflation rate"), plans the
research, uses tools to gather evidence from several sources, evaluates and compares that
evidence, flags gaps and conflicts, and produces a structured, evidence-backed report.

The goal is learning: Zakia should come out able to explain and defend every component and
understand how the whole system works. Finishing the project matters less than understanding it.

## How to work with me (teaching rules)
- Go one milestone at a time. Do not start a milestone before the previous one runs and passes its tests.
- You write ALL the code: complete and ready to run. Do not leave TODOs or parts for me to implement.
- Before each milestone, explain:
  1. What we're building in this stage and why it's needed.
  2. Which Agentic AI concept it teaches, and how that concept maps to real code.
- While building, explain each file and each important function: what it does, why it's written
  that way, and how it fits into the overall architecture.
- After each milestone, give me:
  1. **What you should learn from this stage:** the key takeaways in a few bullets.
  2. **How to run and test it:** exact PowerShell commands and the expected output.
  3. **What to observe:** what to look for in the agent's behaviour or logs that shows the concept working.
  4. 2–3 short questions to check I understood, with answers I can reveal.
- Explain clearly and in a structured way, without padding.
- Do not add frameworks (LangChain/LangGraph) until Milestone 7. Before then, everything is plain Python,
  so I understand what the frameworks abstract away.

## Environment
- Windows, VS Code, Python 3.11+, virtual environment in `.venv`
- Secrets go in `.env` (never committed): `GROQ_API_KEY`, `TAVILY_API_KEY`
- Commands must work in PowerShell

## Stack
| Component | Technology |
|---|---|
| LLM | Groq API (native tool calling) |
| Web search | Tavily API (fallback: DuckDuckGo) |
| Other retrieval | RSS feeds, public statistics APIs |
| Orchestration | Plain Python first, then LangGraph (Milestone 7) |
| Storage | SQLite, PostgreSQL later |
| Semantic retrieval | Sentence Transformers + ChromaDB |
| API | FastAPI |
| Deployment | Docker |

## Roadmap
| # | Build | Learn |
|---|---|---|
| 1 | Basic single agent with one tool (web search) | LLM integration, tool calling |
| 2 | Multiple research tools (search, fetch page, RSS) | Tool selection and execution |
| 3 | Iterative research loop | Agent loops, budgets, stopping conditions |
| 4 | Research planner | Task decomposition |
| 5 | Evidence evaluator | Source credibility, dates, cross-source comparison |
| 6 | Agent memory | Working state vs persistent storage |
| 7 | Multi-agent workflow (LangGraph) | Delegation and coordination |
| 8 | FastAPI research API | Integration |
| 9 | Production version | Tests, logging, rate limits, Docker |

Each milestone must end with working code and a test I can run.

## Non-negotiable engineering rules
- The agent never invents sources or statistics. Every factual claim in a report links to retrieved evidence (URL + date).
- Agreement between sources is not proof: check independence, dates, and methodology.
- Hard limits on iterations, tool calls, and tokens. The agent must stop and report when it hits them.
- Tool failures and malformed tool output are handled safely, never crash the run.
- Reports always include a Research Limitations section.
- The agent investigates and recommends. It never makes financial decisions on its own.

## Report structure
1. Executive Summary 2. Key Findings 3. Evidence & Source Analysis (source / finding / reliability)
4. Conflicting Information 5. Financial Stability Implications 6. Research Limitations
7. Conclusion, plus references with publication dates.

## Git
Do not add Claude attribution to commits or PRs.

## Progress
- Milestone 1 (single agent + web search): built, 27 tests pass, live run OK (2026-11-05).
  Default model is openai/gpt-oss-120b (llama-3.3-70b-versatile was retired on Groq).
  Live-run issues to address later: all evidence came from one secondary source (M5: independence);
  the model misread a headline ("eases to 4.0%" reported as a "rise") and stated things the
  evidence doesn't support (M5: claim-level checks, since the URL check only proves the source exists).
- Milestone 2 (web_search + fetch_page + read_rss_feed, tool registry, SSRF-safe HTTP, context
  compaction for Groq's 8k tokens-per-request free-tier limit): built, 82 tests pass, live run pending.
