## Usage
- Start server with `.venv/bin/python -m uvicorn deep_research_agents.api.main:app --reload --port 8080`
- Open localhost:8080/docs to use FastAPI with Swagger 

# Deep Research Agents 

The code of the agents is based on the "deep reasearch agents from scratch" course by LangChain form this repository https://github.com/langchain-ai/deep_research_from_scratch

They are build with Python and the langgraph library and a FastAPI interface. It uses the Taviliy web search API at the moment. Also get_model() is wired to the OpenRouter API and the thread key is passed as session key so you get a session per run on OpenRouter.

The agents are generally model independent, but the model makes a big difference, it needs to to well on json output and tool calls. Since research with websearch focuses on extracting information relevant to the research brief rather then on summarizing, compressing, or simplifying information, the harness avoids the word "summarizing" and small summarization models altogether to avoid generalization.

## Langgraph + FastAPI
With Langgraph, an agent harness is a graph with nodes that carry the functions. The shared memory that all nodes add to is called `State`. Before you run an agent you create a `thread` with a graph ID (`agent_id` - name of the agent). Threads maintain the state and conversation history. TODO Implement `assistants` for configurable runs of a same agent.

### Endpoints


## Agents

### Research Brief Helper (research_agent_scope)
TODO
clarify pattern for prompt improvement (goal, context, constraints, what good looks like)

### Link Finder
TODO
finds authoritative links relevant to the research brief 

### Research from files, mcp (research_agent_mcp)
TODO

### Subagent for websearch (research_agent_sub)

**What this subagent does:** 
- receives a research sub-topic from a higher-level agent (with the full research brief for context), 
- searches the web, 
- reads what it finds, 
- hands back a citation-backed writeup of everything relevant — without ever asking an AI to "summarize" what it found. Summarizing the web pages and summarizing a final report from those web page summaries was the original design, and it silently dropped facts: in a real early test, a whole country's key data point vanished from a multi-country report, because two separate rounds of AI rewriting each independently judged it not important enough to keep. Instead, every page is checked individually against the actual research question, and anything relevant is kept essentially word-for-word, with its source attached.

**Flow:** `derive_checklist → llm_call ⇄ tool_node → finalize_research` (the AI, in `llm_call`, decides each round whether to search again or stop; `tool_node` runs whatever it asked for and loops back).

**How it works, step by step:**
1. **Turn the topic into a checklist.** Before searching anything, an AI call — forced into a strict list format, not free text — breaks the research topic into the distinct things that need answering (e.g. `["Germany", "Portugal"]` for a multi-country comparison, or an empty list if the topic has no natural sub-parts). This checklist is used later to verify nothing got missed.
2. **Search and read.** The agent searches the web, then for every new page it finds, an AI judges: is this page relevant, and if so, which checklist items does it cover? Crucially, that judgment is anchored to the *specific search query* that surfaced the page — not just "is this vaguely on-topic" — with the broader research topic kept as secondary context. That keeps extraction tightly relevant to why that particular search was run, while still catching genuinely relevant content that doesn't match the query's exact wording.

   The decision-making AI (the one deciding what to search next, separate from the one judging relevance) is not reasoning blind: every search's result comes back to it as a `SOURCE: {title} / URL: {url}` block plus the extracted content, and the full history of every past search stays in view for the rest of the run. So when it reasons things like "I should find the actual standard, not just an article mentioning it," that's a genuine read of the titles/URLs/content it's already seen — not a guess. What it *can't* do yet is act directly on that: there's no tool to re-fetch or re-read a specific page. Its only lever is issuing a new, more targeted search query and hoping that surfaces something better (typically a different, more authoritative URL) — it cannot ask to revisit a URL it already has.
3. **Forced structure, not free text.** Both AI calls above are constrained to a fixed output shape (a checklist; or `{relevant: yes/no, extracted_content: ..., covers: [...]}`). The AI cannot ramble or return something unparseable — every response is guaranteed to be checkable and combinable by ordinary code, without needing another AI call just to make sense of the first one.
4. **Pages aren't checked twice — but nothing is cached.** Every page's URL gets added to a running `visited_urls` list for the rest of that run. This doesn't stop the same page from being *found* again — a later search still queries the web fresh every time, and the same URL can absolutely come back in a new set of results. What it prevents is spending a second AI call re-checking a page already checked: if a search's results include an already-visited URL, that one is simply skipped (the tool's response just notes "N result(s) skipped — already visited"), with no re-fetch and no content shown for it a second time. Nothing is lost, though — whatever was found the first time around already made it into the final writeup; it just isn't repeated mid-run. Note this also means the *raw* page content is never kept anywhere: only the AI's extracted content (or, if extraction ever failed, a short raw excerpt — see Audit trail) is saved. The full raw text of a page exists only for the moment it's being read, then is discarded.
5. **Stop conditions.** The agent keeps going until either the AI itself judges it has enough (a genuine "I'm done"), or a hard limit is hit (see Cost control). Either way, the *reason* the run stopped is recorded and handed back upstream, so a forced cutoff is never confused with "the AI was satisfied it had a complete answer."
6. **Coverage check.** At the end, the checklist from step 1 is checked against what the kept content actually covered. Anything on the checklist that ended up uncovered is flagged as a coverage gap — a safety net independent of whatever the AI itself believes it accomplished.

**Audit trail — what's kept for review:**
- Every fact handed upstream carries its source (title + URL), so any claim can be traced back to where it came from.
- The full, unfiltered record of everything the agent read during the run (`raw_notes`) is kept too, in case the curated writeup ever needs to be checked against the primary material.
- If the AI-based relevance check fails for a page (timeout, API error), that page isn't silently dropped — the agent falls back to a short raw excerpt instead, clearly flagged as "not AI-reviewed," so nothing goes missing without a trace even during an outage.
- Every run writes a step-by-step log (checklist result, every search query and how many results it returned, every failure/fallback, and the final stop reason), tagged with the run's thread ID, so one run's full history can be reconstructed from the logs alone.

**What it hands back to the higher-level agent:**
- `research_findings` — the citation-backed writeup, ready to use.
- `coverage_gaps` — checklist items nothing ended up covering.
- `stop_reason` — why the run ended: the AI decided it was done, or it was cut off by a limit/timeout.
- `raw_notes` — the full unfiltered record, for deeper audit if ever needed.

  *Current caveat:* the sub-agent always computes and returns all four fields, but the supervisor agent above it only reads `research_findings` today — `coverage_gaps` and `stop_reason` aren't acted on yet (see Files below). The signal exists; nothing upstream consumes it yet.

**Cost control:**
- Every AI call made during one run is tagged with that run's ID, so actual spend per run is visible in the OpenRouter dashboard rather than one undifferentiated stream of calls.
- Hard, code-enforced limits bound worst-case cost and runtime. A prompt simply asking the AI to "search at most 5 times" was tested and found unreliable — models ignore it — so these are enforced in code instead: a max number of pages read per run, a max number of search rounds, a wall-clock time budget for the whole run, and a per-call timeout so a single hung AI or search call can't stall the run indefinitely.
- Pages are never re-processed twice within a run (see step 4 above), so the same page is never paid for twice.

**What's configurable** (environment variables, all optional with sensible defaults):
- `MAX_SUBAGENT_LINKS` (default 10) — max unique pages read per run
- `MAX_SUBAGENT_TOOL_ITERATIONS` (default 11) — max search/tool rounds per run
- `SUBAGENT_TIME_BUDGET_SECONDS` (default 600) — wall-clock budget per run
- `SUBAGENT_CALL_TIMEOUT_SECONDS` (default 90) — timeout on each individual AI/search call
- `MODEL_SCOPE` / `MODEL_RESEARCH` / `MODEL_EXTRACT` — which AI model handles checklist derivation, search decisions, and per-page extraction respectively (fall back to `DEFAULT_MODEL` if unset)
- `LOG_LEVEL` (default `INFO`) — how much of the step-by-step log above gets written

FILES:
- `research_agent_sub.py` — the graph itself. Nodes: `derive_checklist` (structured-output checklist, timeout-guarded, falls back to `[]` on timeout) → `llm_call` (checks the resource caps itself before calling the model at all; on a cap hit, a timeout, or the model declining further tool calls, returns a `Command(goto="finalize_research", update={"stop_reason": ...})` — routing and the `stop_reason` write happen together, in one place, so they can't disagree; otherwise `Command(goto="tool_node", ...)`) ⇄ `tool_node` (runs the round's tool calls concurrently, dispatches `tavily_search`/`think_tool`) → `finalize_research` (pure Python, no LLM call — concatenates relevant extractions, computes `coverage_gaps`, reads the already-set `stop_reason` off state). No `add_conditional_edges`/separate routing function — `llm_call` routes itself via `Command`, same pattern `multi_agent_supervisor.py`'s `supervisor`/`supervisor_tools` already use.
- `state_research.py` — `ResearcherState`/`ResearcherOutputState` TypedDicts. `visited_urls` (a `set[str]`, merged across rounds via `operator.or_`) is the cross-round dedup list described above — URLs only, no cached content. Added `stop_reason` (`StopReason` literal: `model_decided` / `llm_timeout` / `link_cap` / `iteration_cap` / `time_budget`).
- `utils.py` — async Tavily search + structured, brief-aware extraction:
  - `tavily_search(...)` — async, timeout-guarded, returns `(formatted_text, extraction_records)`; concurrent per-URL extraction via `asyncio.gather`; falls back to a raw excerpt (flagged `extraction_failed: True`) if extraction fails twice.
  - `extract_relevant_content(...)` — calls the `extract`-role model with structured output (`RelevantExtraction`), one retry before giving up.
- `config.py` — renamed role `summarize` → `extract`; added `max_subagent_links` / `max_subagent_tool_iterations` / `subagent_time_budget_seconds` / `subagent_call_timeout_seconds` / `log_level`; `thread_id_from_config(config)` / `session_kwargs(session_id)` helpers for OpenRouter session grouping.
- `api/main.py` — `logging.basicConfig` configured once at import time (`LOG_LEVEL`-controlled); every module just does `logging.getLogger(__name__)` and inherits it.
- `multi_agent_supervisor.py` — updated to call `researcher_agent_sub` and read `research_findings` (renamed from `compressed_research`). **Not yet updated**: doesn't forward `config`/`session_id` into its parallel sub-agent calls (so OpenRouter session grouping doesn't cover supervisor-spawned runs yet), and doesn't read `coverage_gaps`/`stop_reason` off the sub-agent's output. Both are TODOs for this file's own redesign pass.
- `api/routers/revise.py` — passes a `config` dict into `final_report_generation(...)`, matching its `(state, config)` signature.

### Smarter Subagent (research_subagent_smart)
TODO
can interrupt and wait and search from file

### Main agent for websearch (research_agent_full)
TODO
uses subagents for websearch of link list with research brief for parallel extraction of relevant information of different aspects

### Supervisor Agent (state_supervisor)
TODO
can call smart subagents in parallel for different sources like files, websearch, mcp and adjust sub-prompt according to outcome 

### Research topic in patents
TODO
### Research in specific patent
TODO

## 🚀 Quickstart 

### Prerequisites


### Installation

