# deep_research_agents — Repository Guide

## What this is

A standalone, deployable backend for a LangGraph-based deep research agent system. It is the production successor to the LangChain Academy "Deep Research from Scratch" course (that course repo, `deep_research_from_scratch`, is a separate sibling project and should be treated as historical reference only, not a dependency).

Two hard architectural constraints, enforced throughout the codebase:
1. **No hardcoded models or prompts.** Every model-usage site is configured, not literal.
2. **Every LLM call in this repo goes through OpenRouter** — never direct `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`. This is a deliberate cross-project preference (OpenRouter gives unified activity/cost logging), not a cost-saving hack. If you ever find an `init_chat_model(...)` call or a direct provider SDK call in this repo, that's a regression — route it through `config.get_model(role)` instead.

Backend/API only. There is no frontend yet — that's a separate, later effort.

## Architecture

```
src/deep_research_agents/
├── config.py                 # Settings (pydantic-settings) + get_model(role) — the ONLY place that knows about OpenRouter
├── prompts.py                 # All prompt templates, ported from the course, unmodified
├── state_scope.py             # AgentState / AgentInputState for scope_research and research_agent_full
├── state_research.py          # ResearcherState / ResearcherOutputState for research_agent_sub / research_agent_mcp
├── state_supervisor.py        # SupervisorState + ConductResearch/ResearchComplete tools
├── utils.py                   # Tavily search pipeline + brief-aware per-link extraction, think_tool, get_today_str
├── research_agent_scope.py    # Graph 1: clarify_with_user -> write_research_brief          -> scope_research
├── research_agent_sub.py      # Graph 2: derive_checklist -> search+reflect loop over Tavily -> researcher_agent_sub
├── research_agent_mcp.py      # Graph 3: same loop but over a local MCP filesystem server    -> agent_mcp
├── multi_agent_supervisor.py  # Graph 4: supervisor fans out parallel research_agent_sub calls -> supervisor_agent
├── research_agent_full.py     # Graph 5: full pipeline (1 -> 4 -> final_report_generation)   -> agent
├── graphs.py                  # AGENT_REGISTRY — maps agent_id to {builder, name, description, input_adapter}
├── db.py                      # ThreadStore — separate `threads` metadata table, own aiosqlite connection
├── gdrive.py                  # Google Drive upload primitives for /export
├── files/coffee_shops_sf.md   # Demo corpus for research_agent_mcp
└── api/
    ├── main.py                 # FastAPI app + lifespan: compiles all 5 graphs with a shared checkpointer at startup
    ├── schemas.py               # Pydantic request/response models
    └── routers/
        ├── agents.py            # GET /agents
        ├── threads.py           # POST/GET /threads, /state, /runs
        ├── revise.py            # POST /threads/{id}/revise
        └── export.py            # POST /threads/{id}/export
```

### The 5 graphs are NOT interchangeable

They don't share an input schema:
- `scope_research`, `agent` (full pipeline) take `{"messages": [...]}`
- `researcher_agent_sub`, `agent_mcp` take `{"researcher_messages": [...], "research_topic": str}`
- `supervisor_agent` takes `{"supervisor_messages": [...], "research_brief": str}`

`graphs.py`'s `AGENT_REGISTRY` carries a per-agent `input_adapter: Callable[[str], dict]` so the API's plain `{"message": "..."}` request body gets translated correctly regardless of which agent a thread is bound to.

### Config and model roles

Six model-usage roles, each independently overridable via env var, all falling back to `DEFAULT_MODEL`:

| Role | Env override | Used in |
|---|---|---|
| `scope` | `MODEL_SCOPE` | research_agent_scope.py (temperature=0.0); also research_agent_sub.py's derive_checklist node (same lightweight-judgment role) |
| `research` | `MODEL_RESEARCH` | research_agent_sub.py, research_agent_mcp.py |
| `compress` | `MODEL_COMPRESS` | research_agent_mcp.py only (max_tokens=32000) — research_agent_sub.py dropped its own compress step, see below |
| `extract` | `MODEL_EXTRACT` | utils.py per-link, brief-aware content extraction (formerly `summarize`/generic webpage summarization) |
| `supervisor` | `MODEL_SUPERVISOR` | multi_agent_supervisor.py |
| `report` | `MODEL_REPORT` | research_agent_full.py final report (max_tokens=32000) |

`get_model(role)` in `config.py` returns a `ChatOpenAI` pointed at `base_url="https://openrouter.ai/api/v1"` — `ChatOpenAI`, not `init_chat_model`'s `"provider:model"` shorthand, because OpenRouter isn't a native LangChain provider prefix. Model names use OpenRouter's `provider/model` slug format, e.g. `openai/gpt-4.1`.

**Important**: `config.py` uses `pydantic-settings`, which parses `.env` into its own `Settings` object and does **not** populate `os.environ`. If you add code that reads `os.environ[...]` directly (e.g. copying a pattern from another repo that calls `load_dotenv()`), it will silently fail to find values that are clearly set in `.env`. Thread values through explicit function arguments sourced from `get_settings()` instead (see `gdrive.py` for the pattern — this bit us once already during the original build).

### OpenRouter session grouping (`session_id`)

Every LLM call in the repo passes `session_id=<thread_id>` to OpenRouter, via `extra_body={"session_id": ...}` on the `.invoke()`/`.ainvoke()` call — so every OpenRouter call made while processing one thread (across every node, and any parallel sub-agents the supervisor spawns) shows up as one session in OpenRouter's dashboard, making cost/usage numbers analyzable per-thread. This is unrelated to LangGraph's own checkpointing; it's purely an OpenRouter-side dashboard grouping key, doc'd at [openrouter.ai/docs/guides/best-practices/prompt-caching#using-session_id-for-sticky-sessions](https://openrouter.ai/docs/guides/best-practices/prompt-caching#using-session_id-for-sticky-sessions). OpenRouter's own `@openrouter/agent` npm package (with `SessionStart`/`SessionEnd` lifecycle hooks) is a **separate, JS/TS-only SDK and does not apply here** — this is a pure-Python repo using `langchain_openai.ChatOpenAI` against OpenRouter's OpenAI-compatible REST endpoint, where a "session" is nothing more than requests sharing a `session_id` value; there's no separate start/end call.

Two small helpers in `config.py` carry this:
- `thread_id_from_config(config: RunnableConfig | None) -> str | None` — pulls `configurable.thread_id` out of a node's config, or `None` if absent (e.g. a graph invoked directly from a script rather than through the API).
- `session_kwargs(session_id: str | None) -> dict` — returns `{"extra_body": {"session_id": session_id}}`, or `{}` if `session_id` is falsy (safe to unconditionally splat into any `.invoke()`/`.ainvoke()` call: `model.ainvoke(messages, **session_kwargs(thread_id_from_config(config)))`).

**How `thread_id` gets to every call site, including deeply nested ones** — every node function across all 5 graphs takes `config: RunnableConfig` as its second parameter (LangGraph's standard, documented pattern: `def my_node(state, config: RunnableConfig)`), which LangGraph auto-injects at the top-level API call (`api/routers/threads.py`'s `_config(thread_id)` sets `configurable.thread_id` once, at `graph.ainvoke(...)`). From there:
- Nodes that call an LLM directly just read `config` themselves.
- `research_agent_sub.py`'s `tool_node` reads `config`, then injects `session_id` into `tavily_search`'s call args alongside `research_topic`/`checklist`/`already_visited` (same `InjectedToolArg` pattern) — `tavily_search` forwards it into every per-URL `extract_relevant_content()` call in `utils.py`.
- `multi_agent_supervisor.py`'s `supervisor_tools` explicitly forwards its own `config` into each parallel `researcher_agent_sub.ainvoke({...}, config)` call — this is the one spot config does **not** auto-propagate, since `researcher_agent_sub` is invoked with a manual `.ainvoke()` call from inside a node rather than added as a declarative LangGraph subgraph node. (Contrast with `research_agent_full.py`, which adds `supervisor_agent` — the compiled supervisor graph — directly via `add_node("supervisor_subgraph", supervisor_agent)`; that *is* LangGraph's native subgraph pattern, so config propagates into it automatically with no special handling.) `researcher_agent_sub` is compiled without its own checkpointer (see the architectural note under point 5 below), so this manual forward is the only way its nodes learn which thread they belong to.
- `api/routers/revise.py` calls `final_report_generation()` directly (bypassing the graph entirely) — it already builds a `config = {"configurable": {"thread_id": thread_id}}` dict locally for `graph.aget_state`/`aupdate_state`, so it just passes that same dict through as the second argument.

If you add a new LLM call anywhere, follow the same pattern: make sure the enclosing node has `config: RunnableConfig`, and pass `**session_kwargs(thread_id_from_config(config))` on the `.invoke()`/`.ainvoke()` call.

### research_agent_sub: per-link extraction, checklist coverage, safety net

`research_agent_sub.py` (Graph 2, agent_id `research_agent_sub`) was redesigned from a "search → generic summarize → generic compress" pipeline into a brief-aware, citation-grounded one. Root cause this fixed: two stacked *generic* (brief-unaware) LLM rewrites — a per-page summary and a final holistic compression — could each independently drop specific facts that weren't obviously salient out of context, biased toward whatever surfaced early/broadly over hard-won, specific, late-arriving details. Real-world case: a multi-country research question lost one country's key fact entirely from the final handoff, despite half the searches targeting it.

Pipeline: `START → derive_checklist → llm_call ⇄ tool_node → finalize_research → END`

- **`derive_checklist`** — one-time structured-output call (reuses the `scope` role model) turning `research_topic` into `checklist: list[str]` (e.g. `["Germany", "Portugal"]` for a multi-country question, `[]` if the topic has no natural sub-parts). Also stamps `started_at` for the time-budget cap below, since it's always the first node.
- **`tavily_search`** (`utils.py`) is now brief-aware: `tool_node` injects `research_topic`, `checklist`, and `already_visited` (the running `visited_urls` set) as `InjectedToolArg`s the LLM never sees or supplies — same hand-rolled special-casing pattern `research_agent_mcp.py` already used for `think_tool` vs MCP tools. For each unique URL in a search's results (already-visited URLs are skipped entirely, no re-extraction), it calls `extract_relevant_content()` — brief-aware, returns a `RelevantExtraction` (`relevant: bool`, `extracted_content: str` verbatim, `covers: list[str]` — which checklist items this page addresses). Returns `(formatted_text, list[extraction_record])`, not a bare string; `tool_node` unpacks this, appends the extraction records to state's `extractions` field, and folds newly-seen URLs into `visited_urls` (`Annotated[set[str], operator.or_]`) — this is the **cross-call link dedup**: revisiting the same URL across search rounds within one sub-agent run is now prevented, where before it was unmetered.
- **`finalize_research`** replaces the old `compress_research` — **pure Python, no LLM call**. Concatenates every `relevant=true` extraction verbatim as `"SOURCE: {title} — {url}\n{extracted_content}"` into `research_findings` (renamed from `compressed_research`). Also computes `coverage_gaps = checklist items no surviving extraction covers` and attaches it to output state alongside `research_findings` — **informational only**, it does not override the model's own stop decision (no new control-flow risk from a heuristic that can't perfectly judge completeness).
- **Safety net** (`config.py` settings, all env-overridable): `max_subagent_links` (10), `max_subagent_tool_iterations` (11, incremented once per `tool_node` call — deliberately a different field from the supervisor's own `max_researcher_iterations`, a different scope entirely, to avoid recreating the naming confusion that caused this redesign), `subagent_time_budget_seconds` (600), `subagent_call_timeout_seconds` (90, wraps `llm_call`'s `ainvoke` and each per-URL extraction call in `asyncio.wait_for`). Unlike the coverage flag above, `should_continue`'s link/iteration/time checks **do** override the model's own tool-call decision — this is a resource cap, not a quality judgment. A timed-out `llm_call` is treated as "no tool calls" (routes to `finalize_research`) rather than raising, since `threads.py`'s bare `except Exception` would otherwise mark the whole thread `"failed"` for one slow call. A timed-out/errored per-URL extraction is skipped without marking that URL visited, so it can be retried by a later search — distinct from a considered `relevant=false` judgment, which does count as visited.
- **`research_agent_mcp.py` intentionally still has its own LLM-based `compress_research` step** — it reads local files via MCP, not Tavily, so none of the above applies to it. It only picked up the `compressed_research` → `research_findings` field rename, since both graphs share `ResearcherState`/`ResearcherOutputState`.

### Persistence

Two separate things share one sqlite file (`SQLITE_PATH`, default `data/threads.db`):
1. **LangGraph's own checkpointer** (`AsyncSqliteSaver`) — opened once in `api/main.py`'s lifespan, used to compile all 5 graphs with `checkpointer=`. This is what makes `/threads/{id}/state` and multi-turn threads work.
2. **`db.py`'s `ThreadStore`** — a plain `threads` table (own `aiosqlite` connection to the same file) holding `thread_id, name, agent_id, status, error_detail, created_at, updated_at`. This is *not* routed through the checkpointer; it's just simple metadata CRUD.

WAL mode is enabled explicitly in `ThreadStore.connect()` to avoid `database is locked` errors from having two separate connections to the same file.

**Known, accepted limitation**: `status` isn't tied to any in-process task registry, so a server restart mid-run leaves that thread's status stuck on `"running"` forever. No startup sweep exists yet — fine for local iteration, worth adding before a real multi-restart deployment.

## API endpoints

- `GET /agents` — list `{id, name, description}` from `AGENT_REGISTRY`.
- `POST /threads` `{name, agent_id}` — create a thread, `400` if `agent_id` isn't registered.
- `GET /threads`, `GET /threads/{id}` — list / single metadata + status.
- `GET /threads/{id}/state` — `graph.aget_state(config).values`. **Shape varies by `agent_id`** — only `research_agent_full` threads have `notes`/`final_report`/`research_brief` all present.
- `POST /threads/{id}/runs` `{message}` — adapts the message via the registry, sets `status="running"`, dispatches as a `BackgroundTasks` job, returns `202` immediately. **Reruns the entire pipeline from START every time** — none of these 5 graphs use `interrupt()`/resume, so there is no cheap incremental continuation. Don't mistake this for a bug; it's an architectural fact of the ported graphs.
  - After a `research_agent_full` run completes, a post-hoc heuristic checks `final_report`/`notes`: if the report is empty/very short or there are no notes, status is set to `"done_empty"` instead of `"done"`. This exists to surface the known supervisor bug below — it is a string-length check, not a retry/fix.
- `POST /threads/{id}/state` `{values}` — raw passthrough to `graph.aupdate_state(config, values)`. Reducer-free fields (`research_brief`, `final_report`) replace cleanly. `operator.add` fields (`notes`, `raw_notes`) and `add_messages` fields (`messages`, `supervisor_messages`, `researcher_messages`) **append**, they don't replace — this is LangGraph reducer behavior, not something this endpoint works around.
- `POST /threads/{id}/revise` `{instruction}` — **only valid for `agent_id="research_agent_full"`** (`400` otherwise, since no other graph has a `final_report_generation` step). Calls `final_report_generation()` directly, bypassing the graph entirely, with current `notes` + a `"USER CORRECTION:\n{instruction}"` entry + `research_brief`, then updates just `final_report`. Runs **synchronously in the request** (one cheap LLM call) — deliberately not backgrounded, unlike `/runs`.
- `POST /threads/{id}/export` — writes `final_report` (or joined `notes` as a fallback for non-full-pipeline agents) to `data/exports/{thread_id}.md`, then uploads it to Google Drive via `gdrive.export_report()`, wrapped in `run_in_threadpool` since `googleapiclient` is synchronous.

## Known, deliberately-not-fixed bug

`multi_agent_supervisor.py`'s `supervisor_tools` node has a bare `except Exception: print(...); should_end = True` around the parallel `ConductResearch` fan-out. Any exception during parallel research (a rate-limited OpenRouter call, a Tavily timeout, anything) silently ends the graph with whatever partial notes were gathered — no retry, no error surfaced to the caller. This is ported verbatim from the original course code on purpose. The `done_empty` status heuristic in `threads.py` is the only mitigation — it makes the failure *visible* after the fact, it does not prevent it. If you're asked to make this more robust, that's new scope, not a bug fix to the existing behavior.

## Environment setup

Required env vars (see `.env.example`):
- `OPENROUTER_API_KEY` — routes all LLM calls
- `TAVILY_API_KEY` — web search for research_agent_sub / research_agent_mcp's sibling graphs (research_agent_mcp itself doesn't call Tavily, it reads local files, but the module still imports utils.py which constructs an `AsyncTavilyClient` at import time)

Optional:
- `DEFAULT_MODEL` (default `openai/gpt-4.1`) and per-role `MODEL_*` overrides
- `MAX_CONCURRENT_RESEARCHERS` (default `2` — deliberately lowered from the course's hardcoded `3`, which caused a real 429 rate-limit during development)
- `MAX_RESEARCHER_ITERATIONS` (default `6`)
- `SQLITE_PATH` (default `data/threads.db`)
- `GDRIVE_CLIENT_ID` / `GDRIVE_CLIENT_SECRET` / `GDRIVE_REFRESH_TOKEN` / `GDRIVE_BASE_PATH` — only needed for `/export`. This repo reuses the same Google Cloud OAuth client already set up for the sibling `patent-translation-app` project by default; that's a personal-account-hygiene choice, not a hard requirement — a fresh OAuth client works identically.

`.env` is gitignored and will **not** exist in a fresh clone or Codespace. Copy `.env.example` to `.env` and fill in real values, or (better, for Codespaces) set these as [Codespaces repository secrets](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-encrypted-secrets-for-your-codespaces) so they're injected as real environment variables — either works, since `pydantic-settings` reads both a `.env` file and real process env vars (the latter takes precedence).

## Running locally

```bash
uv sync                       # installs from pyproject.toml / uv.lock into .venv
cp .env.example .env          # then fill in real values
uv pip install -e .           # editable install so `deep_research_agents` is importable
source .venv/bin/activate
uvicorn deep_research_agents.api.main:app --reload --port 8000
```

Health check: `curl localhost:8000/health` → `{"status": "ok"}`.

## Codespaces-specific notes

- **`research_agent_mcp` needs Node.js/`npx` at runtime** (it spawns `@modelcontextprotocol/server-filesystem` as a subprocess). Most default GitHub Codespaces base images include Node; a minimal Python-only devcontainer image will not. If that graph 404s or hangs on its first tool call, check `which npx` first.
- Outbound network access to `openrouter.ai`, `api.tavily.com`, and (for `/export`) `www.googleapis.com`/`oauth2.googleapis.com` must not be blocked by the Codespace's network policy.
- `data/` (sqlite db + exports) is gitignored and won't exist in a fresh clone — it's created automatically on first run (`Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)` in `api/main.py`'s lifespan).
- `.claude/settings.json` in this repo pre-allows the routine dev-loop commands (`uv sync`, `uvicorn` start/stop, `curl localhost:8000/...`, `python3 -c` smoke tests, common `git`/`ls`/`find` forms) so a fresh Claude Code session here hits fewer permission prompts. Anything destructive (force-push, `git reset --hard`, etc.) is intentionally left out of the allowlist.

## Testing

There is **no automated test suite** (no pytest) as of this writing. Everything was verified during the original build via live manual smoke tests against real APIs — every graph invoked directly with real data, every endpoint hit with real curl requests, including a full `research_agent_full` run end to end. If you add pytest coverage later, that's new work, not something you're filling a gap in.

Manual smoke-test sequence (mirrors how this was originally verified):

```bash
curl -s localhost:8000/agents | python3 -m json.tool

curl -s -X POST localhost:8000/threads -H 'content-type: application/json' \
  -d '{"name": "smoke-test-scope", "agent_id": "scope_research"}'
# grab thread_id from the response, then:
curl -s -X POST localhost:8000/threads/<id>/runs -H 'content-type: application/json' \
  -d '{"message": "I want a research report on coffee shops in SF"}'
curl -s localhost:8000/threads/<id>              # poll: idle -> running -> done
curl -s localhost:8000/threads/<id>/state

# expensive — run deliberately last, takes a few minutes
curl -s -X POST localhost:8000/threads -H 'content-type: application/json' \
  -d '{"name": "smoke-test-full", "agent_id": "research_agent_full"}'
curl -s -X POST localhost:8000/threads/<full_id>/runs -H 'content-type: application/json' \
  -d '{"message": "Research the best 3 espresso machines under $500"}'
# poll until done/done_empty, then:
curl -s -X POST localhost:8000/threads/<full_id>/revise -H 'content-type: application/json' \
  -d '{"instruction": "Make the tone more casual, add a one-line TL;DR at the top."}'
curl -s -X POST localhost:8000/threads/<full_id>/export
```

## Deployment status

**Not yet deployed.** `railway.toml` exists (`startCommand = "uvicorn deep_research_agents.api.main:app --host 0.0.0.0 --port $PORT"`, mirroring the working pattern from the sibling `patent-translation-app` project) but a Railway service, persistent Volume for the sqlite file, and env vars have not been provisioned yet. When that happens:
- Mount a persistent Volume (Railway's filesystem is otherwise ephemeral) and set `SQLITE_PATH` to a path on it, e.g. `/data/threads.db`.
- Decide how to handle `research_agent_mcp`'s Node/npx dependency in prod — a custom `nixpacks.toml` with both Python and Node providers, a Dockerfile, or disabling that one agent in the production `AGENT_REGISTRY`.
- This repo is not yet a git repository as of this writing (git init was deliberately deferred). It needs to be initialized and pushed to GitHub before Railway or Codespaces can use it.

## Repository guide provenance

This file, `.claude/settings.json`, and the repo itself were built in one session by porting all 5 graphs from the LangChain Academy "Deep Research from Scratch" course, then swapping every model call to route through OpenRouter and wrapping the result in this FastAPI service. Every claim above about behavior (reducer semantics, the silent-failure bug, the `os.environ` gotcha) was independently verified by actually running the code, not just read off the course source.
