## Usage
- Start server with `.venv/bin/python -m uvicorn deep_research_agents.api.main:app --reload --port 8080`
- Open localhost:8080/docs to use FastAPI with Swagger 

# Deep Research Agents 

The code of the agents is based on the deep reasearch agents from scratch course by LangChain form this repository https://github.com/langchain-ai/deep_research_from_scratch

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
Pattern:
insert graphic of the graph

Features:
- receives research_topic from an upstream main agent backed by full research_brief for context
- extracts main entities from research_topic to check coverage of extracted content in the end
- graph node tool_call (parallel): 
  - uses web_search and think_tool to find relevant links and reads the input it gets from Tavily: raw_content
  - link prioritization for cost-effectivness:
  - a tool_call with websearch node receives raw_content for each link (up to 3); `extract_relevant_content` judges relevance of raw content to search_query and research_topic and checks if the main entities from the research_topic are present; output: fields: extracted_content (empty or filled), relevant  true/false, covers [entity A, entitiy B]
  - a tool_node execution triggers a tool_call_iterations, currently capped at 11 is never reached
- main agent receives concatenated extracted_content list (not a summary) when the subagent thinks it is done (this final conclusion is not send to the main agent)
- a node, when the subagent finishes or is forced to finish, checks if the covers fields contain the entities and if not there sends a note to the main agent.
- a node `should_continue` checks if the agent should continue based on `last_message.tool_calls` (continues if tools call), BUT befor it checks if time elapsed `subagent_time_budget_seconds` or the number of links `len(visited_urls)`, config: `max_subagent_links: 10`, or the call number was exceeded, if so, it forces finish.

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

