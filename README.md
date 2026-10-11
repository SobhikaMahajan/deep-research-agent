# Deep Research Agent — Full Project Specification

> **Purpose**: This document is a complete, self-contained specification for regenerating the entire codebase of the "Deep Research Agent" project. Paste this into Claude/ChatGPT to reproduce the full application.

---

## 1. Project Overview

This is a **multi-agent research system** built with **Streamlit** (frontend) and **Microsoft AG2/AutoGen** (agent orchestration). It is designed for **Philips Healthcare** competitive intelligence — comparing Philips patient monitoring products against competitors (primarily Mindray, GE Healthcare, Siemens, etc.).

### How It Works
1. User submits a research question via a Streamlit chat UI
2. A **GroupChat** of 5 agents collaborates:
   - `user_proxy` — entry point, injects the user query
   - `tool_executor` — executes tool calls on behalf of other agents
   - `internal_data_agent` — searches internal documents via **Azure AI Search** (semantic search)
   - `web_surfer_agent` — performs iterative web searches via **Tavily API**, guided by a **ReasoningAgent** (Tree of Thoughts / LATS)
   - `Research_Manager_Agent` — fuses internal + web findings into a structured final answer with citations
3. Agent messages stream to the UI in real-time via a thread-safe message queue
4. The final answer is displayed with reasoning trace, citations, and latency info

### Tech Stack
- **Python 3.11+**
- **Streamlit** — web UI
- **AG2 (AutoGen)** — multi-agent framework (`ag2`, `pyautogen`)
- **Azure AI Search** — internal document retrieval (semantic search)
- **Tavily API** (via `langchain-tavily`) — web search
- **Azure AD (OAuth2 client credentials)** — authentication for LLM endpoint
- **Azure OpenAI GPT-4o** — LLM backend

---

## 2. Project Structure

```
Research/
├── app.py                          # Streamlit UI entry point
├── main.py                         # Application bootstrap & query processing
├── config.py                       # Centralized configuration (dataclasses)
├── orchestration.py                # Legacy/backup orchestration (not used by refactored code)
├── prompts.py                      # Legacy prompts file (imports from orchestration_test)
├── queue_handler.py                # Thread-safe message queue for UI streaming
├── streamlit_callback.py           # Monkey-patches ConversableAgent.send to capture messages
├── streamlit_helpers.py            # Streamlit UI components, CSS, session management
├── tavily_file.py                  # Simple Tavily wrapper (used by legacy orchestration.py)
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (not committed)
│
├── agents/
│   ├── __init__.py                 # Exports AgentFactory, AgentPrompts
│   ├── agent_factory.py            # Factory class creating all agents
│   └── prompts.py                  # Centralized system prompts for all agents
│
├── orchestration/
│   ├── __init__.py                 # Exports ResearchGroupChatOrchestrator, SpeakerSelector
│   ├── group_chat_manager.py       # Main orchestrator: creates agents, registers tools, runs GroupChat
│   └── speaker_selector.py         # Custom speaker selection logic (deterministic flow)
│
├── services/
│   ├── __init__.py                 # Exports AuthService, AzureSearchService, TavilyService
│   ├── auth_service.py             # Azure AD OAuth2 token management with caching
│   ├── azure_search_service.py     # Azure AI Search client with result parsing
│   └── tavily_service.py           # Tavily web search with domain filtering (trusted/blocked)
│
└── tools/
    ├── __init__.py                 # Exports ToolRegistry, SearchTools, tool functions
    ├── search_tools.py             # Search tool implementations (internal, web, plan)
    └── tool_registry.py            # Tool registration with AG2's register_function
```

---

## 3. Agent Conversation Flow

```
User Query
    │
    ▼
[user_proxy] ──────► [internal_data_agent]
                           │
                           ├──► [tool_executor] (search_internal_docs)
                           │         │
                           │◄────────┘ (returns results)
                           │
                           ├──► [tool_executor] (may call tool multiple times)
                           │         │
                           │◄────────┘
                           │
                           ▼ (when done, no more tool calls)
                     [web_surfer_agent]
                           │
                           ├──► [tool_executor] (plan_web_search)
                           │         │
                           │◄────────┘ (returns plan + subqueries)
                           │
                           ├──► [tool_executor] (search_web — iterative)
                           │         │
                           │◄────────┘
                           │
                           ▼ (when done)
                     [Research_Manager_Agent]
                           │
                           ▼ (fuses internal + web results)
                     [user_proxy] ◄── END
```

### Speaker Selection Rules (deterministic, not LLM-based):
1. **First message** → `internal_data_agent`
2. **internal_data_agent** → `tool_executor` (if tool_calls in message) OR `web_surfer_agent`
3. **web_surfer_agent** → `tool_executor` (if tool_calls) OR `Research_Manager_Agent`
4. **Research_Manager_Agent** → `user_proxy` (terminates)
5. **tool_executor** → returns to the agent that invoked the tool (by checking `messages[-2].name`)

---

## 4. Key Design Patterns

### 4.1 Streaming via Message Queue
- `StreamlitCallbackHandler` patches `ConversableAgent.send` at runtime
- Intercepts messages from specific agents, serializes as JSON, puts in `MessageQueueHandler`
- Streamlit's main thread polls the queue in `render_response_stream()`
- `None` sentinel signals end of stream

### 4.2 Thread Safety
- Agent processing runs in a daemon thread (`threading.Thread`)
- Communication via `queue.Queue` (thread-safe)
- Session state managed by Streamlit's built-in mechanism

### 4.3 Tool Registration
- AG2's `register_function` requires module-level functions (not methods)
- `tool_registry.py` uses a module-level singleton `_search_tools_instance` to bridge this
- Module-level functions delegate to the singleton's methods

### 4.4 Reasoning with Tree of Thoughts
- `plan_web_search` creates a temporary `ReasoningAgent` with LATS method
- Config: `{"method": "lats", "nsim": 3, "max_depth": 4}`
- The reasoning agent generates a strategic plan
- A separate `subquery_generation_agent` converts the plan to actionable search queries (JSON)

### 4.5 Domain Filtering (Tavily)
- Auto-detects company names in queries using regex word-boundary matching
- When company detected: only allows results from company's official domains + trusted sources
- Always blocks social media / job sites (LinkedIn, Reddit, etc.)
- Configurable strict mode: only trusted domains allowed

---

## 5. Key Dependencies (requirements.txt — essential ones)

```
ag2                          # Microsoft AutoGen v2 (multi-agent framework)
pyautogen                    # Legacy AutoGen imports
azure-search-documents       # Azure AI Search SDK
azure-core                   # Azure SDK core
langchain-tavily             # Tavily search integration
langchain-core               # LangChain core
streamlit                    # Web UI framework
python-dotenv                # .env file loading
requests                     # HTTP client (for auth)
```

---

## 6. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set up .env file with all required variables

# Run with Streamlit
streamlit run app.py

# Or run from CLI
python main.py "Compare MX750 VS GE Healthcare products in patient monitoring"
```

---

## 7. Legacy Files (for reference, not used in main flow)

### `orchestration.py` — Original monolithic orchestration
Contains the original non-modular implementation with all agents, tools, and orchestration in one file. The refactored version splits this into `agents/`, `orchestration/`, `services/`, and `tools/` packages.

### `prompts.py` (root level) — Original prompts
Contains the raw prompt strings (reasoning_prompt, subquery_prompt, internal_prompt, web_prompt, research_prompt). These are now in `agents/prompts.py` as a class.

---

## 8. Important Implementation Details

### 8.1 Azure Search Index Schema
The Azure AI Search index uses these field names:
- `name`, `content`, `metadata` (JSON string), `page_number`, `path`
- `sharepoint_link`, `base_folder`, `document_type`, `product_family` (collection)
- Semantic configuration: `competition-semantic-config-v1`

### 8.2 LLM Configuration
- Model: `gpt-4o` via Azure OpenAI
- Temperature: `0.0` (deterministic)
- Authentication: Azure AD token provider (client credentials flow, not API key)
- Token auto-refreshes with 5-minute expiry buffer

### 8.3 GroupChat Settings
- `max_round=60` — allows extensive multi-turn conversations
- Custom `speaker_selection_method` (callable `SpeakerSelector` instance)
- `GroupChatManager` wraps the chat with LLM config

### 8.4 ReasoningAgent Configuration
- Uses LATS (Language Agent Tree Search) method
- `nsim=3` — number of simulations
- `max_depth=4` — maximum reasoning depth
- `silent=False` — outputs reasoning to console

### 8.5 Content Processing in UI
- Agent reasoning content: HTML-escaped, heading sizes reduced (# → ###)
- Final answer: rendered as raw HTML in `<div class="final-answer">`
- Response latency tracked and displayed
- Chat history persisted in `st.session_state.messages`

---
