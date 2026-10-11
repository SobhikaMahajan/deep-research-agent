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

## 3. Environment Variables (.env)

```env
# Azure AD Authentication (client credentials flow)
CLIENT_ID=<azure-ad-app-client-id>
CLIENT_SECRET=<azure-ad-app-client-secret>
FOUNDATION_SCOPE=<azure-ad-scope-url>

# Azure OpenAI GPT-4o
LLM_ENDPOINT_4o=<azure-openai-endpoint-url>
LLM_API_VERSION_4o=<api-version, e.g. 2024-02-15-preview>

# Azure AI Search (internal documents)
AZURE_SEARCH_ENDPOINT=<azure-search-endpoint>
AZURE_SEARCH_INDEX=<index-name>
AZURE_SEARCH_KEY=<azure-search-admin-key>

# Tavily Web Search
TAVILY_API_KEY=<tavily-api-key>

# Optional: Domain filtering
TAVILY_USE_TRUSTED_DOMAINS_ONLY=false
TAVILY_FILTER_BLOCKED_DOMAINS=true
```

---

## 4. File-by-File Specification

### 4.1 `config.py` — Configuration Module

Centralizes all configuration using Python dataclasses.

```python
# Classes to implement:

@dataclass
class AzureSearchConfig:
    endpoint: str           # from AZURE_SEARCH_ENDPOINT
    index_name: str         # from AZURE_SEARCH_INDEX
    api_key: str            # from AZURE_SEARCH_KEY
    semantic_config_name: str = "competition-semantic-config-v1"
    default_top_k: int = 5
    # validate() -> bool: checks all required fields are set

@dataclass
class AzureAuthConfig:
    tenant_id: str = "1a407a2d-7675-4d17-8692-b3ac285306e4"  # Philips tenant
    token_url: str  # constructed as https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token
    client_id: str  # from CLIENT_ID
    client_secret: str  # from CLIENT_SECRET
    scope: str      # from FOUNDATION_SCOPE
    # validate() -> bool

@dataclass
class LLMEndpointConfig:
    model: str              # "gpt-4o"
    base_url: str           # from LLM_ENDPOINT_4o
    api_type: str           # "azure"
    api_version: str        # from LLM_API_VERSION_4o
    api_key: Optional[str] = None
    azure_ad_token_provider: Optional[Callable] = None
    temperature: float = 0.0
    # validate() -> bool

class AppConfig:
    # Holds AzureSearchConfig, AzureAuthConfig
    # create_llm_config(token_provider) -> LLMEndpointConfig
    # validate_all() -> dict[str, bool]

# Global instance:
config = AppConfig()
```

### 4.2 `main.py` — Application Entry Point

```python
class ResearchApplication:
    """Main application class."""
    
    def __init__(self, app_config=None):
        # Uses global config or override
        
    def initialize(self):
        # 1. Create AuthService, get initial token to validate
        # 2. Create LLMConfig (ag2) with token_provider=auth_service.get_bearer_token
        # 3. Create AzureSearchService
        # 4. Create TavilyService
        # 5. Create ResearchGroupChatOrchestrator with all services
    
    def _create_llm_config(self) -> LLMConfig:
        # Returns ag2 LLMConfig with:
        #   model="gpt-4o", temperature=0.0,
        #   base_url, api_type="azure", api_version,
        #   azure_ad_token_provider=auth_service.get_bearer_token
    
    def process_query(self, user_query, session_id=None):
        # Auto-initializes if needed
        # Wraps orchestrator.process_query() in StreamlitCallbackHandler context
        # Puts None into message_queue when done (signals end of stream)
        # Returns the response

# Global singleton:
_app = None
def get_app() -> ResearchApplication: ...

# CLI support:
def main():
    # python main.py "query here"
```

### 4.3 `app.py` — Streamlit Frontend

```python
# Page config: page_title="AI-Powered Research", layout="wide"
# Custom CSS applied (from streamlit_helpers)

def process_query_async(prompt, session_id):
    """Runs in a background thread. Calls get_app().process_query()"""

def render_response_stream(reasoning_placeholder, final_answer_placeholder, reasoning_expander, timeout=60):
    """
    Polls message_queue in a loop:
    - Messages from "Research_Manager_Agent" go to final_answer_placeholder
    - Messages from other agents go to reasoning_placeholder (inside expander)
    - None message = end of stream
    - Returns (final_answer_content, reasoning_content)
    """

def process_prompt(prompt):
    """
    1. Append user message to st.session_state.messages
    2. Clear message_queue
    3. Start background thread with process_query_async
    4. Create UI placeholders: spinner, reasoning expander, final answer
    5. Stream via render_response_stream
    6. Show latency info
    7. Append assistant message to history
    """

def main():
    """
    - initialize_session()
    - sample_questions_sidebar()
    - st.title("Deep Research Agent")
    - display_chat_history()
    - Handle sample question selection
    - Handle chat input
    """
```

### 4.4 `queue_handler.py` — Message Queue

```python
class MessageType(Enum):
    AGENT_MESSAGE = "agent_message"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    END_OF_STREAM = "end_of_stream"

@dataclass
class QueueMessage:
    message_type: MessageType
    agent_name: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict] = None

class MessageQueueHandler:
    """Thread-safe wrapper around queue.Queue"""
    # put(message), get(block, timeout), get_nowait()
    # put_agent_message(agent_name, content)
    # put_error(error_message, agent_name)
    # signal_end() — puts None
    # clear() -> int — drains queue, returns count
    # empty() -> bool
    # qsize property

# Global instance:
message_queue = MessageQueueHandler()
```

### 4.5 `streamlit_callback.py` — Agent Message Capture

This is the key integration layer. It **monkey-patches** `ConversableAgent.send` to intercept messages from specific agents and forward them to the message queue.

```python
class StreamlitCallbackHandler:
    """Context manager that patches ConversableAgent.send"""
    
    FORWARDED_AGENTS = {"reason_agent", "tot_thinker", "web_surfer_agent", "Research_Manager_Agent"}
    
    def __enter__(self):
        # Save original ConversableAgent.send
        # Replace with instrumented version that:
        #   1. Extracts content from message (dict with "content" key or str)
        #   2. If agent_instance.name is in FORWARDED_AGENTS:
        #      - JSON-serializes {"agent_name": name, "content": content}
        #      - Puts it into message_queue
        #   3. Calls original send()
    
    def __exit__(self):
        # Restores original ConversableAgent.send

# Also provide a @contextmanager version:
def capture_agent_messages(forward_agents=None): ...
```

### 4.6 `streamlit_helpers.py` — UI Components

```python
# CUSTOM_CSS: extensive CSS for .reasoning-box, .final-answer, .latency-info, .loader (spinner)
# Key styles:
#   - reasoning-box: bordered, scrollable, muted colors, max-height 400px
#   - final-answer: clean typography with heading size overrides
#   - loader: spinning border animation
#   - thinking-container: flex layout with spinner + text

@dataclass
class SampleQuestion:
    text: str
    category: Optional[str] = None

DEFAULT_SAMPLE_QUESTIONS = [
    SampleQuestion(
        text="Help me create a competitive strategy against mindray's latest product launched in 2025 in patient monitoring segment...",
        category="Strategy"
    ),
    SampleQuestion(
        text="Compare MX750 VS GE Healthcare products in patient monitoring segment...",
        category="Comparison"
    ),
    SampleQuestion(
        text="Consider philips mx400. Which is the competitor of this product in market...",
        category="Competition"
    ),
]

def render_spinner(message="Thinking...") -> str: # returns HTML
def sample_questions_sidebar(questions=None, show_clear_button=True): ...
def clear_session(): ...
def initialize_session(): ...  # messages=[], session_id=uuid4()
def display_chat_history(): ...  # shows reasoning in expander, final answer in div
def display_error(error_message): ...
def display_info(info_message): ...
```

### 4.7 `agents/prompts.py` — System Prompts

```python
class AgentPrompts:
    INTERNAL_DATA_AGENT = """
    You are the Internal Data Agent. Specialize in retrieving info from internal documents.
    Use search_internal_docs tool. Return citations (document title + page number).
    Use exact query without modifications.
    """

    WEB_SURFER_AGENT = """
    # PURPOSE: Iterative sequential web searches (max 6 subqueries)
    # MUST call plan_web_search first
    # Sequential loop: execute query -> extract key points -> rephrase next query
    # ADAPTATION RULES:
    #   - OFF-TOPIC: add stricter keywords
    #   - OUTDATED: add date filters
    #   - SUMMARY: search for original source
    #   - PAYWALLED: try cached/alternate
    # OUTPUT: Executive summary + findings by subtopic + iteration log + citations
    # Focus on Philips vs competitors, favor Philips
    # NEVER stop after first query, NEVER return unsourced assertions
    """

    RESEARCH_MANAGER_AGENT = """
    You are the Research Manager Agent.
    Wait for both internal_data_agent and web_surfer_agent responses.
    Evaluate groundedness:
      - Internal: must have document title + page number
      - External: must have valid URL
    Fuse insights into well-structured lengthy answer.
    Output: Final Answer + Citations (Internal: Document-Page, External: URLs)
    """

    @classmethod
    def get_reasoning_prompt(cls, user_query: str) -> str:
        """Tree of Thoughts reasoning prompt for ReasoningAgent"""
    
    @classmethod
    def get_subquery_generation_prompt(cls, user_query: str) -> str:
        """Prompt to generate JSON subqueries: {"queries": [...]}"""
```

### 4.8 `agents/agent_factory.py` — Agent Factory

```python
class AgentFactory:
    def __init__(self, llm_config: LLMConfig): ...
    
    def create_user_proxy(name="user_proxy") -> UserProxyAgent:
        # human_input_mode="NEVER", code_execution_config=False
    
    def create_tool_executor(name="tool_executor") -> UserProxyAgent:
        # human_input_mode="NEVER", code_execution_config=False
    
    def create_internal_agent(name="internal_data_agent") -> AssistantAgent:
        # Uses AgentPrompts.INTERNAL_DATA_AGENT
    
    def create_web_surfer_agent(name="web_surfer_agent") -> ConversableAgent:
        # Uses AgentPrompts.WEB_SURFER_AGENT
    
    def create_research_manager_agent(name="Research_Manager_Agent") -> ConversableAgent:
        # Uses AgentPrompts.RESEARCH_MANAGER_AGENT
    
    def create_subquery_agent(name="subquery_generation_agent") -> ConversableAgent:
        # No system message
```

### 4.9 `orchestration/group_chat_manager.py` — Main Orchestrator

```python
@dataclass
class OrchestratorConfig:
    max_rounds: int = 60

class ResearchGroupChatOrchestrator:
    def __init__(self, llm_config, azure_search_service, tavily_service, config=None):
        # 1. Create QueryContext (shared state for user query)
        # 2. Create SearchTools with services + llm_config + query_context
        # 3. Create agents via AgentFactory
        # 4. Register tools via ToolRegistry
        # 5. Create GroupChat with SpeakerSelector
    
    def _create_agents(self):
        # user_proxy, tool_executor, internal_agent, web_surfer_agent, research_manager_agent
    
    def _register_tools(self):
        # Uses ToolRegistry.register_search_tools(search_tools, internal_agent, web_surfer_agent, tool_executor)
    
    def _create_group_chat(self):
        # GroupChat with all 5 agents
        # speaker_selection_method=SpeakerSelector(...)
        # GroupChatManager with llm_config
    
    def process_query(self, user_query):
        # Set user query in search_tools context
        # user_proxy.initiate_chat(manager, message=user_query)
```

### 4.10 `orchestration/speaker_selector.py` — Deterministic Speaker Flow

```python
class SpeakerSelector:
    """Custom speaker selection — deterministic conversation flow."""
    
    def __init__(self, user_proxy, tool_executor, internal_agent, web_surfer_agent, research_manager_agent): ...
    
    def __call__(self, last_speaker, groupchat) -> Agent | str:
        """Called by GroupChat for each turn."""
        return self.select_speaker(last_speaker, groupchat)
    
    def select_speaker(self, last_speaker, groupchat):
        """
        Flow:
        1. Start → internal_data_agent (first message)
        2. internal_data_agent:
           - has tool_calls → tool_executor
           - no tool_calls → web_surfer_agent
        3. web_surfer_agent:
           - has tool_calls → tool_executor
           - no tool_calls → Research_Manager_Agent
        4. Research_Manager_Agent → user_proxy (ends conversation)
        5. tool_executor → returns to the agent that made the tool call
           (looks at messages[-2].name to find the caller)
        """
```

### 4.11 `services/auth_service.py` — Azure AD Authentication

```python
@dataclass
class TokenResponse:
    access_token: str
    token_type: str
    expires_in: int
    obtained_at: datetime  # auto-set to utcnow
    
    @property
    def is_expired(self) -> bool:
        # True if within 5 minutes of expiry (300s buffer)

class AuthService:
    def __init__(self, config: AzureAuthConfig): ...
    
    def get_bearer_token(self) -> str:
        # Returns cached token if valid, otherwise acquires new one
        # POST to Azure AD token endpoint with client_credentials grant
        # Sets os.environ["AZURE_AD_TOKEN"] for SDK compatibility
    
    def _acquire_new_token(self) -> str: ...
    def clear_cache(self): ...

class AuthenticationError(Exception): ...
```

### 4.12 `services/azure_search_service.py` — Internal Document Search

```python
@dataclass
class SearchResult:
    title: str
    content: str
    source: str
    page: Optional[int]
    score: Optional[float]
    document_name: Optional[str] = None
    sharepoint_url: Optional[str] = None
    
    def to_snippet(self) -> str:
        # Formats as: **DocName**\ncontent\nDocument: name\nPage: X\nSource: ...\nSharePoint URL: ...

class AzureSearchService:
    def __init__(self, config: AzureSearchConfig): ...
    
    def search(self, query: str, top_k: int = None) -> List[str]:
        # Uses SearchClient with:
        #   query_type="semantic"
        #   semantic_configuration_name from config
        # Parses results into SearchResult objects
        # Sorts by score descending
        # Returns list of formatted snippet strings
    
    def _parse_results(self, results) -> List[SearchResult]:
        # Extracts from Azure Search docs:
        #   name (cleaned of path prefixes), content, metadata (JSON parsed),
        #   page_number, base_folder/path, sharepoint_link, document_type, product_family
        #   @search.score

class SearchServiceError(Exception): ...
```

**Azure Search Index Fields Used:**
- `name` — document name
- `content` — document text
- `metadata` — JSON string with additional info
- `page_number` — page number
- `path` — file path
- `sharepoint_link` — SharePoint URL
- `base_folder` — source folder
- `document_type` — type classification
- `product_family` — product family tags (array)

### 4.13 `services/tavily_service.py` — Web Search with Domain Filtering

This is the most complex service. Key features:
- **Trusted domains list**: Philips, FDA, NIH, WHO, Reuters, etc.
- **Company domain mapping**: 60+ medical device companies mapped to their official domains
- **Blocked domains**: LinkedIn, Medium, Facebook, Reddit, YouTube, etc.
- **Auto-detection**: Detects company names in queries and enforces company-specific domain filtering
- **Configurable**: Strict mode (trusted only) or permissive with blocked filtering

```python
# Constants at module level:
TRUSTED_DOMAINS: Set[str] = {
    "philips.com", "usa.philips.com", "fda.gov", "nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "who.int", "reuters.com", "bloomberg.com", "medtechdive.com", "healthcaredive.com",
    ... # ~40 domains
}

COMPANY_DOMAINS: Dict[str, Set[str]] = {
    "mindray": {"mindray.com", "www.mindray.com", "mindrayna.com"},
    "siemens": {"siemens-healthineers.com", ...},
    "ge healthcare": {"gehealthcare.com", ...},
    "medtronic": {"medtronic.com", ...},
    "masimo": {"masimo.com", ...},
    ... # 60+ medical device companies
}

BLOCKED_DOMAINS: Set[str] = {
    "linkedin.com", "medium.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "reddit.com", "quora.com", "youtube.com", "slideshare.net",
    "scribd.com", "glassdoor.com", "indeed.com",
}

@dataclass
class TavilyConfig:
    api_key: str
    max_results: int = 6
    topic: str = "general"
    search_depth: str = "advanced"
    use_trusted_domains_only: bool = False
    filter_blocked_domains: bool = True
    trusted_domains: Set[str] = TRUSTED_DOMAINS
    blocked_domains: Set[str] = BLOCKED_DOMAINS
    
    @classmethod
    def from_env(cls) -> TavilyConfig: ...

@dataclass
class WebSearchResult:
    url: str
    title: str
    content: str
    score: Optional[float] = None
    def to_dict(self): ...

class TavilyService:
    def __init__(self, config=None):
        # Uses TavilyConfig.from_env() by default
    
    def search(self, query: str) -> Dict[str, Any]:
        # 1. Detect company names in query
        # 2. Get company-specific trusted domains
        # 3. Call Tavily via langchain_tavily.TavilySearch
        # 4. Filter results (blocked domains, untrusted domains)
        # 5. Return filtered results
    
    def search_multiple(self, queries: List[str], deduplicate=True) -> List[Dict]:
        # Iterates queries, aggregates results, deduplicates by URL
    
    def _detect_companies_in_query(self, query) -> Set[str]:
        # Word-boundary regex matching against COMPANY_DOMAINS keys
    
    def _filter_results(self, results, company_domains=None) -> List[Dict]:
        # Removes blocked domains
        # If company detected: only allow company domains + general trusted
        # If strict mode: only allow trusted domains
    
    @staticmethod
    def split_query(query, delimiter=r"[.]") -> List[str]: ...
    
    # Domain management: add/remove trusted/blocked domains
    def add_trusted_domain(domain), remove_trusted_domain(domain)
    def add_blocked_domain(domain), remove_blocked_domain(domain)
    def set_strict_mode(enabled: bool)

class TavilyServiceError(Exception): ...
```

### 4.14 `tools/search_tools.py` — Tool Implementations

```python
@dataclass
class QueryContext:
    last_user_query: str = ""
    last_subqueries: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)

class SearchTools:
    def __init__(self, azure_search_service, tavily_service, llm_config, query_context=None): ...
    
    def set_user_query(self, query): ...
    
    def search_internal_docs(self, query, top_k=5) -> List[str]:
        # Delegates to azure_search_service.search()
    
    def tavily_web_search(self, query) -> Dict:
        # Delegates to tavily_service.search()
    
    def plan_web_search(self) -> Dict:
        # 1. Create ReasoningAgent with LATS method (nsim=3, max_depth=4)
        # 2. Create subquery_generation_agent (ConversableAgent)
        # 3. Create temp UserProxyAgent (human_input_mode=NEVER)
        # 4. Run reasoning: user_proxy.initiate_chat(reason_agent, reasoning_prompt)
        # 5. Run subquery gen: user_proxy.initiate_chat(subquery_agent, subquery_prompt)
        # 6. Parse JSON from response (robust: try json.loads, fallback to regex)
        # 7. Return {"reasoning": summary, "queries": [...], "reasoning1": reasoning_summary}
    
    def advanced_web_search(self, query, additional_queries=None) -> List[Dict]:
        # Split query by ".", combine with additional_queries
        # Delegate to tavily_service.search_multiple()
    
    @staticmethod
    def _parse_json_response(text) -> Dict:
        # Try json.loads, fallback to regex r"\{.*\}" extraction
```

### 4.15 `tools/tool_registry.py` — Tool Registration

```python
# Module-level singleton pattern:
_search_tools_instance = None

# Module-level functions (required by AG2's register_function):
def search_internal_docs(query: str, top_k: int = 5) -> List[str]: ...
def plan_web_search() -> Dict[str, Any]: ...
def advanced_web_search(query: str, additional_queries=None) -> List[Dict]: ...
# Each delegates to _search_tools_instance

class ToolRegistry:
    @staticmethod
    def register_tool(func, caller, executor, name, description):
        # Wraps autogen.register_function()
    
    @classmethod
    def register_search_tools(cls, search_tools, internal_agent, web_surfer_agent, tool_executor):
        # Sets _search_tools_instance = search_tools
        # Registers 3 tools:
        #   1. "search_internal_docs" — caller=internal_agent, executor=tool_executor
        #      desc: "Search Azure AI Search index and return relevant document snippets."
        #   2. "plan_web_search" — caller=web_surfer_agent, executor=tool_executor
        #      desc: "Create a concise visible reasoning summary and a list of search queries."
        #   3. "search_web" (func=advanced_web_search) — caller=web_surfer_agent, executor=tool_executor
        #      desc: "Perform an advanced web search using query parts and return aggregated results."
```

### 4.16 `tavily_file.py` — Simple Tavily Wrapper (Legacy)

```python
# Used by legacy orchestration.py
from langchain_tavily import TavilySearch

def tavily_func(query: str):
    tool = TavilySearch(max_results=6, topic="general", search_depth="advanced")
    return tool.invoke({"query": query})
```

---

## 5. Agent Conversation Flow

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

## 6. Key Design Patterns

### 6.1 Streaming via Message Queue
- `StreamlitCallbackHandler` patches `ConversableAgent.send` at runtime
- Intercepts messages from specific agents, serializes as JSON, puts in `MessageQueueHandler`
- Streamlit's main thread polls the queue in `render_response_stream()`
- `None` sentinel signals end of stream

### 6.2 Thread Safety
- Agent processing runs in a daemon thread (`threading.Thread`)
- Communication via `queue.Queue` (thread-safe)
- Session state managed by Streamlit's built-in mechanism

### 6.3 Tool Registration
- AG2's `register_function` requires module-level functions (not methods)
- `tool_registry.py` uses a module-level singleton `_search_tools_instance` to bridge this
- Module-level functions delegate to the singleton's methods

### 6.4 Reasoning with Tree of Thoughts
- `plan_web_search` creates a temporary `ReasoningAgent` with LATS method
- Config: `{"method": "lats", "nsim": 3, "max_depth": 4}`
- The reasoning agent generates a strategic plan
- A separate `subquery_generation_agent` converts the plan to actionable search queries (JSON)

### 6.5 Domain Filtering (Tavily)
- Auto-detects company names in queries using regex word-boundary matching
- When company detected: only allows results from company's official domains + trusted sources
- Always blocks social media / job sites (LinkedIn, Reddit, etc.)
- Configurable strict mode: only trusted domains allowed

---

## 7. Key Dependencies (requirements.txt — essential ones)

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

## 8. How to Run

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

## 9. Legacy Files (for reference, not used in main flow)

### `orchestration.py` — Original monolithic orchestration
Contains the original non-modular implementation with all agents, tools, and orchestration in one file. The refactored version splits this into `agents/`, `orchestration/`, `services/`, and `tools/` packages.

### `prompts.py` (root level) — Original prompts
Contains the raw prompt strings (reasoning_prompt, subquery_prompt, internal_prompt, web_prompt, research_prompt). These are now in `agents/prompts.py` as a class.

---

## 10. Important Implementation Details

### 10.1 Azure Search Index Schema
The Azure AI Search index uses these field names:
- `name`, `content`, `metadata` (JSON string), `page_number`, `path`
- `sharepoint_link`, `base_folder`, `document_type`, `product_family` (collection)
- Semantic configuration: `competition-semantic-config-v1`

### 10.2 LLM Configuration
- Model: `gpt-4o` via Azure OpenAI
- Temperature: `0.0` (deterministic)
- Authentication: Azure AD token provider (client credentials flow, not API key)
- Token auto-refreshes with 5-minute expiry buffer

### 10.3 GroupChat Settings
- `max_round=60` — allows extensive multi-turn conversations
- Custom `speaker_selection_method` (callable `SpeakerSelector` instance)
- `GroupChatManager` wraps the chat with LLM config

### 10.4 ReasoningAgent Configuration
- Uses LATS (Language Agent Tree Search) method
- `nsim=3` — number of simulations
- `max_depth=4` — maximum reasoning depth
- `silent=False` — outputs reasoning to console

### 10.5 Content Processing in UI
- Agent reasoning content: HTML-escaped, heading sizes reduced (# → ###)
- Final answer: rendered as raw HTML in `<div class="final-answer">`
- Response latency tracked and displayed
- Chat history persisted in `st.session_state.messages`

---

## 11. Package `__init__.py` Files

### `agents/__init__.py`
```python
from .agent_factory import AgentFactory
from .prompts import AgentPrompts
__all__ = ["AgentFactory", "AgentPrompts"]
```

### `orchestration/__init__.py`
```python
from .group_chat_manager import ResearchGroupChatOrchestrator
from .speaker_selector import SpeakerSelector
__all__ = ["ResearchGroupChatOrchestrator", "SpeakerSelector"]
```

### `services/__init__.py`
```python
from .auth_service import AuthService
from .azure_search_service import AzureSearchService
from .tavily_service import TavilyService
__all__ = ["AuthService", "AzureSearchService", "TavilyService"]
```

### `tools/__init__.py`
```python
from .tool_registry import ToolRegistry, search_internal_docs, plan_web_search, advanced_web_search
from .search_tools import SearchTools, QueryContext
__all__ = ["ToolRegistry", "SearchTools", "QueryContext", "search_internal_docs", "plan_web_search", "advanced_web_search"]
```
