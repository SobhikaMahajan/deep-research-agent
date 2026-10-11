"""
Main orchestrator — creates agents, registers tools, manages the GroupChat.
"""

import logging
from dataclasses import dataclass

from autogen import GroupChat, GroupChatManager

from agents.agent_factory import AgentFactory
from tools.search_tools import SearchTools, QueryContext
from tools.tool_registry import ToolRegistry
from .speaker_selector import SpeakerSelector

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_rounds: int = 60


class ResearchGroupChatOrchestrator:
    """
    Main orchestrator: creates agents, registers tools, and runs the GroupChat.
    """

    def __init__(self, llm_config, chroma_search_service, tavily_service, config=None):
        """
        Args:
            llm_config: AG2 LLM configuration dict or LLMConfig
            chroma_search_service: ChromaSearchService instance
            tavily_service: TavilyService instance
            config: OrchestratorConfig (optional)
        """
        self.llm_config = llm_config
        self.chroma_search_service = chroma_search_service
        self.tavily_service = tavily_service
        self.config = config or OrchestratorConfig()

        # Shared state for the current user query
        self.query_context = QueryContext()

        # Create search tools with services
        self.search_tools = SearchTools(
            chroma_search_service=chroma_search_service,
            tavily_service=tavily_service,
            llm_config=llm_config,
            query_context=self.query_context,
        )

        # Create agents
        self._create_agents()

        # Register tools
        self._register_tools()

        # Create group chat
        self._create_group_chat()

        logger.info("ResearchGroupChatOrchestrator initialized")

    def _create_agents(self):
        """Create all agents using the factory."""
        factory = AgentFactory(self.llm_config)

        self.user_proxy = factory.create_user_proxy()
        self.tool_executor = factory.create_tool_executor()
        self.internal_agent = factory.create_internal_agent()
        self.web_surfer_agent = factory.create_web_surfer_agent()
        self.research_manager_agent = factory.create_research_manager_agent()

        logger.info("All agents created")

    def _register_tools(self):
        """Register search tools with the appropriate agents."""
        ToolRegistry.register_search_tools(
            search_tools=self.search_tools,
            internal_agent=self.internal_agent,
            web_surfer_agent=self.web_surfer_agent,
            tool_executor=self.tool_executor,
        )
        logger.info("Tools registered")

    def _create_group_chat(self):
        """Create the GroupChat with custom speaker selection."""
        # Create speaker selector
        speaker_selector = SpeakerSelector(
            user_proxy=self.user_proxy,
            tool_executor=self.tool_executor,
            internal_agent=self.internal_agent,
            web_surfer_agent=self.web_surfer_agent,
            research_manager_agent=self.research_manager_agent,
        )

        # Create GroupChat with all agents
        self.group_chat = GroupChat(
            agents=[
                self.user_proxy,
                self.tool_executor,
                self.internal_agent,
                self.web_surfer_agent,
                self.research_manager_agent,
            ],
            messages=[],
            max_round=self.config.max_rounds,
            speaker_selection_method=speaker_selector,
        )

        # Create GroupChatManager
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )

        logger.info(f"GroupChat created with max_rounds={self.config.max_rounds}")

    def process_query(self, user_query: str):
        """
        Process a user research query through the agent pipeline.

        Args:
            user_query: The user's research question

        Returns:
            The chat result from the group chat
        """
        # Set the current user query in search tools context
        self.search_tools.set_user_query(user_query)

        # Clear previous messages
        self.group_chat.messages.clear()

        logger.info(f"Processing query: {user_query[:100]}...")

        # Initiate the chat
        result = self.user_proxy.initiate_chat(
            self.manager,
            message=user_query,
        )

        logger.info("Query processing complete")
        return result
