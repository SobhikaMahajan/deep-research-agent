"""
Agent factory — creates all AG2 agents with appropriate configurations.
"""

import logging
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

from .prompts import AgentPrompts

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating all agents in the research pipeline."""

    def __init__(self, llm_config):
        """
        Args:
            llm_config: AG2 LLMConfig or dict with LLM configuration
        """
        self.llm_config = llm_config

    def create_user_proxy(self, name: str = "user_proxy") -> UserProxyAgent:
        """Create the user proxy agent (entry point)."""
        return UserProxyAgent(
            name=name,
            human_input_mode="NEVER",
            code_execution_config=False,
        )

    def create_tool_executor(self, name: str = "tool_executor") -> UserProxyAgent:
        """Create the tool executor agent."""
        return UserProxyAgent(
            name=name,
            human_input_mode="NEVER",
            code_execution_config=False,
        )

    def create_internal_agent(self, name: str = "internal_data_agent") -> AssistantAgent:
        """Create the internal data search agent."""
        return AssistantAgent(
            name=name,
            system_message=AgentPrompts.INTERNAL_DATA_AGENT,
            llm_config=self.llm_config,
        )

    def create_web_surfer_agent(self, name: str = "web_surfer_agent") -> ConversableAgent:
        """Create the web search agent."""
        return ConversableAgent(
            name=name,
            system_message=AgentPrompts.WEB_SURFER_AGENT,
            llm_config=self.llm_config,
        )

    def create_research_manager_agent(self, name: str = "Research_Manager_Agent") -> ConversableAgent:
        """Create the research manager (fusion) agent."""
        return ConversableAgent(
            name=name,
            system_message=AgentPrompts.RESEARCH_MANAGER_AGENT,
            llm_config=self.llm_config,
        )

    def create_subquery_agent(self, name: str = "subquery_generation_agent") -> ConversableAgent:
        """Create the subquery generation agent (no system message)."""
        return ConversableAgent(
            name=name,
            llm_config=self.llm_config,
        )
