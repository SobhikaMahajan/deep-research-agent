"""
Deterministic speaker selection for the research group chat.
Controls the conversation flow between agents without LLM-based selection.
"""

import logging

logger = logging.getLogger(__name__)


class SpeakerSelector:
    """
    Custom speaker selection — deterministic conversation flow.

    Flow:
    1. Start → internal_data_agent (first message)
    2. internal_data_agent → tool_executor (if tool_calls) OR web_surfer_agent
    3. web_surfer_agent → tool_executor (if tool_calls) OR Research_Manager_Agent
    4. Research_Manager_Agent → user_proxy (ends conversation)
    5. tool_executor → returns to the agent that made the tool call
    """

    def __init__(self, user_proxy, tool_executor, internal_agent, web_surfer_agent, research_manager_agent):
        self.user_proxy = user_proxy
        self.tool_executor = tool_executor
        self.internal_agent = internal_agent
        self.web_surfer_agent = web_surfer_agent
        self.research_manager_agent = research_manager_agent

        # Map agent names to agent objects for lookup
        self._agents = {
            user_proxy.name: user_proxy,
            tool_executor.name: tool_executor,
            internal_agent.name: internal_agent,
            web_surfer_agent.name: web_surfer_agent,
            research_manager_agent.name: research_manager_agent,
        }

    def __call__(self, last_speaker, groupchat):
        """Called by GroupChat for each turn."""
        return self.select_speaker(last_speaker, groupchat)

    def select_speaker(self, last_speaker, groupchat):
        """
        Determine the next speaker based on deterministic rules.

        Args:
            last_speaker: The agent that just spoke
            groupchat: The GroupChat instance

        Returns:
            The next agent to speak
        """
        messages = groupchat.messages

        # Rule 1: First message → internal_data_agent
        if last_speaker == self.user_proxy:
            logger.debug("Flow: user_proxy → internal_data_agent")
            return self.internal_agent

        # Rule 2: internal_data_agent
        if last_speaker == self.internal_agent:
            if self._has_tool_calls(messages):
                logger.debug("Flow: internal_data_agent → tool_executor (tool call)")
                return self.tool_executor
            else:
                logger.debug("Flow: internal_data_agent → web_surfer_agent (done)")
                return self.web_surfer_agent

        # Rule 3: web_surfer_agent
        if last_speaker == self.web_surfer_agent:
            if self._has_tool_calls(messages):
                logger.debug("Flow: web_surfer_agent → tool_executor (tool call)")
                return self.tool_executor
            else:
                logger.debug("Flow: web_surfer_agent → Research_Manager_Agent (done)")
                return self.research_manager_agent

        # Rule 4: Research_Manager_Agent → user_proxy (end)
        if last_speaker == self.research_manager_agent:
            logger.debug("Flow: Research_Manager_Agent → user_proxy (END)")
            return self.user_proxy

        # Rule 5: tool_executor → return to calling agent
        if last_speaker == self.tool_executor:
            caller = self._find_tool_caller(messages)
            if caller:
                logger.debug(f"Flow: tool_executor → {caller.name} (return to caller)")
                return caller
            else:
                # Fallback: go to web_surfer_agent
                logger.warning("Could not determine tool caller, falling back to web_surfer_agent")
                return self.web_surfer_agent

        # Fallback
        logger.warning(f"Unexpected speaker: {last_speaker.name}, defaulting to user_proxy")
        return self.user_proxy

    def _has_tool_calls(self, messages: list) -> bool:
        """Check if the last message contains tool calls."""
        if not messages:
            return False

        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            # Check for tool_calls in the message
            if "tool_calls" in last_msg and last_msg["tool_calls"]:
                return True
            # Check for function_call
            if "function_call" in last_msg and last_msg["function_call"]:
                return True
        return False

    def _find_tool_caller(self, messages: list):
        """
        Find the agent that made the tool call by looking at messages[-2].
        The tool executor responds to the agent that called it.
        """
        if len(messages) < 2:
            return None

        # Look at the message before the tool executor's message
        caller_msg = messages[-2]
        if isinstance(caller_msg, dict):
            caller_name = caller_msg.get("name", "")
            return self._agents.get(caller_name)

        return None
