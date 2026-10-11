"""
Streamlit callback handler — monkey-patches ConversableAgent.send
to intercept agent messages and forward them to the message queue.
"""

import json
import logging
from contextlib import contextmanager

from autogen import ConversableAgent

from queue_handler import message_queue

logger = logging.getLogger(__name__)


class StreamlitCallbackHandler:
    """
    Context manager that patches ConversableAgent.send to capture
    messages from specific agents and stream them to the UI.
    """

    FORWARDED_AGENTS = {
        "reason_agent",
        "tot_thinker",
        "web_surfer_agent",
        "Research_Manager_Agent",
    }

    def __init__(self, forward_agents=None):
        self._original_send = None
        self._forward_agents = forward_agents or self.FORWARDED_AGENTS

    def __enter__(self):
        """Patch ConversableAgent.send."""
        self._original_send = ConversableAgent.send
        handler = self  # Capture reference for closure

        def instrumented_send(agent_instance, message, recipient, *args, **kwargs):
            try:
                # Extract content from message
                if isinstance(message, dict):
                    content = message.get("content", "")
                elif isinstance(message, str):
                    content = message
                else:
                    content = str(message) if message else ""

                # Forward messages from tracked agents
                if content and agent_instance.name in handler._forward_agents:
                    queue_data = json.dumps({
                        "agent_name": agent_instance.name,
                        "content": content,
                    })
                    message_queue.put_agent_message(
                        agent_name=agent_instance.name,
                        content=content,
                    )
                    logger.debug(
                        f"Captured message from {agent_instance.name}: {content[:80]}..."
                    )

            except Exception as e:
                logger.error(f"Error in instrumented send: {e}")

            # Always call original send
            return handler._original_send(agent_instance, message, recipient, *args, **kwargs)

        ConversableAgent.send = instrumented_send
        logger.info("StreamlitCallbackHandler: ConversableAgent.send patched")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original ConversableAgent.send."""
        if self._original_send:
            ConversableAgent.send = self._original_send
            logger.info("StreamlitCallbackHandler: ConversableAgent.send restored")
        return False  # Don't suppress exceptions


@contextmanager
def capture_agent_messages(forward_agents=None):
    """Convenience context manager for capturing agent messages."""
    handler = StreamlitCallbackHandler(forward_agents=forward_agents)
    with handler:
        yield handler
