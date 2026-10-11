"""
Thread-safe message queue for streaming agent messages to the Streamlit UI.
"""

import queue
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class MessageType(Enum):
    AGENT_MESSAGE = "agent_message"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    END_OF_STREAM = "end_of_stream"


@dataclass
class QueueMessage:
    """A message in the queue."""
    message_type: MessageType
    agent_name: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict] = field(default_factory=dict)


class MessageQueueHandler:
    """Thread-safe wrapper around queue.Queue for agent message streaming."""

    def __init__(self):
        self._queue = queue.Queue()

    def put(self, message):
        """Put a message into the queue."""
        self._queue.put(message)

    def get(self, block=True, timeout=None):
        """Get a message from the queue."""
        return self._queue.get(block=block, timeout=timeout)

    def get_nowait(self):
        """Get a message without blocking."""
        return self._queue.get_nowait()

    def put_agent_message(self, agent_name: str, content: str):
        """Convenience method to put an agent message."""
        msg = QueueMessage(
            message_type=MessageType.AGENT_MESSAGE,
            agent_name=agent_name,
            content=content,
        )
        self.put(msg)
        logger.debug(f"Queued message from {agent_name}: {content[:100]}...")

    def put_error(self, error_message: str, agent_name: Optional[str] = None):
        """Put an error message into the queue."""
        msg = QueueMessage(
            message_type=MessageType.ERROR,
            agent_name=agent_name,
            content=error_message,
        )
        self.put(msg)
        logger.error(f"Queued error from {agent_name}: {error_message}")

    def signal_end(self):
        """Signal end of stream by putting None."""
        self.put(None)
        logger.debug("End of stream signaled")

    def clear(self) -> int:
        """Drain the queue, return count of cleared messages."""
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    @property
    def qsize(self) -> int:
        """Return approximate queue size."""
        return self._queue.qsize()


# Global instance
message_queue = MessageQueueHandler()
