"""
Tool registration — bridges AG2's register_function with SearchTools methods.
Uses module-level singleton pattern required by AG2.
"""

import logging
from typing import Any, Dict, List, Optional

from autogen import register_function

logger = logging.getLogger(__name__)

# Module-level singleton (required by AG2's register_function)
_search_tools_instance = None


# --- Module-level functions (delegates to singleton) ---

def search_internal_docs(query: str, top_k: int = 5) -> List[str]:
    """Search internal documents and return relevant snippets."""
    if _search_tools_instance is None:
        return ["Error: Search tools not initialized"]
    return _search_tools_instance.search_internal_docs(query, top_k)


def plan_web_search() -> Dict[str, Any]:
    """Create a reasoning plan and list of search queries for web research."""
    if _search_tools_instance is None:
        return {"error": "Search tools not initialized"}
    return _search_tools_instance.plan_web_search()


def advanced_web_search(query: str, additional_queries: Optional[List[str]] = None) -> List[Dict]:
    """Perform advanced web search with multiple query parts."""
    if _search_tools_instance is None:
        return [{"error": "Search tools not initialized"}]
    return _search_tools_instance.advanced_web_search(query, additional_queries)


class ToolRegistry:
    """Manages tool registration with AG2 agents."""

    @staticmethod
    def register_tool(func, caller, executor, name: str, description: str):
        """
        Register a single tool with AG2.

        Args:
            func: The function to register
            caller: Agent that can call the tool
            executor: Agent that executes the tool
            name: Tool name
            description: Tool description
        """
        register_function(
            func,
            caller=caller,
            executor=executor,
            name=name,
            description=description,
        )
        logger.info(f"Registered tool: {name}")

    @classmethod
    def register_search_tools(cls, search_tools, internal_agent, web_surfer_agent, tool_executor):
        """
        Register all search tools with appropriate agents.

        Args:
            search_tools: SearchTools instance
            internal_agent: Agent that searches internal docs
            web_surfer_agent: Agent that searches the web
            tool_executor: Agent that executes tool calls
        """
        global _search_tools_instance
        _search_tools_instance = search_tools

        # 1. Internal document search
        cls.register_tool(
            func=search_internal_docs,
            caller=internal_agent,
            executor=tool_executor,
            name="search_internal_docs",
            description="Search internal document store and return relevant document snippets with citations.",
        )

        # 2. Plan web search (reasoning + subquery generation)
        cls.register_tool(
            func=plan_web_search,
            caller=web_surfer_agent,
            executor=tool_executor,
            name="plan_web_search",
            description="Create a concise visible reasoning summary and a list of search queries for web research.",
        )

        # 3. Advanced web search (execution)
        cls.register_tool(
            func=advanced_web_search,
            caller=web_surfer_agent,
            executor=tool_executor,
            name="search_web",
            description="Perform an advanced web search using query parts and return aggregated, deduplicated results.",
        )

        logger.info("All search tools registered successfully")
