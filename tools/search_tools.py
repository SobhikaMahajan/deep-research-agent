"""
Search tool implementations — internal doc search, web search, and planning.
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autogen import ConversableAgent, UserProxyAgent

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Shared state for the current user query."""
    last_user_query: str = ""
    last_subqueries: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)


class SearchTools:
    """
    Search tool implementations that delegate to services.
    Bridges between AG2 tool functions and the service layer.
    """

    def __init__(self, chroma_search_service, tavily_service, llm_config, query_context=None):
        self.chroma_search_service = chroma_search_service
        self.tavily_service = tavily_service
        self.llm_config = llm_config
        self.query_context = query_context or QueryContext()

    def set_user_query(self, query: str):
        """Set the current user query for context."""
        self.query_context.last_user_query = query
        self.query_context.queries.append(query)

    def search_internal_docs(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search internal documents via ChromaDB.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of formatted document snippets
        """
        logger.info(f"Internal search: '{query}' (top_k={top_k})")
        try:
            results = self.chroma_search_service.search(query, top_k=top_k)
            logger.info(f"Internal search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Internal search failed: {e}")
            return [f"Internal search error: {e}"]

    def tavily_web_search(self, query: str) -> Dict:
        """
        Perform a single web search via Tavily.

        Args:
            query: Search query

        Returns:
            Dict with search results
        """
        logger.info(f"Web search: '{query}'")
        return self.tavily_service.search(query)

    def plan_web_search(self) -> Dict[str, Any]:
        """
        Create a reasoning plan and subqueries for web search.
        Uses ReasoningAgent with LATS method + subquery generation agent.

        Returns:
            Dict with 'reasoning', 'queries', and 'reasoning1' keys
        """
        from agents.prompts import AgentPrompts

        user_query = self.query_context.last_user_query
        logger.info(f"Planning web search for: '{user_query[:100]}'")

        try:
            # Step 1: Create ReasoningAgent with LATS
            from autogen.agents.experimental import ReasoningAgent

            reason_agent = ReasoningAgent(
                name="reason_agent",
                llm_config=self.llm_config,
                reason_config={"method": "lats", "nsim": 3, "max_depth": 4},
                silent=False,
            )

            # Step 2: Create subquery generation agent
            subquery_agent = ConversableAgent(
                name="subquery_generation_agent",
                llm_config=self.llm_config,
            )

            # Step 3: Create temp user proxy
            temp_proxy = UserProxyAgent(
                name="plan_proxy",
                human_input_mode="NEVER",
                code_execution_config=False,
            )

            # Step 4: Run reasoning
            reasoning_prompt = AgentPrompts.get_reasoning_prompt(user_query)
            reason_result = temp_proxy.initiate_chat(
                reason_agent,
                message=reasoning_prompt,
                max_turns=1,
            )

            # Extract reasoning summary
            reasoning_summary = ""
            if reason_result and hasattr(reason_result, "chat_history"):
                for msg in reason_result.chat_history:
                    if isinstance(msg, dict) and msg.get("content"):
                        reasoning_summary = msg["content"]

            # Step 5: Run subquery generation
            subquery_prompt = AgentPrompts.get_subquery_generation_prompt(user_query)
            subquery_result = temp_proxy.initiate_chat(
                subquery_agent,
                message=subquery_prompt,
                max_turns=1,
            )

            # Step 6: Parse JSON from response
            subqueries = []
            if subquery_result and hasattr(subquery_result, "chat_history"):
                for msg in reversed(subquery_result.chat_history):
                    if isinstance(msg, dict) and msg.get("content"):
                        parsed = self._parse_json_response(msg["content"])
                        if "queries" in parsed:
                            subqueries = parsed["queries"]
                            break

            if not subqueries:
                # Fallback: split user query
                subqueries = [user_query]

            self.query_context.last_subqueries = subqueries

            return {
                "reasoning": reasoning_summary[:2000] if reasoning_summary else "Reasoning completed",
                "queries": subqueries,
                "reasoning1": reasoning_summary[:1000] if reasoning_summary else "",
            }

        except Exception as e:
            logger.error(f"Plan web search failed: {e}")
            # Fallback: return the user query as-is
            return {
                "reasoning": f"Planning failed, using direct query: {e}",
                "queries": [user_query],
                "reasoning1": "",
            }

    def advanced_web_search(self, query: str, additional_queries: Optional[List[str]] = None) -> List[Dict]:
        """
        Perform advanced web search with multiple queries.

        Args:
            query: Primary search query
            additional_queries: Extra queries to search

        Returns:
            List of aggregated, deduplicated result dicts
        """
        # Split query by periods
        queries = self.tavily_service.split_query(query)

        # Add additional queries
        if additional_queries:
            queries.extend(additional_queries)

        # Deduplicate queries
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        logger.info(f"Advanced web search: {len(unique_queries)} queries")
        return self.tavily_service.search_multiple(unique_queries)

    @staticmethod
    def _parse_json_response(text: str) -> Dict:
        """Parse JSON from agent response, with regex fallback."""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: extract JSON from text using regex
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

        return {}
