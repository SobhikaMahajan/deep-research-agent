"""
Simple Tavily wrapper — legacy compatibility file.
"""

from langchain_tavily import TavilySearch


def tavily_func(query: str):
    """Simple Tavily search wrapper."""
    tool = TavilySearch(max_results=6, topic="general", search_depth="advanced")
    return tool.invoke({"query": query})
