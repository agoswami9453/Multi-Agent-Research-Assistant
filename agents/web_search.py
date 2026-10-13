import os
from typing import List, Dict, Any
from config import TAVILY_API_KEY
from utils import get_logger

logger = get_logger("WebSearchAgent")

class WebSearchAgent:
    """Agent responsible for performing real-time web searches using Tavily or fallback APIs."""

    def __init__(self):
        self.api_key = TAVILY_API_KEY or os.getenv("TAVILY_API_KEY", "")

    def search(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """
        Executes web search query and returns structured list of search results.
        Each result contains 'title', 'content', and 'url'.
        """
        if not self.api_key:
            logger.warning("TAVILY_API_KEY is not configured. Attempting fallback mock search.")
            return self._mock_fallback_search(query)

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.api_key)
            response = client.search(query=query, max_results=max_results, search_depth="basic")
            
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", "Web Result"),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "source": "Web Search (Tavily)"
                })
            
            logger.info(f"Tavily web search returned {len(results)} results for query: '{query}'")
            return results

        except Exception as e:
            logger.error(f"Tavily search execution failed: {e}. Falling back to mock search.")
            return self._mock_fallback_search(query)

    def _mock_fallback_search(self, query: str) -> List[Dict[str, Any]]:
        """Fallback simulation for search when API key is unavailable or fails."""
        return [
            {
                "title": f"External Web Search Result for '{query}'",
                "content": f"Real-time information synthesis for '{query}'. Tavily API key can be set in .env for live web results.",
                "url": "https://tavily.com",
                "source": "Web Search (Fallback)"
            }
        ]
