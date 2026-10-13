from typing import List, Dict, Any
from config import get_llm
from prompts import MEMORY_REFORMULATE_PROMPT
from database import get_chat_history
from utils import get_logger

logger = get_logger("MemoryAgent")

class MemoryAgent:
    """Agent responsible for context management and query reformulation for follow-up questions."""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)

    def get_history_summary(self, session_id: str, max_turns: int = 5) -> str:
        history = get_chat_history(session_id, limit=max_turns * 2)
        if not history:
            return "No previous conversation history."

        summary_lines = []
        for msg in history:
            role = msg["role"].capitalize()
            summary_lines.append(f"{role}: {msg['content'][:150]}...")
        return "\n".join(summary_lines)

    # Combined follow-up marker set (superset of both prior heuristic versions):
    # single-word pronoun/reference markers...
    _FOLLOWUP_WORD_MARKERS = (
        "it", "its", "this", "that", "these", "those", "them",
        "he", "she", "they", "above", "previous", "earlier",
        "again", "also", "more", "further", "continue", "compare", "same"
    )
    # ...and multi-word phrase markers (checked as substrings, not exact word matches).
    _FOLLOWUP_PHRASE_MARKERS = (
        "explain further", "what about", "how about", "and the"
    )

    def _looks_like_followup(self, query: str) -> bool:
        """
        Cheap heuristic to avoid an LLM call when the query is clearly standalone.
        Combines both prior heuristics: very short queries are treated as likely
        follow-ups (e.g. "what about page 3?", "why?"), and longer queries are only
        flagged as follow-ups if they contain an actual reference marker.
        """
        q = query.strip().lower()
        words = q.split()
        word_count = len(words)

        if word_count <= 4:
            return True

        if any(marker in words for marker in self._FOLLOWUP_WORD_MARKERS):
            return True

        if any(phrase in q for phrase in self._FOLLOWUP_PHRASE_MARKERS):
            return True

        return False

    def reformulate_query_if_needed(self, session_id: str, query: str) -> str:
        """Determines if query is a follow-up and reformulates it to be standalone."""
        history = get_chat_history(session_id, limit=6)
        if not history:
            return query

        # Skip the LLM round-trip entirely when the query clearly doesn't depend on
        # prior context - this is the majority of queries and saves a full LLM call.
        if not self._looks_like_followup(query):
            return query

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        prompt = MEMORY_REFORMULATE_PROMPT.format(history=history_text, query=query)

        try:
            response = self.llm.invoke(prompt)
            standalone = response.content.strip()
            if standalone and len(standalone) > 5:
                logger.info(f"Reformulated query from '{query}' to '{standalone}'")
                return standalone
        except Exception as e:
            logger.warning(f"Failed to reformulate query: {e}")

        return query
