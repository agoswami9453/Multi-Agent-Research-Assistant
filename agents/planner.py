import json
from typing import Dict, Any, List
from config import get_llm
from prompts import PLANNER_SYSTEM_PROMPT
from utils import get_logger

logger = get_logger("PlannerAgent")

class PlannerAgent:
    """Agent responsible for analyzing query intent, document relevance, freshness needs, and routing."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)

    def plan_route(
        self,
        query: str,
        has_docs: bool,
        doc_names: List[str],
        history_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Determines the optimal execution route based on query intent and document availability.
        """
        query_lower = query.lower().strip()

        # Fast rule-based heuristics for clear cases
        # Rule 1: Code / programming standard questions -> llm_only
        code_keywords = ["write code", "how to code", "python function", "syntax", "fix bug", "def ", "class ", "import "]
        if any(kw in query_lower for kw in code_keywords) and not any(d.lower() in query_lower for d in doc_names):
            logger.info("Planner heuristic triggered: Code question -> llm_only")
            return {
                "route": "llm_only",
                "reasoning": "Standard programming/code request directed straight to LLM."
            }

        # Rule 2: Explicit request for latest/current news -> web_search
        freshness_keywords = ["latest", "recent", "today", "current news", "update", "2026", "2025", "news"]
        if any(kw in query_lower for kw in freshness_keywords) and not has_docs:
            logger.info("Planner heuristic triggered: Fresh info request -> web_search")
            return {
                "route": "web_search",
                "reasoning": "Query requests current or real-time external information."
            }

        # Rule 3: Broad research or course-content requests need external sources
        research_keywords = ["course", "curriculum", "course content", "research", "comprehensive", "complete guide"]
        if any(kw in query_lower for kw in research_keywords) and not has_docs:
            logger.info("Planner heuristic triggered: Research request -> web_search")
            return {
                "route": "web_search",
                "reasoning": "Broad research or course-content request directed to external web sources."
            }

        # Rule 4: General knowledge question with no docs -> llm_only (avoid expensive LLM route planner)
        if not has_docs and not any(d.lower() in query_lower for d in doc_names):
            if len(query_lower.split()) <= 12 and not any(kw in query_lower for kw in ["compare", "summarize", "analyze", "find in", "based on", "according to", "document", "file", "report", "pdf", "docx", "txt"]):
                logger.info("Planner heuristic triggered: General knowledge question -> llm_only")
                return {
                    "route": "llm_only",
                    "reasoning": "General knowledge request and no document context; direct LLM response is the fastest valid path."
                }

        # Rule 5: Document-specific query with uploaded docs -> rag_only (avoid LLM planner call)
        if has_docs and any(d.lower() in query_lower for d in doc_names):
            logger.info("Planner heuristic triggered: Document-specific query -> rag_only")
            return {
                "route": "rag_only",
                "reasoning": "User is explicitly asking about an uploaded document, so direct retrieval is the fastest and most grounded path."
            }

        if has_docs and any(k in query_lower for k in ["summarize", "compare", "based on", "according to", "find in", "document", "report", "file"]):
            logger.info("Planner heuristic triggered: Document-context question -> rag_only")
            return {
                "route": "rag_only",
                "reasoning": "The query clearly depends on uploaded document context; run retrieval directly."
            }

        # LLM-based intelligent classification
        formatted_prompt = PLANNER_SYSTEM_PROMPT.format(
            has_docs=has_docs,
            doc_names=", ".join(doc_names) if doc_names else "None",
            history_summary=history_summary or "None"
        )
        
        try:
            full_prompt = f"{formatted_prompt}\n\nUser Query: {query}"
            response = self.llm.invoke(full_prompt)
            content = response.content.strip()
            
            # Attempt JSON parse
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            data = json.loads(content)
            route = data.get("route", "llm_only")
            reasoning = data.get("reasoning", "LLM intent classification")
            
            # Sanity check: if no docs available, fallback from RAG to web/llm
            if not has_docs and route in ["rag_only", "hybrid_rag_llm"]:
                route = "llm_only"
                reasoning += " (Fallback to LLM as no documents are uploaded)"
                
            logger.info(f"Planner decision: Route={route}, Reason={reasoning}")
            return {"route": route, "reasoning": reasoning}

        except Exception as e:
            logger.warning(f"Planner JSON parsing or LLM invocation error: {e}. Fallback to heuristic.")
            if has_docs:
                route = "hybrid_rag_llm"
            else:
                route = "llm_only"
            return {"route": route, "reasoning": "Fallback route assignment based on document presence."}
