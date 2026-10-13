from typing import List, Dict, Any, Optional
from config import get_llm
from prompts import REASONING_SYSTEM_PROMPT
from utils import get_logger

logger = get_logger("ReasoningAgent")

class ReasoningAgent:
    """Agent responsible for synthesizing retrieved context, web results, and LLM logic into a grounded answer."""

    def __init__(self):
        self.llm = get_llm(temperature=0.2)

    def generate_answer(
        self,
        query: str,
        document_chunks: Optional[List[Dict[str, Any]]] = None,
        web_results: Optional[List[Dict[str, Any]]] = None,
        response_style: str = "Detailed"
    ) -> str:
        """
        Synthesizes an answer grounded in provided source contexts.
        """
        # Format document context
        doc_context_str = "None provided."
        if document_chunks:
            doc_lines = []
            for idx, chunk in enumerate(document_chunks, 1):
                src = chunk.get("source", "Document")
                page = chunk.get("page", 1)
                content = chunk.get("content", "")
                doc_lines.append(f"[{idx}] Source: {src} (Page {page})\nContent: {content}")
            doc_context_str = "\n\n".join(doc_lines)

        # Format web context
        web_context_str = "None provided."
        if web_results:
            web_lines = []
            for idx, res in enumerate(web_results, 1):
                title = res.get("title", "Web Result")
                url = res.get("url", "")
                content = res.get("content", "")
                web_lines.append(f"[{idx}] Web Title: {title}\nURL: {url}\nSnippet: {content}")
            web_context_str = "\n\n".join(web_lines)

        has_context = bool(document_chunks or web_results)
        if has_context:
            context_guidance = (
                "Use the supplied document and/or web context as the primary source. "
                "If it only partially covers the question, fill minor, non-contradictory "
                "gaps with general knowledge. Never invent specific facts, numbers, "
                "quotes, or names that are not supported by the supplied context."
            )
        else:
            context_guidance = (
                "No source context was supplied. Answer the question directly using "
                "general knowledge. Do not mention missing source material or refuse "
                "to answer for that reason."
            )

        # Build prompt
        formatted_prompt = REASONING_SYSTEM_PROMPT.format(
            query=query,
            response_style=response_style,
            document_context=doc_context_str,
            web_context=web_context_str,
            context_guidance=context_guidance
        )

        try:
            logger.info("Executing Reasoning Agent synthesis...")
            response = self.llm.invoke(formatted_prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error in Reasoning Agent: {e}")
            return f"An error occurred while synthesizing the answer: {str(e)}"
