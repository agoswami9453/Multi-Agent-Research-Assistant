from typing import List, Dict, Any, Tuple
from utils import get_logger

logger = get_logger("CitationAgent")

class CitationAgent:
    """Agent responsible for compiling and attaching document and web citations."""

    def format_citations(
        self,
        document_chunks: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Extracts citations into a structured list and generates a formatted Markdown citations footer.
        """
        citations = []
        citation_text_lines = []

        # Process document chunks
        seen_docs = set()
        for chunk in document_chunks:
            source = chunk.get("source", "Document")
            page = chunk.get("page", 1)
            key = f"{source}_p{page}"
            if key not in seen_docs:
                seen_docs.add(key)
                item = {
                    "type": "document",
                    "source": source,
                    "page": page,
                    "title": f"📄 Document: {source} (Page {page})"
                }
                citations.append(item)
                citation_text_lines.append(f"- 📄 **{source}** (Page {page})")

        # Process web results
        seen_urls = set()
        for res in web_results:
            url = res.get("url", "")
            title = res.get("title", "Web Source")
            if url and url not in seen_urls:
                seen_urls.add(url)
                item = {
                    "type": "web",
                    "title": title,
                    "url": url,
                    "source": f"🌐 {title}"
                }
                citations.append(item)
                citation_text_lines.append(f"- 🌐 [{title}]({url})")

        formatted_footer = ""
        if citation_text_lines:
            formatted_footer = "\n\n---\n**Sources & References:**\n" + "\n".join(citation_text_lines)

        logger.info(f"Compiled {len(citations)} citations.")
        return citations, formatted_footer
