from config import get_llm
from prompts import SUMMARIZER_PROMPT_TEMPLATES
from utils import get_logger

logger = get_logger("SummarizerAgent")

class SummarizerAgent:
    """Agent responsible for formatting the final response according to user-selected response styles."""

    def __init__(self):
        self.llm = get_llm(temperature=0.2)

    def summarize(self, text: str, style: str = "Detailed") -> str:
        """
        Formats or summarizes text into the target style.
        """
        style_instruction = SUMMARIZER_PROMPT_TEMPLATES.get(style, SUMMARIZER_PROMPT_TEMPLATES["Detailed"])
        
        # If text is short or already matching style requirement
        if style == "Detailed" and len(text) > 100:
            return text

        prompt = f"""Re-format the following answer according to these instructions:
Instruction: {style_instruction}

Answer:
{text}

Formatted Output:"""

        try:
            logger.info(f"Applying response style: {style}")
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error in Summarizer Agent: {e}")
            return text
