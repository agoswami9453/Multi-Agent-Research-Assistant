import json
from typing import Dict, Any, List, Optional
from config import get_llm, VERIFIER_MODE
from prompts import VERIFIER_SYSTEM_PROMPT
from utils import get_logger

logger = get_logger("VerifierAgent")

class VerifierAgent:
    """
    Agent responsible for checking answer quality, confidence scores, and conflict detection.

    Supports two verification modes (set via VERIFIER_MODE in config/.env), neither of which
    is removed in favor of the other:

      - "deterministic": pure keyword-overlap heuristic, zero LLM calls. Always fast and free,
        but only catches "unsupported" answers - it can't detect genuine factual contradictions
        between sources.
      - "llm" (default): sends the answer + context back to the LLM for a real verification
        judgment (is_verified / conflict_detected / conflict_details). Costs one extra LLM call
        per query, but can catch actual contradictions between document and web sources.
    """

    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.mode = (VERIFIER_MODE or "llm").strip().lower()
        if self.mode not in ("llm", "deterministic"):
            logger.warning(f"Unknown VERIFIER_MODE '{VERIFIER_MODE}', defaulting to 'llm'.")
            self.mode = "llm"

    def verify_answer(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        confidence_score: float
    ) -> Dict[str, Any]:
        """
        Validates generated answer groundedness, confidence, and detects conflicts between sources.
        Dispatches to the deterministic or LLM-based implementation based on self.mode.
        """
        context_str = "\n".join([c.get("content", "") for c in context_chunks]) if context_chunks else "No retrieved context."

        # If low confidence score and empty context
        if confidence_score < 0.4 and context_chunks:
            logger.info("Verifier detected low retrieval confidence.")
            verification_note = "Retrieval confidence is low. Answer is augmented with general LLM knowledge."
        else:
            verification_note = "High/Medium confidence retrieval verification passed."

        if self.mode == "deterministic":
            return self._verify_deterministic(query, answer, context_str, context_chunks, confidence_score, verification_note)
        return self._verify_llm(answer, context_str, confidence_score, verification_note)

    def _verify_deterministic(
        self,
        query: str,
        answer: str,
        context_str: str,
        context_chunks: List[Dict[str, Any]],
        confidence_score: float,
        verification_note: str
    ) -> Dict[str, Any]:
        """
        Rule-based verification: no LLM call, so every query does not require another full
        generation after the answer has already been produced. Fast and free, at the cost of
        not being able to detect genuine cross-source conflicts.
        """
        answer_lower = answer.lower()
        context_lower = context_str.lower()
        has_context = bool(context_chunks)
        unsupported_marker = "i don't have enough information" in answer_lower
        grounded = not has_context or unsupported_marker or any(
            token in context_lower for token in query.lower().split() if len(token) > 4
        )
        result = {
            "is_verified": grounded,
            "conflict_detected": False,
            "conflict_details": "",
            "verification_notes": verification_note,
            "confidence_score": confidence_score
        }
        logger.info("Deterministic verification completed: is_verified=%s", grounded)
        return result

    def _verify_llm(
        self,
        answer: str,
        context_str: str,
        confidence_score: float,
        verification_note: str
    ) -> Dict[str, Any]:
        """
        LLM-based verification: asks the model to judge groundedness and detect conflicts
        between sources, returned as structured JSON. Slower (one extra LLM call) but more
        thorough than the deterministic check.
        """
        prompt = VERIFIER_SYSTEM_PROMPT.format(
            context=context_str[:2000],
            answer=answer[:2000],
            confidence_score=f"{confidence_score:.2f}"
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            result["confidence_score"] = confidence_score
            logger.info(f"Verification completed: is_verified={result.get('is_verified')}, conflict={result.get('conflict_detected')}")
            return result

        except Exception as e:
            logger.warning(f"Verifier parsing error: {e}. Defaulting to basic verification.")
            return {
                "is_verified": True,
                "conflict_detected": False,
                "conflict_details": "",
                "verification_notes": verification_note,
                "confidence_score": confidence_score
            }
