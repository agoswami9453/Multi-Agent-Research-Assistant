import unittest
from unittest.mock import patch

from agents.planner import PlannerAgent


class PlannerFastPathTests(unittest.TestCase):
    def test_general_question_short_circuits_without_llm(self):
        agent = PlannerAgent()

        with patch.object(agent, "llm", side_effect=AssertionError("LLM should not be called for a simple general question")):
            result = agent.plan_route("What is Python?", has_docs=False, doc_names=[])

        self.assertEqual(result["route"], "llm_only")
        self.assertIn("General knowledge", result["reasoning"])

    def test_document_query_uses_rag_without_llm(self):
        agent = PlannerAgent()

        with patch.object(agent, "llm", side_effect=AssertionError("LLM should not be called for a document-specific query")):
            result = agent.plan_route("Summarize the project report in sample_research.txt", has_docs=True, doc_names=["sample_research.txt"])

        self.assertEqual(result["route"], "rag_only")

    def test_course_content_request_uses_web_search_without_llm(self):
        agent = PlannerAgent()

        with patch.object(agent, "llm", side_effect=AssertionError("LLM should not be called for a research request")):
            result = agent.plan_route("Can you provide me the complete course content about machine learning and agentic ai", has_docs=False, doc_names=[])

        self.assertEqual(result["route"], "web_search")


if __name__ == "__main__":
    unittest.main()
