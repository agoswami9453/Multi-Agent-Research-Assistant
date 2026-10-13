from time import perf_counter
from typing import TypedDict, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END

from agents.memory import MemoryAgent
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.web_search import WebSearchAgent
from agents.reasoning import ReasoningAgent
from agents.verifier import VerifierAgent
from agents.summarizer import SummarizerAgent
from agents.citation import CitationAgent
from utils import get_logger

logger = get_logger("GraphOrchestrator")


def _elapsed(started_at: float) -> str:
    return f"{perf_counter() - started_at:.1f}s"

# Define AgentState schema
class AgentState(TypedDict):
    session_id: str
    user_query: str
    original_query: str
    has_docs: bool
    doc_names: List[str]
    response_style: str
    route: str
    planner_reasoning: str
    retrieved_chunks: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    confidence_score: float
    verification: Dict[str, Any]
    raw_answer: str
    final_answer: str
    citations: List[Dict[str, Any]]
    sources_text: str
    execution_steps: List[str]

# Instantiate singletons for stateful components
memory_agent = MemoryAgent()
planner_agent = PlannerAgent()
retriever_agent = RetrieverAgent()
web_search_agent = WebSearchAgent()
reasoning_agent = ReasoningAgent()
verifier_agent = VerifierAgent()
summarizer_agent = SummarizerAgent()
citation_agent = CitationAgent()

# Node implementations
def memory_node(state: AgentState) -> Dict[str, Any]:
    started_at = perf_counter()
    logger.info("Executing Memory Node...")
    session_id = state["session_id"]
    query = state["user_query"]
    reformulated = memory_agent.reformulate_query_if_needed(session_id, query)

    steps = state.get("execution_steps", [])
    steps.append(f"🧠 **Memory Agent**: Processed query in {_elapsed(started_at)} (Reformulated: '{reformulated}')")

    return {
        "user_query": reformulated,
        "original_query": query,
        "execution_steps": steps
    }

def planner_node(state: AgentState) -> Dict[str, Any]:
    started_at = perf_counter()
    logger.info("Executing Planner Node...")
    query = state["user_query"]
    has_docs = state["has_docs"]
    doc_names = state["doc_names"]

    history_summary = memory_agent.get_history_summary(state["session_id"])
    plan = planner_agent.plan_route(query, has_docs, doc_names, history_summary)

    route = plan["route"]
    reasoning = plan["reasoning"]

    steps = state.get("execution_steps", [])
    steps.append(f"🎯 **Planner Agent**: Route set to `{route}` in {_elapsed(started_at)} ({reasoning})")

    return {
        "route": route,
        "planner_reasoning": reasoning,
        "execution_steps": steps
    }

def retriever_node(state: AgentState) -> Dict[str, Any]:
    started_at = perf_counter()
    logger.info("Executing Retriever Node...")
    query = state["user_query"]
    chunks, confidence = retriever_agent.search(query, top_k=4)

    steps = state.get("execution_steps", [])
    steps.append(f"📚 **Retriever Agent**: Found {len(chunks)} document chunks in {_elapsed(started_at)} (Confidence: {confidence:.2f})")

    return {
        "retrieved_chunks": chunks,
        "confidence_score": confidence,
        "execution_steps": steps
    }

def web_search_node(state: AgentState) -> Dict[str, Any]:
    started_at = perf_counter()
    logger.info("Executing Web Search Node...")
    query = state["user_query"]
    results = web_search_agent.search(query, max_results=4)

    steps = state.get("execution_steps", [])
    steps.append(f"🌐 **Web Search Agent**: Retrieved {len(results)} search results in {_elapsed(started_at)}")

    return {
        "web_results": results,
        "execution_steps": steps
    }

def hybrid_retriever_web_node(state: AgentState) -> Dict[str, Any]:
    started_at = perf_counter()
    logger.info("Executing Hybrid Retriever + Web Search Node...")
    query = state["user_query"]

    # Doc retrieval (FAISS) and web search (Tavily) don't depend on each other,
    # so run them concurrently instead of back-to-back to cut wall-clock time.
    with ThreadPoolExecutor(max_workers=2) as executor:
        retriever_future = executor.submit(retriever_agent.search, query, 4)
        web_future = executor.submit(web_search_agent.search, query, 4)
        chunks, confidence = retriever_future.result()
        results = web_future.result()

    steps = state.get("execution_steps", [])
    steps.append(f"⚡ **Hybrid Agent**: Retrieved {len(chunks)} doc chunks and {len(results)} web results in {_elapsed(started_at)} (parallelized)")

    return {
        "retrieved_chunks": chunks,
        "confidence_score": confidence,
        "web_results": results,
        "execution_steps": steps
    }

def reasoning_node(state: AgentState) -> Dict[str, Any]:
    started_at = perf_counter()
    logger.info("Executing Reasoning Node...")
    query = state["user_query"]
    chunks = state.get("retrieved_chunks", [])
    web_results = state.get("web_results", [])
    style = state.get("response_style", "Detailed")

    answer = reasoning_agent.generate_answer(
        query=query,
        document_chunks=chunks,
        web_results=web_results,
        response_style=style
    )

    steps = state.get("execution_steps", [])
    steps.append(f"💡 **Reasoning Agent**: Synthesized grounded answer in {_elapsed(started_at)}")

    return {
        "raw_answer": answer,
        "execution_steps": steps
    }

def verifier_node(state: AgentState) -> Dict[str, Any]:
    started_at = perf_counter()
    logger.info("Executing Verifier Node...")
    query = state["user_query"]
    answer = state["raw_answer"]
    chunks = state.get("retrieved_chunks", [])
    confidence = state.get("confidence_score", 1.0)

    verification = verifier_agent.verify_answer(query, answer, chunks, confidence)

    steps = state.get("execution_steps", [])
    conflict_flag = "⚠️ Conflict detected!" if verification.get("conflict_detected") else "✅ Verified"
    steps.append(f"🔍 **Verifier Agent**: {conflict_flag} in {_elapsed(started_at)}")

    # If conflict detected, append warning note to answer
    if verification.get("conflict_detected") and verification.get("conflict_details"):
        answer += f"\n\n> ⚠️ **Verification Warning**: {verification.get('conflict_details')}"

    return {
        "verification": verification,
        "raw_answer": answer,
        "execution_steps": steps
    }

def summarizer_node(state: AgentState) -> Dict[str, Any]:
    started_at = perf_counter()
    logger.info("Executing Summarizer Node...")
    answer = state["raw_answer"]
    style = state.get("response_style", "Detailed")

    formatted_answer = summarizer_agent.summarize(answer, style=style)

    steps = state.get("execution_steps", [])
    steps.append(f"✍️ **Summarizer Agent**: Applied '{style}' style format in {_elapsed(started_at)}")

    return {
        "final_answer": formatted_answer,
        "execution_steps": steps
    }

def citation_node(state: AgentState) -> Dict[str, Any]:
    started_at = perf_counter()
    logger.info("Executing Citation Node...")
    chunks = state.get("retrieved_chunks", [])
    web_results = state.get("web_results", [])
    answer = state["final_answer"]

    citations, footer = citation_agent.format_citations(chunks, web_results)
    if footer:
        answer += footer

    steps = state.get("execution_steps", [])
    steps.append(f"📌 **Citation Agent**: Attached {len(citations)} citations in {_elapsed(started_at)}")

    return {
        "citations": citations,
        "sources_text": footer,
        "final_answer": answer,
        "execution_steps": steps
    }

# Routing function
def route_planner_decision(state: AgentState) -> str:
    route = state.get("route", "llm_only")
    if route == "rag_only" or route == "hybrid_rag_llm":
        return "retriever"
    elif route == "web_search":
        return "web_search"
    elif route == "hybrid_rag_web":
        return "hybrid"
    else:
        return "reasoning"

# Build LangGraph Workflow
workflow = StateGraph(AgentState)
# Add Nodes
workflow.add_node("memory", memory_node)
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("hybrid", hybrid_retriever_web_node)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("citation", citation_node)

# Add Edges
workflow.set_entry_point("memory")
workflow.add_edge("memory", "planner")

workflow.add_conditional_edges(
    "planner",
    route_planner_decision,
    {
        "retriever": "retriever",
        "web_search": "web_search",
        "hybrid": "hybrid",
        "reasoning": "reasoning"
    }
)

workflow.add_edge("retriever", "reasoning")
workflow.add_edge("web_search", "reasoning")
workflow.add_edge("hybrid", "reasoning")

workflow.add_edge("reasoning", "verifier")
workflow.add_edge("verifier", "summarizer")
workflow.add_edge("summarizer", "citation")
workflow.add_edge("citation", END)

# Compile graph
graph_app = workflow.compile()

def run_research_assistant_graph(
    session_id: str,
    user_query: str,
    has_docs: bool,
    doc_names: List[str],
    response_style: str = "Detailed"
) -> Dict[str, Any]:
    """
    Executes the multi-agent graph with initial state and returns final result state.
    """
    initial_state: AgentState = {
        "session_id": session_id,
        "user_query": user_query,
        "original_query": user_query,
        "has_docs": has_docs,
        "doc_names": doc_names,
        "response_style": response_style,
        "route": "llm_only",
        "planner_reasoning": "",
        "retrieved_chunks": [],
        "web_results": [],
        "confidence_score": 1.0,
        "verification": {},
        "raw_answer": "",
        "final_answer": "",
        "citations": [],
        "sources_text": "",
        "execution_steps": []
    }

    result_state = graph_app.invoke(initial_state)
    return result_state
