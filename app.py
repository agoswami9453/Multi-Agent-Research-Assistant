import os
import uuid
import tempfile
import streamlit as st
from pathlib import Path

from config import UPLOADS_DIR
from database import add_message, get_chat_history, get_latest_chat_session_id, clear_session_history
from graph import run_research_assistant_graph, retriever_agent
from utils import export_to_markdown, export_to_docx, export_to_pdf, get_logger

logger = get_logger("StreamlitApp")

# Page Config
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode & Sleek Premium Aesthetics
st.markdown("""
<style>
    /* Dark theme base styles */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* Execution Route Badges */
    .badge-route {
        background-color: #2563EB;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-right: 8px;
    }
    
    .badge-confidence-high {
        background-color: #059669;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-confidence-medium {
        background-color: #D97706;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-confidence-low {
        background-color: #DC2626;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Thinking Expanders */
    .stExpander {
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        background-color: #1E293B !important;
    }
    
    /* Sources Box */
    .sources-box {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 12px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = get_latest_chat_session_id() or str(uuid.uuid4())

if "messages" not in st.session_state:
    # Load past messages from SQLite database
    past_history = get_chat_history(st.session_state.session_id)
    if past_history:
        st.session_state.messages = past_history
    else:
        st.session_state.messages = []
elif not st.session_state.messages and not st.session_state.get("history_cleared", False):
    # Recover a conversation when Streamlit reconnects with a stale, empty session.
    latest_session_id = get_latest_chat_session_id()
    if latest_session_id:
        past_history = get_chat_history(latest_session_id)
        if past_history:
            st.session_state.session_id = latest_session_id
            st.session_state.messages = past_history

# Sidebar Controls
with st.sidebar:
    st.title("🤖 Agent Settings")
    
    # 1. Document Uploader
    st.subheader("📄 Document Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("📥 Process & Index Documents", use_container_width=True):
            with st.spinner("Processing & Indexing Documents into FAISS..."):
                saved_paths = []
                for uploaded_file in uploaded_files:
                    file_path = UPLOADS_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(str(file_path))
                
                result = retriever_agent.ingest_documents(saved_paths)
                if result.get("status") == "success":
                    st.success(f"Indexed {result.get('chunks_count')} text chunks successfully!")
                else:
                    st.error(f"Error: {result.get('message')}")
                    
    # Vector DB Status
    v_status = retriever_agent.get_status()
    st.markdown("---")
    st.subheader("📊 Vector DB Status")
    col1, col2 = st.columns(2)
    col1.metric("Documents", v_status["total_documents"])
    col2.metric("Chunks", v_status["total_chunks"])
    
    if v_status["documents"]:
        st.caption("Active Documents:")
        for doc in v_status["documents"]:
            st.markdown(f"- 📄 `{doc}`")
            
        if st.button("🗑️ Clear Vector Database", use_container_width=True):
            retriever_agent.clear_index()
            st.toast("Vector Database cleared!")
            st.rerun()

    # 2. Response Style Selector
    st.markdown("---")
    st.subheader("⚙️ Agent Response Style")
    response_style = st.selectbox(
        "Select output format:",
        ["Detailed", "Brief", "Bullet Points", "Beginner (ELI5)", "Professional", "Research"]
    )

    # 3. Exports
    st.markdown("---")
    st.subheader("📥 Export Conversation")
    has_messages = bool(st.session_state.messages)
    if not has_messages:
        st.caption("No conversation to export yet.")

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    # Markdown Export
    md_data = export_to_markdown(st.session_state.messages)
    exp_col1.download_button(
        label="MD",
        data=md_data,
        file_name="conversation_export.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not has_messages
    )

    # DOCX Export
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
        docx_path = tmp_docx.name
    try:
        export_to_docx(st.session_state.messages, docx_path)
        with open(docx_path, "rb") as f:
            docx_bytes = f.read()
        exp_col2.download_button(
            label="DOCX",
            data=docx_bytes,
            file_name="conversation_export.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            disabled=not has_messages
        )
    except Exception:
        logger.exception("DOCX conversation export failed")
        st.error("DOCX export failed. Check the application logs for details.")
        exp_col2.button("DOCX", disabled=True)
    finally:
        try:
            os.unlink(docx_path)
        except OSError:
            logger.warning("Could not remove temporary DOCX export file: %s", docx_path)

    # PDF Export
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf_path = tmp_pdf.name
    try:
        export_to_pdf(st.session_state.messages, pdf_path)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        exp_col3.download_button(
            label="PDF",
            data=pdf_bytes,
            file_name="conversation_export.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=not has_messages
        )
    except Exception:
        logger.exception("PDF conversation export failed")
        st.error("PDF export failed. Check the application logs for details.")
        exp_col3.button("PDF", disabled=True)
    finally:
        try:
            os.unlink(pdf_path)
        except OSError:
            logger.warning("Could not remove temporary PDF export file: %s", pdf_path)

    # Clear History
    if st.button("🧹 Clear Chat History", use_container_width=True):
        clear_session_history(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.history_cleared = True
        st.rerun()


# Main Application Interface
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.2rem; color: #60A5FA;">🤖 Enterprise Multi-Agent Research Assistant</h1>
    <p style="margin: 6px 0 0 0; color: #94A3B8;">Powered by LangGraph • LangChain • Ollama • RAG • Tavily Web Search • SQLite</p>
</div>
""", unsafe_allow_html=True)


# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Display metadata badges if present
            route = message.get("route", "")
            confidence = message.get("confidence")
            
            if route:
                badge_html = f'<span class="badge-route">Route: {route.upper()}</span>'
                if confidence is not None:
                    if confidence >= 0.7:
                        badge_html += f'<span class="badge-confidence-high">Confidence: {confidence*100:.0f}%</span>'
                    elif confidence >= 0.4:
                        badge_html += f'<span class="badge-confidence-medium">Confidence: {confidence*100:.0f}%</span>'
                    else:
                        badge_html += f'<span class="badge-confidence-low">Confidence: {confidence*100:.0f}%</span>'
                st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown("<br/>", unsafe_allow_html=True)

        st.markdown(message["content"])


# User Input Box
if prompt := st.chat_input("Ask a question, request document synthesis, or ask for latest research..."):
    st.session_state.history_cleared = False
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    add_message(st.session_state.session_id, "user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response Generation
    with st.chat_message("assistant"):
        thinking_container = st.empty()
        response_container = st.empty()
        
        v_status = retriever_agent.get_status()
        has_docs = v_status["has_index"] and v_status["total_documents"] > 0
        doc_names = v_status["documents"]

        with st.status("🧠 Agents Thinking & Executing Plan...", expanded=True) as status:
            st.write("🔄 Initializing LangGraph state graph...")
            
            # Execute Multi-Agent Graph
            result_state = run_research_assistant_graph(
                session_id=st.session_state.session_id,
                user_query=prompt,
                has_docs=has_docs,
                doc_names=doc_names,
                response_style=response_style
            )
            
            # Display agent execution steps inside status
            for step in result_state.get("execution_steps", []):
                st.write(step)
                
            status.update(label="✅ Execution Complete", state="complete", expanded=False)

        # Extract graph output
        final_answer = result_state.get("final_answer", "")
        route = result_state.get("route", "llm_only")
        confidence = result_state.get("confidence_score", 1.0)
        citations = result_state.get("citations", [])

        # Display route & confidence badges
        badge_html = f'<span class="badge-route">Route: {route.upper()}</span>'
        if confidence is not None:
            if confidence >= 0.7:
                badge_html += f'<span class="badge-confidence-high">Confidence: {confidence*100:.0f}%</span>'
            elif confidence >= 0.4:
                badge_html += f'<span class="badge-confidence-medium">Confidence: {confidence*100:.0f}%</span>'
            else:
                badge_html += f'<span class="badge-confidence-low">Confidence: {confidence*100:.0f}%</span>'
        
        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(final_answer)

        # Append assistant message to session state & SQLite DB
        assistant_msg = {
            "role": "assistant",
            "content": final_answer,
            "route": route,
            "confidence": confidence,
            "sources": citations
        }
        st.session_state.messages.append(assistant_msg)
        add_message(
            session_id=st.session_state.session_id,
            role="assistant",
            content=final_answer,
            route=route,
            confidence=confidence,
            sources=citations
        )
