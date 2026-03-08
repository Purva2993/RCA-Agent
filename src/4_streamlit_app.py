"""
4_streamlit_app.py

WHAT THIS FILE DOES:
Streamlit web UI that wraps both the Simple RAG pipeline
and the ReAct Agent into a clean interface.

RUN WITH:
  streamlit run src/4_streamlit_app.py

REQUIRES:
  - data/chroma_db/ must exist (run 1_ingest_logs.py first)
  - GROQ_API_KEY in .env file or entered in the sidebar
"""

import os
from pdb import main
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add src/ to path so we can import our other files
sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import load_retriever, load_llm, build_rag_chain, query_rag
from agent       import build_agent, run_agent

st.set_page_config(
    page_title = "RCA Agent — Log Analysis",
    page_icon  = "🔍",
    layout     = "wide",
)

# Custom CSS for cleaner styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    .step-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .tool-badge {
        background: #dbeafe;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_components(api_key: str):
    """
    Loads RAG components once and caches them.
    Without caching, these would reload on every interaction — 
    making the app slow.

    @st.cache_resource means:
    'load this once, keep it in memory, reuse it every time'
    """
    os.environ["GROQ_API_KEY"] = api_key
    retriever, vectorstore     = load_retriever()
    llm                        = load_llm()
    chain                      = build_rag_chain(retriever, llm)
    return retriever, chain


@st.cache_resource
def get_agent_component(api_key: str):
    """
    Loads the ReAct agent once and caches it.
    Same reason as above — agent setup takes time.
    """
    os.environ["GROQ_API_KEY"] = api_key
    return build_agent()

def render_sidebar():
    """
    Renders the left sidebar with configuration options.
    Returns the api_key and selected mode.
    """
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.divider()

        # API Key input
        st.subheader("🔑 Groq API Key")
        api_key = st.text_input(
            "Enter your Groq API key",
            value       = os.getenv("GROQ_API_KEY", ""),
            type        = "password",
            help        = "Get a free key at console.groq.com",
            placeholder = "gsk_...",
        )

        st.divider()

        # Mode selector
        st.subheader("🧠 Analysis Mode")
        mode = st.radio(
            "Choose mode:",
            options     = ["Simple RAG", "ReAct Agent"],
            index       = 0,
            help        = (
                "Simple RAG: one search, fast answer.\\n"
                "ReAct Agent: multi-step reasoning, more thorough."
            ),
        )

        st.divider()

        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "Why did the payment service crash at 2am?",
            "What caused the database connection errors?",
            "Was there a deployment before the incident?",
            "What is the timeline of the failure?",
            "How was the service recovered?",
        ]
        for q in sample_questions:
            st.markdown(f"• {q}")

        st.divider()

        # Log file status
        st.subheader("📄 Log Files")
        log_dir = "logs"
        if os.path.exists(log_dir):
            for f in sorted(os.listdir(log_dir)):
                if f.endswith(".log"):
                    size = os.path.getsize(
                        os.path.join(log_dir, f)
                    ) / 1024
                    st.markdown(f"✅ `{f}` ({size:.1f} KB)")
        else:
            st.warning("No logs/ folder found.")

        # Vector store status
        st.subheader("🗃️ Vector Store")
        if os.path.exists("data/chroma_db"):
            st.success("ChromaDB ready")
        else:
            st.error("Run 1_ingest_logs.py first!")

    return api_key, mode

def display_rag_results(result: dict):
    """
    Displays the Simple RAG results in two columns:
    left column = retrieved chunks (the evidence)
    right column = the LLM's answer
    """
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📄 Retrieved Log Chunks")
        st.caption(
            f"{len(result['retrieved_chunks'])} most relevant chunks found"
        )

        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            with st.expander(
                f"Chunk {i} — {chunk['source']}", expanded=(i == 1)
            ):
                st.code(chunk["text"], language="text")

    with col2:
        st.subheader("🤖 Root Cause Analysis")
        st.markdown(result["answer"])

def display_agent_results(result: dict):
    """
    Displays the ReAct agent results in three sections:
    1. Reasoning trace — every thought and tool call
    2. Summary stats
    3. Final answer
    """
    # Summary stats at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tool Calls", result["num_steps"])
    with col2:
        st.metric("Tools Used", len(set(result["tools_used"])))
    with col3:
        status = "✅ Complete" if result["final_answer"] else "⚠️ Incomplete"
        st.metric("Status", status)

    st.divider()

    # Reasoning trace
    st.subheader("🔍 Agent Reasoning Trace")
    st.caption("Every step the agent took to reach its conclusion")

    for i, step in enumerate(result["steps"], 1):
        with st.expander(
            f"Step {i} — {step['tool']}", expanded=True
        ):
            st.markdown(
                f'<span class="tool-badge">🔧 {step["tool"]}</span>',
                unsafe_allow_html=True
            )

            st.markdown("**Input to tool:**")
            st.code(str(step["input"]), language="text")

            st.markdown("**Tool returned:**")
            st.code(str(step["observation"])[:800], language="text")

    st.divider()

    # Final answer
    st.subheader("🤖 Final Root Cause Analysis")
    st.markdown(result["final_answer"])

def main():
    # Header
    st.markdown(
        '<p class="main-header">🔍 RCA Agent — Log Analysis</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        "Ask questions about your system logs in plain English. "
        "Powered by RAG + LLM."
    )
    st.divider()

    # Render sidebar and get config
    api_key, mode = render_sidebar()

    # Check API key
    if not api_key:
        st.warning(
            "⚠️ Please enter your Groq API key in the sidebar to get started."
        )
        st.stop()

    # Check ChromaDB exists
    if not os.path.exists("data/chroma_db"):
        st.error(
            "❌ Vector store not found. "
            "Please run `python src/1_ingest_logs.py` first."
        )
        st.stop()

    # Question input
    st.subheader("💬 Ask A Question")

    question = st.text_area(
        "Type your question about the logs:",
        placeholder = (
            "e.g. Why did the payment service crash at 2am? "
            "What caused the database errors?"
        ),
        height = 80,
    )

    # Mode description
    if mode == "Simple RAG":
        st.info(
            "**Simple RAG mode:** One similarity search → LLM generates answer. "
            "Fast. Good for focused questions."
        )
    else:
        st.info(
            "**ReAct Agent mode:** Multi-step reasoning with tools. "
            "Checks logs + metrics + deployments. "
            "Slower but more thorough."
        )

    # Submit button
    submit = st.button(
        "🔍 Analyze",
        type = "primary",
        use_container_width = True,
    )

    # Run analysis when button clicked
    if submit and question.strip():
        with st.spinner(
            "Searching logs and generating analysis..."
            if mode == "Simple RAG"
            else "Agent is investigating... (may take 30-60 seconds)"
        ):
            try:
                if mode == "Simple RAG":
                    retriever, chain = get_rag_components(api_key)
                    result           = query_rag(question, chain, retriever)
                    display_rag_results(result)

                else:
                    agent  = get_agent_component(api_key)
                    result = run_agent(question, agent)
                    display_agent_results(result)

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info(
                    "Common fixes:\\n"
                    "- Check your Groq API key is correct\\n"
                    "- Make sure 1_ingest_logs.py has been run\\n"
                    "- Check your internet connection"
                )

    elif submit and not question.strip():
        st.warning("Please type a question before clicking Analyze.")


if __name__ == "__main__":
    main()