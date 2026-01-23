"""
Streamlit frontend for PaperLens.

Provides an interactive UI for:
- Chat-based paper search
- Paper comparison
- Search history and preferences
"""

import os
import sys
from uuid import uuid4

import streamlit as st

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agent.agent import PaperLensAgent, get_agent
from src.memory.manager import get_memory_manager


# =========================================================================
# Page Configuration
# =========================================================================

st.set_page_config(
    page_title="PaperLens",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
# Session State Initialization
# =========================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "papers_viewed" not in st.session_state:
    st.session_state.papers_viewed = []


# =========================================================================
# Helper Functions
# =========================================================================

@st.cache_resource
def get_cached_agent():
    """Get the agent (cached for performance)."""
    return get_agent()


@st.cache_resource
def get_cached_memory_manager():
    """Get the memory manager (cached for performance)."""
    return get_memory_manager()


def run_agent_query(query: str, session_id: str):
    """Run a query through the agent."""
    agent = get_cached_agent()
    try:
        result = agent.run(query, session_id=session_id)
        return {
            "success": True,
            "response": result.response,
            "papers": result.papers,
            "steps": len(result.steps),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def format_paper_card(paper: dict) -> str:
    """Format a paper as a markdown card."""
    return f"""
**{paper.get('title', 'Unknown Title')}**

- **ArXiv ID**: `{paper.get('arxiv_id', 'N/A')}`
- **Year**: {paper.get('year', 'N/A')}
- **Score**: {paper.get('score', 0):.3f}

{paper.get('abstract', 'No abstract available.')[:300]}...

[📄 PDF]({paper.get('pdf_url', '#')}) | [🔗 ArXiv]({paper.get('arxiv_url', '#')})

---
"""


# =========================================================================
# Sidebar
# =========================================================================

with st.sidebar:
    st.title("📚 PaperLens")
    st.caption("Agentic RAG Paper Search")

    st.divider()

    # Session info
    st.subheader("Session")
    st.text(f"ID: {st.session_state.session_id[:8]}...")

    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid4())
        st.session_state.messages = []
        st.session_state.papers_viewed = []
        st.rerun()

    st.divider()

    # Quick search
    st.subheader("Quick Search")
    quick_query = st.text_input(
        "Search papers",
        placeholder="e.g., transformer attention",
        key="quick_search",
    )

    col1, col2 = st.columns(2)
    with col1:
        year_from = st.number_input("From year", min_value=2000, max_value=2030, value=2020)
    with col2:
        year_to = st.number_input("To year", min_value=2000, max_value=2030, value=2025)

    if st.button("🔍 Search", use_container_width=True):
        if quick_query:
            search_msg = f"Find papers about {quick_query}"
            if year_from != 2020 or year_to != 2025:
                search_msg += f" from {year_from} to {year_to}"
            st.session_state.messages.append({"role": "user", "content": search_msg})
            st.rerun()

    st.divider()

    # Example queries
    st.subheader("Try asking...")
    examples = [
        "Find papers about vision transformers",
        "Compare BERT and GPT architectures",
        "Summarize the attention is all you need paper",
        "What papers have I looked at recently?",
    ]

    for example in examples:
        if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()

    st.divider()

    # Stats
    st.subheader("Stats")
    try:
        memory_manager = get_cached_memory_manager()
        stats = memory_manager.semantic.get_stats()
        st.metric("Papers indexed", stats.get("total_papers", 0))
        st.metric("Active sessions", len(memory_manager.working.list_sessions()))
    except Exception:
        st.text("Stats unavailable")


# =========================================================================
# Main Chat Interface
# =========================================================================

st.title("🔬 PaperLens Chat")
st.caption("Ask questions about ML papers, compare research, or explore the literature.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show papers if this is an assistant message with papers
        if message["role"] == "assistant" and message.get("papers"):
            with st.expander(f"📄 Referenced papers ({len(message['papers'])})"):
                for paper_id in message["papers"][:5]:
                    st.code(paper_id)

# Chat input
if prompt := st.chat_input("Ask about papers..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent_query(prompt, st.session_state.session_id)

            if result["success"]:
                st.markdown(result["response"])

                # Store papers
                papers = result.get("papers", [])
                if papers:
                    st.session_state.papers_viewed.extend(papers)
                    with st.expander(f"📄 Referenced papers ({len(papers)})"):
                        for paper_id in papers[:5]:
                            st.code(paper_id)

                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "papers": papers,
                })
            else:
                error_msg = f"Sorry, I encountered an error: {result['error']}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


# =========================================================================
# Footer
# =========================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("Built with Streamlit + FastAPI")

with col2:
    st.caption("Powered by Groq + SPECTER2")

with col3:
    st.caption(f"Session: {st.session_state.session_id[:8]}...")


# =========================================================================
# Run info
# =========================================================================

if __name__ == "__main__":
    # This allows running with: streamlit run frontend/app.py
    pass
