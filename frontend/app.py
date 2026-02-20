"""
Streamlit frontend for PaperLens.

Provides an interactive UI for:
- Chat-based paper search
- Paper comparison
- Search history and preferences
"""

import os
from typing import Any
from uuid import uuid4

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

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

def get_api_client() -> httpx.Client:
    """Get a reusable HTTP client for the PaperLens API."""
    return httpx.Client(base_url=API_BASE_URL, timeout=120.0)


def run_agent_query(query: str, session_id: str) -> dict[str, Any]:
    """Run a query through the PaperLens API."""
    try:
        with get_api_client() as client:
            response = client.post(
                "/api/chat",
                json={"message": query, "session_id": session_id},
            )
            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "response": data["response"],
                "papers": data.get("papers", []),
                "steps": data.get("steps_taken", 0),
            }
    except httpx.ConnectError:
        return {"success": False, "error": "Cannot connect to API. Is the backend running?"}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"API error: {e.response.status_code}"}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def format_paper_card(paper: dict[str, Any]) -> str:
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
        with get_api_client() as client:
            resp = client.get("/health")
            if resp.status_code == 200:
                health = resp.json()
                memory_stats = health.get("memory", {})
                semantic = memory_stats.get("semantic", {})
                st.metric("Papers indexed", semantic.get("total_papers", 0))
                st.metric(
                    "Active sessions",
                    memory_stats.get("working_sessions", 0),
                )
            else:
                st.text("Stats unavailable")
    except Exception:
        st.text("API not reachable")


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
