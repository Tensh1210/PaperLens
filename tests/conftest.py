"""
Pytest configuration and fixtures for PaperLens tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =========================================================================
# Environment Setup
# =========================================================================

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("GROQ_API_KEY", "test-api-key")
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("QDRANT_PORT", "6333")
    monkeypatch.setenv("MEMORY_DB_PATH", "data/test_memory.db")
    monkeypatch.setenv("DEBUG", "true")


# =========================================================================
# Mock Fixtures
# =========================================================================

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing without API calls."""
    with patch("src.services.llm.LLMService") as mock:
        instance = MagicMock()
        instance.chat_completion.return_value = "This is a test response."
        instance.achat_completion.return_value = "This is an async test response."
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing without model loading."""
    with patch("src.services.embedding.EmbeddingService") as mock:
        instance = MagicMock()
        instance.dimension = 768
        instance.embed_text.return_value = [0.1] * 768
        instance.embed_texts.return_value = [[0.1] * 768]
        instance.embed_query.return_value = [0.1] * 768
        instance.embed_paper.return_value = [0.1] * 768
        instance.embed_papers.return_value = [[0.1] * 768]
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing without Qdrant."""
    with patch("src.services.vector_store.VectorStore") as mock:
        instance = MagicMock()
        instance.collection_name = "test_papers"
        instance.count.return_value = 100
        instance.search.return_value = []
        instance.get_collection_info.return_value = {
            "name": "test_papers",
            "points_count": 100,
            "status": "green",
        }
        mock.return_value = instance
        yield instance


# =========================================================================
# Sample Data Fixtures
# =========================================================================

@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    from src.models.paper import Paper

    return Paper(
        arxiv_id="2301.12345",
        title="Test Paper: A Study of Testing",
        abstract="This paper presents a comprehensive study of testing methodologies. We propose a novel approach that improves test coverage by 50%.",
        authors=["Alice Tester", "Bob Debugger"],
        categories=["cs.SE", "cs.LG"],
        citation_count=42,
    )


@pytest.fixture
def sample_papers():
    """Create multiple sample papers for testing."""
    from src.models.paper import Paper

    return [
        Paper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            categories=["cs.CL", "cs.LG"],
            citation_count=50000,
        ),
        Paper(
            arxiv_id="1810.04805",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
            authors=["Jacob Devlin", "Ming-Wei Chang"],
            categories=["cs.CL"],
            citation_count=40000,
        ),
        Paper(
            arxiv_id="2005.14165",
            title="Language Models are Few-Shot Learners",
            abstract="Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text.",
            authors=["Tom Brown", "Benjamin Mann"],
            categories=["cs.CL", "cs.AI"],
            citation_count=20000,
        ),
    ]


@pytest.fixture
def sample_search_results(sample_papers):
    """Create sample search results."""
    from src.models.paper import PaperSearchResult

    return [
        PaperSearchResult(paper=paper, score=0.9 - i * 0.1)
        for i, paper in enumerate(sample_papers)
    ]


# =========================================================================
# Memory Fixtures
# =========================================================================

@pytest.fixture
def working_memory():
    """Create a fresh working memory instance."""
    from src.memory.working import WorkingMemory

    return WorkingMemory(max_size=10)


@pytest.fixture
def episodic_memory_store(tmp_path):
    """Create an episodic memory store with temp database."""
    from src.memory.episodic import EpisodicMemoryStore

    db_path = tmp_path / "test_memory.db"
    return EpisodicMemoryStore(db_path=str(db_path))


@pytest.fixture
def belief_memory_store(tmp_path):
    """Create a belief memory store with temp database."""
    from src.memory.belief import BeliefMemoryStore

    db_path = tmp_path / "test_memory.db"
    return BeliefMemoryStore(db_path=str(db_path))


# =========================================================================
# Agent Fixtures
# =========================================================================

@pytest.fixture
def tool_registry():
    """Create a tool registry for testing."""
    from src.agent.tools import ToolRegistry

    return ToolRegistry()


# =========================================================================
# API Fixtures
# =========================================================================

@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    return TestClient(app)


# =========================================================================
# Cleanup
# =========================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files(tmp_path):
    """Clean up test files after each test."""
    yield
    # Any cleanup code here
